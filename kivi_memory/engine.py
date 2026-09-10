"""The engine: one object that wires the pieces together and holds the API.

Everything the interface, the CLI and the evaluator call goes through
`KiviMemoryEngine`. It owns the database handle, resolves each pluggable
backend once at construction, and coordinates the two paths:

    write   ingest_record / dictate
              -> extractor (sensor)   proposes Candidates
              -> controller.py        decides ADD / UPDATE / REINFORCE / REJECT

    read    ask
              -> intent.py            routes the question
              -> retrieval.py         scores and selects memories
              -> answerer.py          composes, with citations enforced here

The deliberate split is that no model touches the middle column. Sensors and
answer composition are pluggable; admission, conflict resolution,
reinforcement, decay and abstention are arithmetic, defined in `controller.py`,
`retrieval.py` and `config.KiviConfig`.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from .answerer import get_answerer
from .config import KiviConfig
from .controller import MemoryController
from .core import (
    Candidate,
    normalize_text,
    parse_dt,
    snippet,
    stable_id,
    to_second_person,
    utc_now,
)
from .db import connect, migrate
from .embeddings import get_embedder
from .extractors import get_extractor
from .import inspection
from .intent import get_router
from .llm import get_client
from .nli import get_nli
from .query import QueryService


class KiviMemoryEngine:
    def __init__(self, db_path: str = "data/kivi.db", config: KiviConfig | None = None) -> None:
        self.db_path = db_path
        self.config = config or KiviConfig.from_env()

        # One client shared by every model-backed component, so health checks,
        # retries and the response cache are shared too.
        needs_model = not (
            self.config.extractor == "rule"
            and self.config.nli in ("off", "cross-encoder")
            and self.config.retrieval == "lexical"
            and self.config.answerer == "template"
        )
        self.client = get_client(self.config) if needs_model else None

        self.extractor = get_extractor(self.config, self.client)
        self.nli = get_nli(self.config, self.client)
        self.embedder = get_embedder(self.config, self.client)
        self.answerer = get_answerer(self.config, self.client)
        self.router = get_router(self.config, self.client, self.embedder)
        self.controller = MemoryController(
            self.config, self.nli, self.embedder,
            self._embedding_for_memory, self._store_embedding)
        self.query = QueryService(self)
        with connect(db_path) as conn:
            migrate(conn)

    # -- introspection ------------------------------------------------------

    def backends(self) -> dict[str, Any]:
        describe = lambda obj, default: obj.describe() if hasattr(obj, "describe") else default  # noqa: E731
        return {
            "extractor": describe(self.extractor, {"backend": "rule"}),
            "nli": self.nli.describe(),
            "embedder": self.embedder.describe(),
            "answerer": self.answerer.describe(),
            "router": self.router.describe(),
            "config": self.config.describe(),
            "ollama": self.client.doctor() if self.client else {"reachable": False, "note": "not required in this mode"},
        }

    def usage(self) -> dict[str, Any]:
        if self.client is None:
            return {
                "chat_calls": 0,
                "embed_calls": 0,
                "external_api_calls": 0,
                "estimated_cost_usd": 0.0,
                "note": "Deterministic mode: no model calls at all.",
            }
        return self.client.usage()

    # -- embeddings ---------------------------------------------------------

    def _embedding_model_name(self) -> str:
        return str(self.embedder.describe().get("backend", "unknown"))

    def _store_embedding(self, conn, owner_kind: str, owner_id: str, text: str) -> list[float]:
        model = self._embedding_model_name()
        vector = self.embedder.embed_one(text)
        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings(owner_kind, owner_id, model, dim, vector_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (owner_kind, owner_id, model, len(vector), json.dumps(vector), utc_now()),
        )
        return vector

    def _load_embeddings(self, conn, owner_kind: str) -> dict[str, list[float]]:
        model = self._embedding_model_name()
        rows = conn.execute(
            "SELECT owner_id, vector_json FROM embeddings WHERE owner_kind=? AND model=?",
            (owner_kind, model),
        ).fetchall()
        return {row["owner_id"]: json.loads(row["vector_json"]) for row in rows}

    def _embedding_for_memory(self, conn, row) -> list[float]:
        model = self._embedding_model_name()
        stored = conn.execute(
            "SELECT vector_json FROM embeddings WHERE owner_kind='memory' AND owner_id=? AND model=?",
            (row["id"], model),
        ).fetchone()
        if stored:
            return json.loads(stored["vector_json"])
        text = " | ".join([
            row["canonical_text"], row["memory_type"], row["predicate"], row["object"], row["scope"],
        ])
        return self._store_embedding(conn, "memory", row["id"], text)

    def ingest_report(self, result: dict[str, Any], source: str = "") -> dict[str, Any]:
        """Everything measurable about one ingest, in one object.

        The assignment asks for latency, database growth, model usage and cost
        "wherever they matter". An ingest is where they matter most and it is
        the run nobody repeats - a 500-record model-backed pass can take hours,
        so the numbers have to survive the terminal.
        """
        usage = self.usage()
        calls = int(usage.get("chat_calls", 0)) + int(usage.get("embed_calls", 0))
        model_ms = float(usage.get("total_latency_ms", 0.0))
        processed = max(1, int(result.get("processed", 1)))
        return {
            "source": source,
            "recorded_at": utc_now(),
            "records_processed": result.get("processed"),
            "decisions": result.get("decisions"),
            "latency": {
                "total_ms": result.get("latency_ms"),
                "per_record_ms": round(float(result.get("latency_ms", 0)) / processed, 2),
                "inside_model_calls_ms": round(model_ms, 2),
                "per_model_call_ms": round(model_ms / calls, 2) if calls else 0.0,
                "share_of_time_in_model": round(model_ms / float(result.get("latency_ms") or 1), 3),
            },
            "model_usage": usage,
            "growth": {key: value for key, value in self.growth().items() if key != "samples"},
            "database": self.stats(),
            "backends": self.backends(),
        }

    def write_ingest_report(self, result: dict[str, Any], out_path: str, source: str = "") -> dict[str, Any]:
        from pathlib import Path as _Path  # noqa: PLC0415
        report = self.ingest_report(result, source)
        output = _Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        report["written_to"] = out_path
        return report

    def ingest_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Ingest a corpus on one connection, committing in batches.

        This used to call `ingest_record` per record, which opened a
        connection, re-ran `migrate`, and committed - one fsync per dictation.
        On the 500-record corpus the commits alone dominated the wall clock.
        The decision path is byte-for-byte identical; only the transaction
        boundary moved.
        """
        started = time.perf_counter()
        processed = 0
        decisions = 0
        sample_every = max(1, len(records) // 10) if records else 1
        with connect(self.db_path) as conn:
            migrate(conn)
            for record in records:
                result = self._ingest_one(conn, record)
                processed += 1
                decisions += len(result["decisions"])
                if processed % sample_every == 0 or processed == len(records):
                    conn.commit()
                    self._sample_growth(conn)
            conn.commit()
        return {
            "processed": processed,
            "decisions": decisions,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def ingest_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Ingest exactly one record, in its own transaction."""
        with connect(self.db_path) as conn:
            migrate(conn)
            result = self._ingest_one(conn, record)
            conn.commit()
        return result

    def _ingest_one(self, conn, record: dict[str, Any]) -> dict[str, Any]:
        """The write path for one record. Does not commit - the caller owns
        the transaction boundary, so a corpus can be ingested in batches."""
        normalized = self._normalize_record(record)
        candidates, ignored_reasons = self.extractor.extract(normalized)
        decisions = []
        self._upsert_transcript(conn, normalized)
        for candidate in candidates:
            decisions.append(self._apply_candidate(conn, normalized, candidate))
        if not candidates:
            reason = ignored_reasons[0] if ignored_reasons else "No durable memory candidate."
            decisions.append(self._record_decision(
                conn,
                normalized["id"],
                "NO_OP",
                None,
                {"ignored_text": snippet(normalized["formatted_text"]), "kind": "ignored"},
                reason,
                0.0,
                0.0,
            ))
        return {"record_id": normalized["id"], "decisions": decisions}

    # -- dictation mode -----------------------------------------------------

    def dictate(self, text: str, app: str = "dictation", created_at: str | None = None) -> dict[str, Any]:
        """Regular dictation: speech in, written text out.

        This is the other half of the product, and the boundary is deliberate.
        In dictation, memory has exactly two jobs: shape HOW the text is
        written (style preferences the person has already stated), and learn
        quietly from what was said. It never answers, never uses tools, and
        never interrupts to ask permission - the person is writing, not
        talking to an assistant.

        Everything memory does interactively - retrieval, answering, refusing,
        correcting - belongs to Hey Kivi (`ask`).
        """
        started = time.perf_counter()
        raw = normalize_text(text)
        record = self._normalize_record({
            "created_at": created_at or utc_now(),
            "app": app,
            "raw_asr": raw,
            "formatted_text": raw,
            "metadata": {"mode": "dictation"},
        })

        with connect(self.db_path) as conn:
            migrate(conn)
            style_rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE status IN ('active', 'tentative')
                  AND scope = 'communication'
                  AND predicate LIKE 'prefers_%'
                ORDER BY confidence DESC LIMIT 4
                """
            ).fetchall()
            claims = [row["canonical_text"] for row in style_rows]
            formatted = self.answerer.polish(raw, claims)
            # Only claim a preference was used if the polisher can actually use
            # one. The template polisher cleans disfluencies and deliberately
            # applies no style, so reporting "written using 4 things Kivi knows
            # about you" would be false - and a memory product that overstates
            # what it did is worse than one that stays quiet.
            applied = [
                {"id": row["id"], "claim": self._fragment(row["canonical_text"])}
                for row in style_rows
            ] if getattr(self.answerer, "applies_style", False) else []

        record["formatted_text"] = formatted
        result = self.ingest_record(record)

        learned = []
        ignored = []
        for decision in result["decisions"]:
            action = decision["action"]
            candidate = decision.get("candidate") or {}
            if action in ("ADD", "ADD_TENTATIVE", "UPDATE"):
                learned.append({
                    "id": decision.get("target_memory_id"),
                    "claim": self._fragment(str(candidate.get("canonical_text") or "")),
                    "action": action,
                })
            elif action in ("NO_OP", "REJECT"):
                ignored.append(decision.get("reason") or "Nothing durable in this dictation.")

        return {
            "raw": raw,
            "formatted": formatted,
            "app": record["app"],
            "created_at": record["created_at"],
            "transcript_id": record["id"],
            "applied": applied,
            "learned": learned,
            "ignored": ignored,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    @staticmethod
    def _fragment(canonical: str) -> str:
        """"User prefers X." -> "you prefer X" - a fragment the UI can inline."""
        sentence = to_second_person(normalize_text(canonical)).rstrip(".")
        return sentence[0].lower() + sentence[1:] if sentence else sentence

    # -- explicit control ---------------------------------------------------

    def forget(self, memory_id: str, reason: str = "The person asked Kivi to forget this.") -> dict[str, Any]:
        """Archive one memory by id. Never deletes: the audit trail survives."""
        with connect(self.db_path) as conn:
            migrate(conn)
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if row is None:
                return {"ok": False, "error": "No such memory."}
            payload = self._memory_payload(conn, row)
            if row["status"] != "archived":
                conn.execute("UPDATE memories SET status='archived', updated_at=? WHERE id=?",
                             (utc_now(), memory_id))
                self._record_decision(
                    conn, None, "FORGET", memory_id,
                    {"canonical_text": row["canonical_text"], "kind": "user_control"},
                    reason, float(row["confidence"]), float(row["utility"]), source="user",
                )
            conn.commit()
        payload["status"] = "archived"
        return {"ok": True, "memory": payload}

    # -- growth -------------------------------------------------------------

    def sample_growth(self) -> dict[str, Any]:
        """Record one point on the growth curve."""
        with connect(self.db_path) as conn:
            migrate(conn)
            result = self._sample_growth(conn)
            conn.commit()
        return result

    def _sample_growth(self, conn) -> dict[str, Any]:
        """One growth-curve point on a caller-owned connection. Does not commit."""
        from pathlib import Path as _Path  # noqa: PLC0415

        counts = {
            "transcripts": conn.execute("SELECT count(*) n FROM transcripts").fetchone()["n"],
            "memories": conn.execute("SELECT count(*) n FROM memories").fetchone()["n"],
            "active_memories": conn.execute(
                "SELECT count(*) n FROM memories WHERE status IN ('active','tentative')").fetchone()["n"],
            "decisions": conn.execute("SELECT count(*) n FROM decisions").fetchone()["n"],
            "evidence_rows": conn.execute("SELECT count(*) n FROM memory_evidence").fetchone()["n"],
            "embedding_rows": conn.execute("SELECT count(*) n FROM embeddings").fetchone()["n"],
        }
        try:
            db_bytes = _Path(self.db_path).stat().st_size
        except OSError:
            db_bytes = 0
        conn.execute(
            """
            INSERT INTO growth_samples(
                transcripts, memories, active_memories, decisions,
                evidence_rows, embedding_rows, db_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (counts["transcripts"], counts["memories"], counts["active_memories"],
             counts["decisions"], counts["evidence_rows"], counts["embedding_rows"], db_bytes),
        )
        return {**counts, "db_bytes": db_bytes}

    def growth(self) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            migrate(conn)
            rows = [dict(row) for row in conn.execute(
                "SELECT * FROM growth_samples ORDER BY id")]
        if not rows:
            return {"samples": [], "note": "No growth samples recorded yet."}
        last = rows[-1]
        transcripts = max(1, last["transcripts"])
        return {
            "samples": rows,
            "latest": last,
            "memories_per_100_transcripts": round(100 * last["active_memories"] / transcripts, 2),
            "bytes_per_transcript": round(last["db_bytes"] / transcripts, 1),
            "selectivity": round(1 - last["active_memories"] / transcripts, 4),
        }

    # Field aliases for imported corpora. The reviewer's records carry "the
    # ordinary metadata available in our logs", whose exact key names are not
    # published, so import accepts the obvious spellings rather than failing on
    # a column called `text` instead of `formatted_text`.
    FIELD_ALIASES = {
        "formatted_text": ("formatted_text", "formatted", "llm_formatted", "formatted_output",
                           "text", "content", "transcript", "final_text", "output"),
        "raw_asr": ("raw_asr", "asr", "raw", "raw_text", "asr_text", "asr_output", "transcription"),
        "created_at": ("created_at", "timestamp", "time", "date", "recorded_at", "started_at", "ts"),
        "app": ("app", "application", "app_name", "source", "surface", "client", "target_app"),
        "id": ("id", "record_id", "uuid", "_id", "dictation_id"),
    }

    @classmethod
    def _pick(cls, record: dict[str, Any], field: str) -> Any:
        for key in cls.FIELD_ALIASES[field]:
            value = record.get(key)
            if value not in (None, ""):
                return value
        return None

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        formatted = normalize_text(str(
            self._pick(record, "formatted_text") or self._pick(record, "raw_asr") or ""))
        raw = normalize_text(str(self._pick(record, "raw_asr") or formatted))
        created_at = str(self._pick(record, "created_at") or utc_now())

        # Anything not recognised is preserved rather than dropped, so an
        # imported corpus keeps its own metadata in the audit trail.
        known = {key for keys in self.FIELD_ALIASES.values() for key in keys} | {"metadata"}
        extra = {key: value for key, value in record.items() if key not in known}
        metadata = dict(record.get("metadata") or {})
        if extra:
            metadata.setdefault("source_fields", extra)

        return {
            "id": str(self._pick(record, "id") or stable_id("tr", created_at, raw, formatted)),
            "created_at": created_at,
            "app": str(self._pick(record, "app") or "dictation").lower(),
            "raw_asr": raw,
            "formatted_text": formatted,
            "metadata": metadata,
        }

    def _upsert_transcript(self, conn, record: dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO transcripts(id, created_at, app, raw_asr, formatted_text, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                created_at=excluded.created_at,
                app=excluded.app,
                raw_asr=excluded.raw_asr,
                formatted_text=excluded.formatted_text,
                metadata_json=excluded.metadata_json
            """,
            (
                record["id"],
                record["created_at"],
                record["app"],
                record["raw_asr"],
                record["formatted_text"],
                json.dumps(record["metadata"], sort_keys=True),
            ),
        )
        self._store_embedding(conn, "transcript", record["id"], record["formatted_text"])

    # -- controller delegates -----------------------------------------------
    #
    # The decision ladder lives in `controller.MemoryController`. These two
    # names are kept on the engine because they are the write path's public
    # surface: the ingest loop, the correction handler and the tests all reach
    # for them.

    def _apply_candidate(self, conn, record: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
        return self.controller.apply(conn, record, candidate)

    def _record_decision(self, conn, *args, **kwargs) -> dict[str, Any]:
        return self.controller.record_decision(conn, *args, **kwargs)

    # -- read path ----------------------------------------------------------
    #
    # `ask` is the engine's public entry point; the routing and the five
    # handlers behind it live in `query.py`.

    def ask(self, query: str) -> dict[str, Any]:
        return self.query.ask(query)

    def _style_memories(self, conn) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE status IN ('active', 'tentative')
              AND predicate IN ('prefers_response_style', 'prefers_response_format')
            ORDER BY confidence DESC
            LIMIT 3
            """
        ).fetchall()
        return [self._memory_payload(conn, row) for row in rows]

    def _polish_text(self, conn, text: str) -> str:
        """Polishing is delegated to the answer backend.

        Previously this stripped a fixed filler list and then ran
        `WHERE object LIKE '%concise%'` to decide whether to truncate to two
        sentences - a hardcoded reading of one specific preference string. The
        stored style memories are now handed to the polisher as claims, so any
        style the extractor learns is applied, not just the one that was
        anticipated.
        """
        claims = [row["canonical_text"] for row in conn.execute(
            """
            SELECT canonical_text FROM memories
            WHERE status IN ('active', 'tentative')
              AND scope = 'communication'
              AND predicate LIKE 'prefers_%'
            ORDER BY confidence DESC LIMIT 4
            """
        )]
        return self.answerer.polish(text, claims)

    def _corpus_now(self, conn) -> datetime:
        """The clock the user is implicitly using.

        "Yesterday" means yesterday relative to the data, not to the wall
        clock, so an imported corpus stays searchable months later.
        Replaces `_target_date` / `_target_hour`, whose vocabulary was
        "yesterday", "today" and an ISO date; parsing now lives in
        `intent.BaseIntentRouter.resolve_datetime`, which the LLM router fills
        with structured slots and the offline router parses generically
        (weekday names, "N days ago", dayparts, explicit times).
        """
        row = conn.execute("SELECT max(created_at) AS max_created FROM transcripts").fetchone()
        return parse_dt(row["max_created"]) if row and row["max_created"] else datetime.utcnow()

    def _memory_payload(self, conn, row) -> dict[str, Any]:
        evidence = conn.execute(
            """
            SELECT e.*, t.created_at, t.app, t.formatted_text
            FROM memory_evidence e
            JOIN transcripts t ON t.id = e.transcript_id
            WHERE e.memory_id=?
            ORDER BY e.observed_at DESC
            LIMIT 5
            """,
            (row["id"],),
        ).fetchall()
        return {
            "id": row["id"],
            "type": row["memory_type"],
            "status": row["status"],
            "scope": row["scope"],
            "predicate": row["predicate"],
            "object": row["object"],
            "canonical_text": row["canonical_text"],
            "confidence": float(row["confidence"]),
            "utility": float(row["utility"]),
            "importance": float(row["importance"]),
            "recurrence": int(row["recurrence"]),
            "first_seen_at": row["first_seen_at"],
            "last_seen_at": row["last_seen_at"],
            "evidence": [dict(item) for item in evidence],
        }

    def _sources_for_memories(self, conn, memory_ids: list[str]) -> list[dict[str, Any]]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        rows = conn.execute(
            f"""
            SELECT DISTINCT t.*
            FROM transcripts t
            JOIN memory_evidence e ON e.transcript_id = t.id
            WHERE e.memory_id IN ({placeholders})
            ORDER BY t.created_at DESC
            LIMIT 12
            """,
            memory_ids,
        ).fetchall()
        return [dict(row) for row in rows]

    def _abstain(self, answer: str, reason: str) -> dict[str, Any]:
        return {
            "answer": answer,
            "abstained": True,
            "used_memories": [],
            "source_transcripts": [],
            "reason": reason,
        }

    # -- read-only views ------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return inspection.stats(self)

    def memories(self, limit: int = 100) -> list[dict[str, Any]]:
        return inspection.memories(self, limit)

    def decisions(self, limit: int = 100) -> list[dict[str, Any]]:
        return inspection.decisions(self, limit)

    def transcripts(self, limit: int = 100, search: str | None = None) -> list[dict[str, Any]]:
        return inspection.transcripts(self, limit, search)
