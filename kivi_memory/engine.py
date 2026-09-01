"""The memory controller.

This is the deterministic part of the system and it is deliberately the part
that did NOT change when the sensors became model-backed. Extraction, NLI and
embeddings are pluggable; admission, conflict resolution, reinforcement,
decay and abstention are arithmetic defined here and in `config.KiviConfig`.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from .answerer import get_answerer
from .config import KiviConfig
from .core import (
    Candidate,
    is_exclusive,
    clamp,
    cosine,
    effective_confidence,
    normalize_text,
    parse_dt,
    snippet,
    stable_id,
    tokens,
    to_second_person,
    utc_now,
)
from .db import connect, migrate
from .embeddings import get_embedder
from .extractors import RuleExtractor, SemanticExtractor, get_extractor
from .intent import get_router
from .llm import OllamaClient, get_client
from .nli import NLIResult, get_nli


# Last-resort safety net only. Routing is done by `intent.get_router`; these
# patterns exist so an unmistakable retraction is never missed even if the
# router is degraded and mis-scores it.
FORGET_PATTERNS = (
    r"\bforget\b",
    r"\bthat'?s wrong\b",
    r"\bthat is wrong\b",
    r"\bthat'?s incorrect\b",
    r"\bstop remembering\b",
    r"\bdelete (?:that|this) memory\b",
    r"\bremove (?:that|this) memory\b",
    r"\bno longer true\b",
)


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
        started = time.perf_counter()
        processed = 0
        decisions = 0
        sample_every = max(1, len(records) // 10) if records else 1
        for record in records:
            result = self.ingest_record(record)
            processed += 1
            decisions += len(result["decisions"])
            if processed % sample_every == 0 or processed == len(records):
                self.sample_growth()
        return {
            "processed": processed,
            "decisions": decisions,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def ingest_record(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_record(record)
        candidates, ignored_reasons = self.extractor.extract(normalized)
        decisions = []
        with connect(self.db_path) as conn:
            migrate(conn)
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
            conn.commit()
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
                WHERE status IN ('active', 'tentative') AND predicate LIKE 'prefers%'
                ORDER BY confidence DESC LIMIT 4
                """
            ).fetchall()
            claims = [row["canonical_text"] for row in style_rows]
            formatted = self.answerer.polish(raw, claims)
            applied = [
                {"id": row["id"], "claim": self._fragment(row["canonical_text"])}
                for row in style_rows
            ]

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
        from pathlib import Path as _Path  # noqa: PLC0415

        with connect(self.db_path) as conn:
            migrate(conn)
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
            conn.commit()
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

    def _apply_candidate(self, conn, record: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
        existing = conn.execute(
            """
            SELECT * FROM memories
            WHERE memory_type=? AND subject=? AND predicate=? AND scope=? AND object=?
              AND status IN ('active', 'tentative')
            ORDER BY updated_at DESC LIMIT 1
            """,
            (candidate.memory_type, candidate.subject, candidate.predicate, candidate.scope, candidate.object),
        ).fetchone()
        if existing:
            return self._update_memory(conn, record, candidate, existing)

        # A paraphrase of something already known is a reinforcement, not a
        # second memory. Exact-key matching cannot see this ("bangalore" vs
        # "bangalore, india"); bidirectional entailment can.
        twin, twin_nli = self._find_equivalent(conn, candidate)
        if twin is not None:
            return self._update_memory(conn, record, candidate, twin, nli=twin_nli)

        conflict, conflict_nli = self._find_conflict(conn, candidate)
        if (
            conflict is not None
            and candidate.status == "active"
            and candidate.confidence >= self.config.supersede_confidence
        ):
            conn.execute(
                "UPDATE memories SET status='archived', updated_at=? WHERE id=?",
                (utc_now(), conflict["id"]),
            )
            self._insert_memory(conn, record, candidate, supersedes_id=conflict["id"])
            basis = (
                f"NLI {conflict_nli.label} p={conflict_nli.probability:.2f} via {conflict_nli.backend}"
                if conflict_nli else "exclusive-predicate rule"
            )
            return self._record_decision(
                conn,
                record["id"],
                "UPDATE",
                candidate.id,
                asdict(candidate),
                f"Superseded conflicting memory {conflict['id']} ({basis}); new evidence was stronger.",
                candidate.confidence,
                candidate.utility,
                nli=conflict_nli,
                source=candidate.source,
            )

        if candidate.utility >= self.config.admit_utility and candidate.confidence >= self.config.admit_confidence:
            self._insert_memory(conn, record, candidate)
            action = "ADD_TENTATIVE" if candidate.status == "tentative" else "ADD"
            return self._record_decision(
                conn,
                record["id"],
                action,
                candidate.id,
                asdict(candidate),
                candidate.reason,
                candidate.confidence,
                candidate.utility,
                source=candidate.source,
            )

        return self._record_decision(
            conn,
            record["id"],
            "REJECT",
            None,
            asdict(candidate),
            (
                f"Candidate had insufficient confidence/utility for durable memory "
                f"(utility {candidate.utility:.2f} < {self.config.admit_utility}, "
                f"confidence {candidate.confidence:.2f} < {self.config.admit_confidence})."
            ),
            candidate.confidence,
            candidate.utility,
            source=candidate.source,
        )

    # -- conflict / equivalence detection -----------------------------------

    def _neighbourhood(self, conn, candidate: Candidate, limit: int = 6) -> list[Any]:
        """Stored memories that plausibly talk about the same thing.

        Restricted to the same subject and scope, then ranked by embedding
        similarity, so NLI is asked at most `limit` questions per candidate
        instead of once per stored memory.
        """
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE subject=? AND scope=? AND status IN ('active', 'tentative') AND id<>?
            """,
            (candidate.subject, candidate.scope, candidate.id),
        ).fetchall()
        if not rows:
            return []
        query_vector = self.embedder.embed_one(candidate.embedding_text())
        scored = []
        for row in rows:
            similarity = cosine(query_vector, self._embedding_for_memory(conn, row))
            same_predicate = row["predicate"] == candidate.predicate
            if similarity >= self.config.semantic_conflict_similarity or same_predicate:
                scored.append((similarity + (0.25 if same_predicate else 0.0), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:limit]]

    def _find_equivalent(self, conn, candidate: Candidate) -> tuple[Any | None, NLIResult | None]:
        if self.nli.name == "off":
            return None, None
        for row in self._neighbourhood(conn, candidate):
            if row["object"] == candidate.object and row["predicate"] == candidate.predicate:
                continue  # exact-key path already handled this
            same, result = self.nli.equivalent(
                row["canonical_text"], candidate.canonical_text, self.config.nli_equivalence_threshold
            )
            if same:
                return row, result
        return None, None

    def _find_conflict(self, conn, candidate: Candidate) -> tuple[Any | None, NLIResult | None]:
        """Conflict detection: NLI first, exclusive-predicate whitelist as backstop.

        The whitelist is kept because it is cheap, exact, and correct for the
        predicates it covers. NLI extends the same decision to everything else,
        including predicates nobody enumerated in advance.
        """
        if self.nli.name != "off" and candidate.memory_type in self.config.nli_conflict_types:
            for row in self._neighbourhood(conn, candidate):
                if row["memory_type"] not in self.config.nli_conflict_types:
                    continue
                if row["object"] == candidate.object and row["predicate"] == candidate.predicate:
                    continue
                contradicts, result = self.nli.contradicts(
                    row["canonical_text"], candidate.canonical_text, self.config.nli_contradiction_threshold
                )
                if contradicts:
                    return row, result

        if not is_exclusive(candidate.predicate):
            return None, None
        row = conn.execute(
            """
            SELECT * FROM memories
            WHERE memory_type=? AND subject=? AND predicate=? AND scope=?
              AND object<>? AND status IN ('active', 'tentative')
            ORDER BY confidence DESC, updated_at DESC LIMIT 1
            """,
            (candidate.memory_type, candidate.subject, candidate.predicate, candidate.scope, candidate.object),
        ).fetchone()
        return row, None

    def _insert_memory(self, conn, record: dict[str, Any], candidate: Candidate, supersedes_id: str | None = None) -> None:
        now = utc_now()
        conn.execute(
            """
            INSERT OR REPLACE INTO memories(
                id, memory_type, subject, predicate, object, scope, canonical_text,
                status, confidence, utility, importance, recurrence, decay_rate,
                first_seen_at, last_seen_at, updated_at, supersedes_id, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.id,
                candidate.memory_type,
                candidate.subject,
                candidate.predicate,
                candidate.object,
                candidate.scope,
                candidate.canonical_text,
                candidate.status,
                candidate.confidence,
                candidate.utility,
                candidate.importance,
                1,
                candidate.decay_rate,
                record["created_at"],
                record["created_at"],
                now,
                supersedes_id,
                json.dumps(candidate.metadata, sort_keys=True),
            ),
        )
        conn.execute("UPDATE memories SET source=? WHERE id=?", (candidate.source, candidate.id))
        self._store_embedding(conn, "memory", candidate.id, candidate.embedding_text())
        self._insert_evidence(conn, candidate.id, record, "supports", candidate.evidence, candidate.confidence)

    def _update_memory(
        self,
        conn,
        record: dict[str, Any],
        candidate: Candidate,
        existing,
        nli: NLIResult | None = None,
    ) -> dict[str, Any]:
        recurrence = int(existing["recurrence"]) + 1
        confidence = clamp(1 - (1 - float(existing["confidence"])) * (1 - candidate.confidence * 0.45))
        utility = clamp(max(float(existing["utility"]), candidate.utility) + min(0.08, recurrence * 0.006))
        status = "active" if confidence >= self.config.tentative_confidence else existing["status"]
        conn.execute(
            """
            UPDATE memories
            SET confidence=?, utility=?, importance=max(importance, ?), recurrence=?,
                last_seen_at=?, updated_at=?, status=?
            WHERE id=?
            """,
            (
                round(confidence, 3),
                round(utility, 3),
                candidate.importance,
                recurrence,
                record["created_at"],
                utc_now(),
                status,
                existing["id"],
            ),
        )
        self._insert_evidence(conn, existing["id"], record, "supports", candidate.evidence, candidate.confidence)
        if nli is not None:
            note = (
                f"Merged a paraphrase of an existing memory (NLI {nli.label} "
                f"p={nli.probability:.2f} via {nli.backend}); recurrence #{recurrence}."
            )
        else:
            note = f"Matched existing memory and raised confidence through recurrence #{recurrence}."
        return self._record_decision(
            conn,
            record["id"],
            "REINFORCE",
            existing["id"],
            asdict(candidate),
            note,
            confidence,
            utility,
            nli=nli,
            source=candidate.source,
        )

    def _insert_evidence(
        self,
        conn,
        memory_id: str,
        record: dict[str, Any],
        relation: str,
        evidence_snippet: str,
        contribution: float,
    ) -> None:
        evidence_id = stable_id("ev", memory_id, record["id"], relation, evidence_snippet)
        conn.execute(
            """
            INSERT OR IGNORE INTO memory_evidence(
                id, memory_id, transcript_id, relation, snippet, contribution, observed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (evidence_id, memory_id, record["id"], relation, evidence_snippet, contribution, record["created_at"]),
        )

    def _record_decision(
        self,
        conn,
        transcript_id: str | None,
        action: str,
        target_memory_id: str | None,
        candidate: dict[str, Any],
        reason: str,
        confidence: float,
        utility: float,
        nli: NLIResult | None = None,
        source: str = "rule",
    ) -> dict[str, Any]:
        decision_id = stable_id("dec", transcript_id, action, target_memory_id, json.dumps(candidate, sort_keys=True))
        row = {
            "id": decision_id,
            "transcript_id": transcript_id,
            "action": action,
            "target_memory_id": target_memory_id,
            "candidate": candidate,
            "reason": reason,
            "confidence": round(float(confidence), 3),
            "utility": round(float(utility), 3),
            "extractor": source,
            "nli": nli.as_dict() if nli else None,
        }
        conn.execute(
            """
            INSERT OR IGNORE INTO decisions(
                id, transcript_id, action, target_memory_id, candidate_json, reason,
                confidence, utility, extractor, nli_label, nli_probability
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["transcript_id"],
                row["action"],
                row["target_memory_id"],
                json.dumps(candidate, sort_keys=True),
                row["reason"],
                row["confidence"],
                row["utility"],
                source,
                nli.label if nli else None,
                round(float(nli.probability), 3) if nli else None,
            ),
        )
        return row

    def ask(self, query: str) -> dict[str, Any]:
        started = time.perf_counter()
        query = normalize_text(query)
        lower = query.lower()
        with connect(self.db_path) as conn:
            migrate(conn)
            known_apps = [row["app"] for row in conn.execute(
                "SELECT DISTINCT app FROM transcripts WHERE app IS NOT NULL")]
            intent = self.router.route(query, known_apps)
            # Safety net: an unmistakable retraction outranks the router.
            if any(re.search(pattern, lower) for pattern in FORGET_PATTERNS):
                intent.name = "correction"

            if intent.name == "correction":
                result = self._handle_correction(conn, query, intent)
            elif intent.name == "memory_overview":
                result = self._answer_memory_overview(conn, query)
            elif intent.name == "ignored_audit":
                result = self._answer_ignored(conn)
            elif intent.name == "find_transcript":
                result = self._answer_find_transcript(conn, query, intent)
            else:
                result = self._answer_from_memories(conn, query)
            result["intent"] = intent.as_dict()
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["query"] = query
        return result

    def _answer_memory_overview(self, conn, query: str) -> dict[str, Any]:
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE status IN ('active', 'tentative')
            ORDER BY utility DESC, confidence DESC, recurrence DESC
            LIMIT 12
            """
        ).fetchall()
        if not rows:
            return self._abstain("Kivi has not learned any durable memories yet.", "No active memories.")
        memories = [self._memory_payload(conn, row) for row in rows]
        lines = [f"- {item['canonical_text']} ({item['status']}, confidence {item['confidence']:.2f})" for item in memories]
        return {
            "answer": "Here is what Kivi currently remembers:\n" + "\n".join(lines),
            "abstained": False,
            "used_memories": memories,
            "source_transcripts": self._sources_for_memories(conn, [row["id"] for row in rows]),
            "reason": "User asked for an inspectable memory overview.",
        }

    def _answer_ignored(self, conn) -> dict[str, Any]:
        rows = conn.execute(
            """
            SELECT d.*, t.formatted_text, t.created_at, t.app
            FROM decisions d
            LEFT JOIN transcripts t ON t.id = d.transcript_id
            WHERE d.action IN ('NO_OP', 'REJECT')
            ORDER BY d.created_at DESC
            LIMIT 8
            """
        ).fetchall()
        if not rows:
            return self._abstain("I do not have ignored-memory decisions to show yet.", "No NO_OP or REJECT decisions.")
        source_transcripts = []
        lines = []
        for row in rows:
            text = row["formatted_text"] or json.loads(row["candidate_json"]).get("ignored_text", "")
            lines.append(f"- {row['reason']} Source: {snippet(text, limit=90)}")
            if row["transcript_id"]:
                source_transcripts.append({
                    "id": row["transcript_id"],
                    "created_at": row["created_at"],
                    "app": row["app"],
                    "formatted_text": text,
                })
        return {
            "answer": "Kivi ignored these because they had low durable utility:\n" + "\n".join(lines),
            "abstained": False,
            "used_memories": [],
            "source_transcripts": source_transcripts,
            "reason": "User asked to inspect ignored candidates and controller decisions.",
        }

    def _answer_find_transcript(self, conn, query: str, intent=None) -> dict[str, Any]:
        slots = dict(intent.slots) if intent is not None else {}
        app = slots.get("app")
        if app is None:
            known = [row["app"] for row in conn.execute(
                "SELECT DISTINCT app FROM transcripts WHERE app IS NOT NULL")]
            app = self.router.resolve_app(query, known)
        base = self._corpus_now(conn)
        target_date, target_hour = self.router.resolve_datetime(query, slots, base)
        query_tokens = tokens(query)
        rows = conn.execute("SELECT * FROM transcripts ORDER BY created_at DESC").fetchall()
        semantic = bool(self.embedder.describe().get("semantic"))
        query_vector = self.embedder.embed_one(query) if semantic else None
        vectors = self._load_embeddings(conn, "transcript") if query_vector is not None else {}
        scored = []
        for row in rows:
            if app and row["app"].lower() != app:
                continue
            created = parse_dt(row["created_at"])
            score = 0.0
            if target_date:
                day_diff = abs((created.date() - target_date.date()).days)
                score += max(0, 3 - day_diff * 1.5)
            if target_hour is not None:
                score += max(0, 3 - abs(created.hour + created.minute / 60 - target_hour) * 0.8)
            overlap = len(query_tokens & tokens(row["formatted_text"]))
            score += min(3, overlap * 0.45)
            if query_vector is not None and row["id"] in vectors:
                score += max(0.0, cosine(query_vector, vectors[row["id"]])) * 1.5
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored or scored[0][0] < 2.2:
            return self._abstain(
                "I could not find a transcript matching that app/time request.",
                "No transcript crossed the retrieval threshold.",
            )
        top = scored[0][1]
        polished = self._polish_text(conn, top["formatted_text"])
        return {
            "answer": (
                f"I found this {top['app']} transcript from {top['created_at']}.\n\n"
                f"Polished version:\n{polished}"
            ),
            "abstained": False,
            "used_memories": self._style_memories(conn),
            "source_transcripts": [{
                "id": top["id"],
                "created_at": top["created_at"],
                "app": top["app"],
                "formatted_text": top["formatted_text"],
                "raw_asr": top["raw_asr"],
            }],
            "reason": "Hybrid retrieval used app, date, approximate time, lexical overlap, and response-style memory.",
        }

    def _answer_from_memories(self, conn, query: str) -> dict[str, Any]:
        rows = conn.execute(
            "SELECT * FROM memories WHERE status IN ('active', 'tentative')"
        ).fetchall()
        q_tokens = tokens(query)
        cfg = self.config
        semantic = bool(self.embedder.describe().get("semantic"))
        query_vector = self.embedder.embed_one(query) if semantic else None
        now = datetime.utcnow()

        scored = []
        for row in rows:
            text = " ".join([
                row["canonical_text"],
                row["memory_type"],
                row["predicate"],
                row["object"],
                row["scope"],
            ])
            overlap = len(q_tokens & tokens(text))

            # decay_rate was dead weight in the first version: stored on every
            # memory and never read. Retrieval now scores on time-decayed
            # confidence, damped by how often the memory has recurred.
            conf_eff = effective_confidence(
                row["confidence"], row["decay_rate"], row["recurrence"], row["last_seen_at"], now
            )

            similarity = 0.0
            if query_vector is not None:
                similarity = max(0.0, cosine(query_vector, self._embedding_for_memory(conn, row)))

            score = (
                overlap * cfg.w_lexical
                + similarity * cfg.w_semantic
                + conf_eff * cfg.w_confidence
                + row["utility"] * cfg.w_utility
            )
            if row["status"] == "tentative":
                score -= cfg.tentative_penalty
            if score > cfg.retrieval_floor:
                scored.append((score, similarity, conf_eff, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [item[3] for item in scored][:4]
        if not selected or scored[0][0] < cfg.answer_threshold:
            return self._abstain(
                "I do not have enough supported memory to answer that.",
                "No active memory crossed the answer threshold.",
            )

        memories = [self._memory_payload(conn, row) for row in selected]
        composed = self.answerer.compose(query, selected, {"style": self._style_hint(conn)})
        if composed.get("abstain"):
            return self._abstain(
                composed.get("answer") or "I do not have enough supported memory to answer that.",
                f"Answer backend ({composed.get('backend')}) declined to answer from the retrieved memories.",
            )

        cited = set(composed.get("cited_ids") or [])
        if cited:
            memories = [item for item in memories if item["id"] in cited] or memories

        detail = {item[3]["id"]: {"score": round(item[0], 3), "similarity": round(item[1], 3),
                                  "decayed_confidence": round(item[2], 3)} for item in scored}
        for item in memories:
            item["retrieval"] = detail.get(item["id"], {})

        return {
            "answer": composed["answer"],
            "abstained": False,
            "used_memories": memories,
            "source_transcripts": self._sources_for_memories(conn, [item["id"] for item in memories]),
            "reason": (
                f"Hybrid retrieval: {cfg.w_lexical}*lexical + {cfg.w_semantic}*cosine("
                f"{self.embedder.describe().get('backend')}) + {cfg.w_confidence}*decayed_confidence + "
                f"{cfg.w_utility}*utility, tentative penalty {cfg.tentative_penalty}. "
                f"Answer composed by {composed.get('backend')} from the cited memories only."
            ),
        }

    def _style_hint(self, conn) -> str | None:
        row = conn.execute(
            """
            SELECT object FROM memories
            WHERE status IN ('active', 'tentative') AND predicate LIKE 'prefers_response%'
            ORDER BY confidence DESC LIMIT 1
            """
        ).fetchone()
        return row["object"] if row else None

    # -- corrections --------------------------------------------------------

    def _handle_correction(self, conn, query: str, intent=None) -> dict[str, Any]:
        """Explicit 'that's wrong' / 'forget that' handling.

        Archives rather than deletes: the memory stays in the audit trail with
        its evidence, and the FORGET decision records why it left.
        """
        rows = conn.execute("SELECT * FROM memories WHERE status IN ('active','tentative')").fetchall()
        if not rows:
            return self._abstain("There is nothing stored to correct yet.", "No active memories.")

        # The router extracts what the user wants dropped. The regex strip is
        # only the fallback for when it did not fill the slot.
        target = normalize_text(str((intent.slots.get("target") if intent else None) or ""))
        if not target:
            target = normalize_text(re.sub(
                r"\b(hey kivi|forget|that'?s wrong|that is wrong|that'?s incorrect|"
                r"stop remembering|delete|remove|that|this|memory|no longer true|about|please)\b",
                " ",
                query.lower(),
            ))
        q_tokens = tokens(target or query)
        semantic = bool(self.embedder.describe().get("semantic"))
        query_vector = self.embedder.embed_one(target or query) if semantic else None

        best = None
        best_score = 0.0
        for row in rows:
            overlap = len(q_tokens & tokens(f"{row['canonical_text']} {row['object']} {row['predicate']}"))
            similarity = (
                cosine(query_vector, self._embedding_for_memory(conn, row)) if query_vector is not None else 0.0
            )
            score = overlap * 0.6 + max(0.0, similarity) * 2.0
            if score > best_score:
                best, best_score = row, score

        if best is None or best_score < 0.8:
            return self._abstain(
                "I could not tell which memory you want me to drop. Ask me what I remember and name one.",
                "No stored memory matched the correction target confidently enough.",
            )

        conn.execute("UPDATE memories SET status='archived', updated_at=? WHERE id=?", (utc_now(), best["id"]))
        payload = self._memory_payload(conn, best)
        self._record_decision(
            conn,
            None,
            "FORGET",
            best["id"],
            {"canonical_text": best["canonical_text"], "match_score": round(best_score, 3), "kind": "user_correction"},
            f"User explicitly corrected or retracted this memory: {normalize_text(query)}",
            float(best["confidence"]),
            float(best["utility"]),
            source="user",
        )
        conn.commit()
        return {
            "answer": f"Dropped that memory: \"{best['canonical_text']}\" It is archived, not deleted, so the audit trail stays intact.",
            "abstained": False,
            "used_memories": [payload],
            "source_transcripts": self._sources_for_memories(conn, [best["id"]]),
            "reason": "User correction: memory archived and a FORGET decision recorded.",
        }

    # Removed: `_select_answer_memories` and `_phrase_bonus`.
    #
    # Both were hand-written tables mapping query keywords to predicates
    # ("language" -> uses_language, "golden goose" -> project:golden_goose,
    # "response"/"style" -> prefers_response_*). They existed to paper over a
    # tokenizer that treated `prefers_response_style` as one opaque token, so
    # the words already present in every predicate name could never match a
    # query. `core.expand_identifier` fixes that at the source, and the
    # association now emerges from the stored memory itself - for any
    # predicate, including ones the extractor invents at runtime.
    #
    # Ablation before removal: without the tables and without the tokenizer
    # fix, 5/6 answer checks. With the tokenizer fix and no tables, 6/6.

    # `_compose_answer` moved to answerer.TemplateAnswerer, where it is now the
    # fallback rather than the only option. See KiviMemoryEngine.answerer.

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
            WHERE status IN ('active', 'tentative') AND predicate LIKE 'prefers%'
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

    def stats(self) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            migrate(conn)
            counts = {}
            for table in ("transcripts", "memories", "memory_evidence", "decisions"):
                counts[table] = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]
            by_status = {
                row["status"]: row["n"]
                for row in conn.execute("SELECT status, count(*) AS n FROM memories GROUP BY status")
            }
            by_action = {
                row["action"]: row["n"]
                for row in conn.execute("SELECT action, count(*) AS n FROM decisions GROUP BY action")
            }
            by_source = {
                row["source"]: row["n"]
                for row in conn.execute("SELECT source, count(*) AS n FROM memories GROUP BY source")
            }
            nli_used = conn.execute(
                "SELECT count(*) AS n FROM decisions WHERE nli_label IS NOT NULL"
            ).fetchone()["n"]
            return {
                "counts": counts,
                "memories_by_status": by_status,
                "memories_by_source": by_source,
                "decisions_by_action": by_action,
                "decisions_with_nli": nli_used,
                "backends": self.backends(),
            }

    def memories(self, limit: int = 100) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            migrate(conn)
            rows = conn.execute(
                """
                SELECT * FROM memories
                ORDER BY status, utility DESC, confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._memory_payload(conn, row) for row in rows]

    def decisions(self, limit: int = 100) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            migrate(conn)
            rows = conn.execute(
                """
                SELECT d.*, t.formatted_text, t.created_at, t.app
                FROM decisions d
                LEFT JOIN transcripts t ON t.id = d.transcript_id
                ORDER BY d.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["candidate"] = json.loads(item.pop("candidate_json"))
                result.append(item)
            return result

    def transcripts(self, limit: int = 100, search: str | None = None) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            migrate(conn)
            if search:
                q = f"%{search.lower()}%"
                rows = conn.execute(
                    """
                    SELECT * FROM transcripts
                    WHERE lower(formatted_text) LIKE ? OR lower(raw_asr) LIKE ? OR lower(app) LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (q, q, q, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM transcripts ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]
