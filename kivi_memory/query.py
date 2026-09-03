"""The read path: turning a question into a grounded answer, or an abstention.

`ask` routes the question to one of five handlers and each returns the same
shape, so the interface never has to care which one ran:

    memory_question   answer from retrieved memories, with citations
    memory_overview   everything learned, in plain sentences
    ignored_audit     what was deliberately not remembered, and why
    find_transcript   locate a past dictation by time / app / content
    correction        "that's wrong" - archive the memory behind the claim

Two rules make the difference between this and a chatbot with a database.
Retrieval has an abstention floor: if the best-scoring memory does not clear
`answer_threshold`, nothing downstream runs and the system says it does not
know. And an answer that cites a memory id which was not retrieved is
discarded and replaced by the deterministic template, so grounding is enforced
structurally rather than requested in a prompt.

This holds a reference to the engine rather than owning state of its own -
retrieval, the answerer, the router and the memory payload helpers all belong
to the engine, and duplicating them here would be the actual coupling.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .core import cosine, normalize_text, parse_dt, snippet, tokens, utc_now
from .db import connect, migrate
from .retrieval import describe_weights, score_memories


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


class QueryService:
    """Everything `KiviMemoryEngine.ask` does, kept out of the engine."""

    def __init__(self, engine) -> None:
        self.engine = engine


    def ask(self, query: str) -> dict[str, Any]:
        started = time.perf_counter()
        query = normalize_text(query)
        lower = query.lower()
        with connect(self.engine.db_path) as conn:
            migrate(conn)
            known_apps = [row["app"] for row in conn.execute(
                "SELECT DISTINCT app FROM transcripts WHERE app IS NOT NULL")]
            intent = self.engine.router.route(query, known_apps)
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
            return self.engine._abstain("Kivi has not learned any durable memories yet.", "No active memories.")
        memories = [self.engine._memory_payload(conn, row) for row in rows]
        lines = [f"- {item['canonical_text']} ({item['status']}, confidence {item['confidence']:.2f})" for item in memories]
        return {
            "answer": "Here is what Kivi currently remembers:\n" + "\n".join(lines),
            "abstained": False,
            "used_memories": memories,
            "source_transcripts": self.engine._sources_for_memories(conn, [row["id"] for row in rows]),
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
            LIMIT 200
            """
        ).fetchall()
        if not rows:
            return self.engine._abstain("I do not have ignored-memory decisions to show yet.", "No NO_OP or REJECT decisions.")
        # One line per distinct reason with a single example, plus how many
        # others it stood for. The previous version listed the eight most recent
        # rows, which on a real corpus means the same sentence six times.
        source_transcripts = []
        lines = []
        seen: dict[str, int] = {}
        chosen: list[tuple[str, str]] = []
        for row in rows:
            reason = str(row["reason"])
            seen[reason] = seen.get(reason, 0) + 1
            if reason not in {item[0] for item in chosen}:
                text = row["formatted_text"] or json.loads(row["candidate_json"]).get("ignored_text", "")
                chosen.append((reason, text))
        for reason, text in chosen[:6]:
            others = seen[reason] - 1
            suffix = f" (and {others} more like it)" if others else ""
            lines.append(f"- {reason}{suffix}\n    e.g. \"{snippet(text, limit=90)}\"")
        for row in rows[:8]:
            text = row["formatted_text"] or json.loads(row["candidate_json"]).get("ignored_text", "")
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
            app = self.engine.router.resolve_app(query, known)
        base = self.engine._corpus_now(conn)
        target_date, target_hour = self.engine.router.resolve_datetime(query, slots, base)
        query_tokens = tokens(query)
        rows = conn.execute("SELECT * FROM transcripts ORDER BY created_at DESC").fetchall()
        semantic = bool(self.engine.embedder.describe().get("semantic"))
        query_vector = self.engine.embedder.embed_one(query) if semantic else None
        vectors = self.engine._load_embeddings(conn, "transcript") if query_vector is not None else {}
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
            return self.engine._abstain(
                "I could not find a transcript matching that app/time request.",
                "No transcript crossed the retrieval threshold.",
            )
        top = scored[0][1]
        polished = self.engine._polish_text(conn, top["formatted_text"])
        return {
            "answer": (
                f"I found this {top['app']} transcript from {top['created_at']}.\n\n"
                f"Polished version:\n{polished}"
            ),
            "abstained": False,
            "used_memories": self.engine._style_memories(conn),
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
        scored = score_memories(
            conn, query, self.engine.config, self.engine.embedder, self.engine._embedding_for_memory)
        selected = [item.row for item in scored][:4]

        # The abstention gate. Nothing below this point can invent an answer,
        # because nothing below this point runs unless a real memory cleared it.
        if not selected or scored[0].score < self.engine.config.answer_threshold:
            return self.engine._abstain(
                "I do not have enough supported memory to answer that.",
                "No active memory crossed the answer threshold.",
            )

        memories = [self.engine._memory_payload(conn, row) for row in selected]
        composed = self.engine.answerer.compose(query, selected, {"style": self._style_hint(conn)})
        if composed.get("abstain"):
            return self.engine._abstain(
                composed.get("answer") or "I do not have enough supported memory to answer that.",
                f"Answer backend ({composed.get('backend')}) declined to answer from the retrieved memories.",
            )

        cited = set(composed.get("cited_ids") or [])
        if cited:
            memories = [item for item in memories if item["id"] in cited] or memories

        detail = {item.row["id"]: item.as_detail() for item in scored}
        for item in memories:
            item["retrieval"] = detail.get(item["id"], {})

        return {
            "answer": composed["answer"],
            "abstained": False,
            "used_memories": memories,
            "source_transcripts": self.engine._sources_for_memories(conn, [item["id"] for item in memories]),
            "reason": (
                f"{describe_weights(self.engine.config, self.engine.embedder)} "
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
            return self.engine._abstain("There is nothing stored to correct yet.", "No active memories.")

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
        semantic = bool(self.engine.embedder.describe().get("semantic"))
        query_vector = self.engine.embedder.embed_one(target or query) if semantic else None

        best = None
        best_score = 0.0
        for row in rows:
            overlap = len(q_tokens & tokens(f"{row['canonical_text']} {row['object']} {row['predicate']}"))
            similarity = (
                cosine(query_vector, self.engine._embedding_for_memory(conn, row)) if query_vector is not None else 0.0
            )
            score = overlap * 0.6 + max(0.0, similarity) * 2.0
            if score > best_score:
                best, best_score = row, score

        if best is None or best_score < 0.8:
            return self.engine._abstain(
                "I could not tell which memory you want me to drop. Ask me what I remember and name one.",
                "No stored memory matched the correction target confidently enough.",
            )

        conn.execute("UPDATE memories SET status='archived', updated_at=? WHERE id=?", (utc_now(), best["id"]))
        payload = self.engine._memory_payload(conn, best)
        self.engine._record_decision(
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
            "source_transcripts": self.engine._sources_for_memories(conn, [best["id"]]),
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
