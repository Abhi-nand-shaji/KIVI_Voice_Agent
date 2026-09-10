"""The memory controller: what is remembered, changed, or thrown away.

This is the deterministic heart of the system, and deliberately the part that
did NOT change when the sensors became model-backed. A sensor - regex frames or
a local instruct model - proposes `Candidate` objects and stops there. Every
decision about those candidates is made here, by arithmetic, from the
thresholds in `KiviConfig`.

The ladder, first match wins:

  1. exact key already stored                     -> REINFORCE
  1b. same relation, object names the same thing  -> REINFORCE (dedup, no model)
  2. NLI says a stored memory means the same      -> REINFORCE (paraphrase merge)
  3. a stored memory contradicts it, and
     confidence >= supersede_confidence           -> UPDATE (archive the old)
  4. utility >= admit_utility and
     confidence >= admit_confidence               -> ADD / ADD_TENTATIVE
  5. otherwise                                    -> REJECT

NLI is consulted, but never decides: `nli_conflict_types` restricts which
memory types inference is even allowed to archive, because two different
to-dos are not a contradiction and a model calling them one would silently
delete the user's task list.

Every rung writes a row to `decisions` carrying the candidate, the reason, both
scalars and the NLI verdict, so "why do you know this?" is a join rather than a
generated rationalisation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .config import KiviConfig
from .core import Candidate, clamp, cosine, is_exclusive, same_thing, stable_id, utc_now
from .nli import NLIResult


class MemoryController:
    """Owns the write-path policy. Sensors propose; this disposes.

    The two embedding callables are injected rather than imported, so the
    controller has no opinion about where vectors are cached: it only needs a
    way to fetch one for a stored row when ranking the neighbourhood, and a
    way to persist one for a memory it has just admitted.
    """

    def __init__(self, config: KiviConfig, nli, embedder,
                 embedding_for_memory, store_embedding) -> None:
        self.config = config
        self.nli = nli
        self.embedder = embedder
        self.embedding_for_memory = embedding_for_memory
        self.store_embedding = store_embedding

    def apply(self, conn, record: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
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
            return self.update_memory(conn, record, candidate, existing)

        # A near-duplicate of something already known is a reinforcement, not a
        # second memory. Exact-key matching cannot see this ("concise
        # explanation" vs "concise technical explanations"); token containment
        # can, and unlike NLI it needs no model.
        duplicate = self.find_near_duplicate(conn, candidate)
        if duplicate is not None:
            return self.update_memory(conn, record, candidate, duplicate)

        # A paraphrase of something already known is a reinforcement, not a
        # second memory. Exact-key matching cannot see this ("bangalore" vs
        # "bangalore, india"); bidirectional entailment can.
        twin, twin_nli = self.find_equivalent(conn, candidate)
        if twin is not None:
            return self.update_memory(conn, record, candidate, twin, nli=twin_nli)

        conflict, conflict_nli = self.find_conflict(conn, candidate)
        if (
            conflict is not None
            and candidate.status == "active"
            and candidate.confidence >= self.config.supersede_confidence
        ):
            conn.execute(
                "UPDATE memories SET status='archived', updated_at=? WHERE id=?",
                (utc_now(), conflict["id"]),
            )
            self.insert_memory(conn, record, candidate, supersedes_id=conflict["id"])
            basis = (
                f"NLI {conflict_nli.label} p={conflict_nli.probability:.2f} via {conflict_nli.backend}"
                if conflict_nli else "exclusive-predicate rule"
            )
            return self.record_decision(
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
            self.insert_memory(conn, record, candidate)
            action = "ADD_TENTATIVE" if candidate.status == "tentative" else "ADD"
            return self.record_decision(
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

        return self.record_decision(
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

    def neighbourhood(self, conn, candidate: Candidate, limit: int = 6) -> list[Any]:
        """Stored memories that plausibly talk about the same thing.

        Restricted to the same subject, then ranked by embedding similarity, so
        NLI is asked at most `limit` questions per candidate instead of once
        per stored memory.

        Scope used to be part of the WHERE clause. That was the single largest
        source of duplicate memories: a project mentioned as "the Golden Goose
        task" and "the Golden Goose prototype" produces two scope slugs, and
        restricting comparison to matching slugs meant neither NLI nor the
        structural rule could ever see the two claims as being about one thing.
        Scope is now a ranking bonus - same scope is evidence that two memories
        are related, not a precondition for asking.
        """
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE subject=? AND status IN ('active', 'tentative') AND id<>?
            """,
            (candidate.subject, candidate.id),
        ).fetchall()
        if not rows:
            return []
        query_vector = self.embedder.embed_one(candidate.embedding_text())
        scored = []
        for row in rows:
            similarity = cosine(query_vector, self.embedding_for_memory(conn, row))
            same_predicate = row["predicate"] == candidate.predicate
            same_scope = row["scope"] == candidate.scope
            if similarity >= self.config.semantic_conflict_similarity or same_predicate:
                bonus = (0.25 if same_predicate else 0.0) + (0.15 if same_scope else 0.0)
                scored.append((similarity + bonus, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:limit]]

    def find_near_duplicate(self, conn, candidate: Candidate) -> Any | None:
        """The same claim, worded or scoped slightly differently.

        Runs with no model at all, which is the point: NLI catches paraphrase
        only when it is switched on, and the deterministic path was therefore
        accumulating `prefers_explanation = "concise explanation"` alongside
        `prefers_explanation = "concise technical explanations"` as two separate
        preferences that never reconciled.

        The test is deliberately narrow - same type, same subject, same
        predicate, and objects that name the same thing by `core.same_thing`.
        Two values of a genuinely multi-valued relation are left alone, because
        merging two different to-dos would silently delete one.
        """
        if not is_exclusive(candidate.predicate):
            return None
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE memory_type=? AND subject=? AND predicate=?
              AND status IN ('active', 'tentative') AND id<>?
            ORDER BY recurrence DESC, confidence DESC
            """,
            (candidate.memory_type, candidate.subject, candidate.predicate, candidate.id),
        ).fetchall()
        for row in rows:
            if same_thing(row["object"], candidate.object):
                return row
        return None

    def find_equivalent(self, conn, candidate: Candidate) -> tuple[Any | None, NLIResult | None]:
        if self.nli.name == "off":
            return None, None
        for row in self.neighbourhood(conn, candidate):
            if row["object"] == candidate.object and row["predicate"] == candidate.predicate:
                continue  # exact-key path already handled this
            same, result = self.nli.equivalent(
                row["canonical_text"], candidate.canonical_text, self.config.nli_equivalence_threshold
            )
            if same:
                return row, result
        return None, None

    def find_conflict(self, conn, candidate: Candidate) -> tuple[Any | None, NLIResult | None]:
        """Conflict detection: NLI first, exclusive-predicate whitelist as backstop.

        The whitelist is kept because it is cheap, exact, and correct for the
        predicates it covers. NLI extends the same decision to everything else,
        including predicates nobody enumerated in advance.
        """
        if self.nli.name != "off" and candidate.memory_type in self.config.nli_conflict_types:
            for row in self.neighbourhood(conn, candidate):
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

    def insert_memory(self, conn, record: dict[str, Any], candidate: Candidate, supersedes_id: str | None = None) -> None:
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
        self.store_embedding(conn, "memory", candidate.id, candidate.embedding_text())
        self.insert_evidence(conn, candidate.id, record, "supports", candidate.evidence, candidate.confidence)

    def update_memory(
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
        self.insert_evidence(conn, existing["id"], record, "supports", candidate.evidence, candidate.confidence)
        if nli is not None:
            note = (
                f"Merged a paraphrase of an existing memory (NLI {nli.label} "
                f"p={nli.probability:.2f} via {nli.backend}); recurrence #{recurrence}."
            )
        else:
            note = f"Matched existing memory and raised confidence through recurrence #{recurrence}."
        return self.record_decision(
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

    def insert_evidence(
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

    def record_decision(
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
