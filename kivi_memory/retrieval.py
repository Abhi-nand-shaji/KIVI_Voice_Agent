"""Which memories are relevant to a question, and how sure we are.

Retrieval is a weighted sum over four signals, all of them owned by the
controller rather than by any model:

    score = w_lexical   * overlap            # shared content words
          + w_semantic  * cosine             # embedding similarity
          + w_confidence * decayed_confidence
          + w_utility   * utility
          - tentative_penalty                # if the memory is provisional

`overlap` is an unbounded integer count while `cosine` is in [0, 1], which is
why `w_semantic` (1.60) is numerically larger than `w_lexical` (0.55) - they are
weights on different scales, not a statement that meaning matters three times
as much as words.

The decay term is the reason `decay_rate` exists on every row. Confidence is
not a fixed number: it erodes with the age of the most recent supporting
evidence, and recurrence flattens that curve rather than bumping it once, so a
preference stated every week asymptotically stops decaying while a one-off
remark quietly drops out of contention without anything deleting it.

Two floors keep the system honest. `retrieval_floor` prunes rows that are
merely present; `answer_threshold` is the abstention gate - if the best row
does not clear it, the system says it does not know instead of composing an
answer from whatever ranked highest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .config import KiviConfig
from .core import cosine, effective_confidence, tokens


@dataclass
class ScoredMemory:
    """One candidate answer row, with the parts of its score kept separate.

    The breakdown is carried through to the interface's "Why?" panel, so a
    ranking can be inspected rather than taken on trust.
    """

    score: float
    similarity: float
    decayed_confidence: float
    row: Any

    def as_detail(self) -> dict[str, float]:
        return {
            "score": round(self.score, 3),
            "similarity": round(self.similarity, 3),
            "decayed_confidence": round(self.decayed_confidence, 3),
        }


def memory_text(row: Any) -> str:
    """The text a memory is matched against.

    Includes the predicate and scope, not just the sentence, because those are
    snake_case identifiers whose words are exactly what a person asks with -
    "what response style do I prefer" against `prefers_response_style`.
    `core.expand_identifier` splits them during tokenisation.
    """
    return " ".join([
        row["canonical_text"],
        row["memory_type"],
        row["predicate"],
        row["object"],
        row["scope"],
    ])


def score_memories(
    conn,
    query: str,
    config: KiviConfig,
    embedder,
    embedding_for_memory: Callable[[Any, Any], list[float]],
    now: datetime | None = None,
) -> list[ScoredMemory]:
    """Rank every active memory against the query, best first.

    Rows scoring at or below `retrieval_floor` are dropped entirely rather than
    returned with a low score, so the caller cannot accidentally answer from
    something the policy already rejected.
    """
    rows = conn.execute(
        "SELECT * FROM memories WHERE status IN ('active', 'tentative')"
    ).fetchall()

    q_tokens = tokens(query)
    semantic = bool(embedder.describe().get("semantic"))
    query_vector = embedder.embed_one(query) if semantic else None
    now = now or datetime.utcnow()

    scored: list[ScoredMemory] = []
    for row in rows:
        overlap = len(q_tokens & tokens(memory_text(row)))
        conf_eff = effective_confidence(
            row["confidence"], row["decay_rate"], row["recurrence"], row["last_seen_at"], now
        )
        similarity = 0.0
        if query_vector is not None:
            similarity = max(0.0, cosine(query_vector, embedding_for_memory(conn, row)))

        score = (
            overlap * config.w_lexical
            + similarity * config.w_semantic
            + conf_eff * config.w_confidence
            + row["utility"] * config.w_utility
        )
        if row["status"] == "tentative":
            score -= config.tentative_penalty
        if score > config.retrieval_floor:
            scored.append(ScoredMemory(score, similarity, conf_eff, row))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def describe_weights(config: KiviConfig, embedder) -> str:
    """The scoring formula as a sentence, for the audit trail."""
    return (
        f"Hybrid retrieval: {config.w_lexical}*lexical + {config.w_semantic}*cosine("
        f"{embedder.describe().get('backend')}) + {config.w_confidence}*decayed_confidence + "
        f"{config.w_utility}*utility, tentative penalty {config.tentative_penalty}."
    )
