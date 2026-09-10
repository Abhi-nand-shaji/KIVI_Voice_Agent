"""Which memories are relevant to a question, and how sure we are.

Retrieval is a weighted sum over four signals, all of them owned by the
controller rather than by any model:

    score = w_lexical   * overlap            # shared content words
          + w_semantic  * cosine             # embedding similarity
          + w_confidence * decayed_confidence
          + w_utility   * utility
          - tentative_penalty                # if the memory is provisional

`cosine` is not raw cosine: it is rescaled above `semantic_floor`, because a
dense embedding scores ~0.45 between two unrelated sentences and that constant
lift is enough on its own to defeat the abstention gate. See `KiviConfig`.

`overlap` is the fraction of the QUESTION's content words the memory accounts
for, so both it and `cosine` are in [0, 1] and their weights are directly
comparable. It used to be an unbounded token count, which made the weights live
on different scales and - more importantly - made `answer_threshold` mean
different things for a three-word question and a fifteen-word one. A memory
sharing one generic word with a long question could clear the gate on raw count
alone; now it has to account for a real share of what was asked.

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

import math
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


def inverse_document_frequency(texts: list[set[str]]) -> dict[str, float]:
    """How discriminative each token is across the stored memories.

    Plain overlap counting treats every shared word alike, and that is what
    made retrieval answer the wrong question. Asking "what do I know about my
    Kivi memory project" shares `project` with seven different memories and
    shares nothing with any of them uniquely; asking "what response style do I
    prefer" shares `prefer` with four. Raw coverage ranks the first match
    higher (2 shared words out of 4, versus 1 out of 3) even though its shared
    words carry almost no information.

    Weighting by `log(1 + N / (1 + df))` says the obvious thing instead: a word
    that appears in most memories cannot tell them apart, so matching on it is
    weak evidence. This is a property of the corpus, computed at query time -
    there is no list of important words anywhere.
    """
    total = len(texts)
    frequency: dict[str, int] = {}
    for text_tokens in texts:
        for token in text_tokens:
            frequency[token] = frequency.get(token, 0) + 1
    return {
        token: math.log(1.0 + total / (1.0 + count))
        for token, count in frequency.items()
    }


def weighted_coverage(query_tokens: set[str], other: set[str], idf: dict[str, float]) -> float:
    """Share of the query's *information* the other text accounts for, in [0, 1].

    Falls back to unweighted coverage when the query has no token the corpus
    has ever seen, which is the case where idf carries no signal anyway.
    """
    if not query_tokens:
        return 0.0
    default = max(idf.values(), default=1.0)
    mass = sum(idf.get(token, default) for token in query_tokens)
    if mass <= 0.0:
        return 0.0
    matched = sum(idf.get(token, default) for token in query_tokens & other)
    return matched / mass


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
    something the policy already rejected. Lexical overlap is weighted by how
    discriminative each shared word is across the memory set, so confidence and
    utility - which are properties of the memory, not of its fit to this
    question - can no longer carry a barely-related row over the threshold on a
    single common word.
    """
    rows = conn.execute(
        "SELECT * FROM memories WHERE status IN ('active', 'tentative')"
    ).fetchall()

    q_tokens = tokens(query)
    semantic = bool(embedder.describe().get("semantic"))
    query_vector = embedder.embed_one(query) if semantic else None
    now = now or datetime.utcnow()

    row_tokens = [tokens(memory_text(row)) for row in rows]
    idf = inverse_document_frequency(row_tokens)

    scored: list[ScoredMemory] = []
    for row, memory_tokens in zip(rows, row_tokens):
        overlap = weighted_coverage(q_tokens, memory_tokens, idf)
        conf_eff = effective_confidence(
            row["confidence"], row["decay_rate"], row["recurrence"], row["last_seen_at"], now
        )
        similarity = 0.0
        if query_vector is not None:
            raw = cosine(query_vector, embedding_for_memory(conn, row))
            # Rescale so that "no better than an unrelated sentence" contributes
            # zero, rather than the constant lift a raw cosine gives every row.
            floor = config.semantic_floor
            similarity = max(0.0, (raw - floor) / (1.0 - floor)) if floor < 1.0 else 0.0

        # Noise floor only. A flat coverage threshold was tried here and is the
        # wrong instrument: the bad match in the motivating example had HIGHER
        # raw coverage (0.50) than the good one (0.33). Specificity is what
        # separates them, so that job belongs to the idf weighting above and to
        # `answer_threshold`; this just drops rows with nothing in common.
        if overlap <= config.min_query_coverage and similarity <= 0.0:
            continue

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
