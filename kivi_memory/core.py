"""Shared primitives for the Kivi memory prototype.

Everything in here is backend-agnostic: it is used identically by the
deterministic rule path and by the local-LLM path, so the memory controller's
arithmetic never changes depending on which sensor produced a candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence


MEMORY_TYPES = ("preference", "fact", "project_state", "task", "event")

# Future-value prior per memory type. Used by the controller's utility
# function; never supplied by a model.
FUTURE_VALUE = {
    "preference": 0.92,
    "fact": 0.85,
    "project_state": 0.82,
    "task": 0.68,
    "event": 0.52,
}


def future_value(memory_type: str) -> float:
    return FUTURE_VALUE.get(memory_type, 0.4)


def utility_score(importance: float, confidence: float, memory_type: str) -> float:
    """Controller-owned utility function.

    Deliberately kept as the single definition in the codebase so that an LLM
    sensor cannot invent its own notion of usefulness.
    """
    return clamp(
        0.45 * float(importance)
        + 0.35 * float(confidence)
        + 0.20 * future_value(memory_type)
    )


def effective_confidence(
    confidence: float,
    decay_rate: float,
    recurrence: int,
    last_seen_at: str,
    now: datetime | None = None,
) -> float:
    """Time-decayed confidence.

    The prototype previously stored `decay_rate` on every memory and never read
    it. This applies it: confidence erodes exponentially with the age of the
    most recent supporting evidence, and repeated observation slows that
    erosion.

        conf_eff = conf * exp( -(decay_rate / (1 + ln(1 + recurrence))) * age_days )
    """
    try:
        last_seen = parse_dt(last_seen_at)
    except Exception:
        return clamp(float(confidence))
    now = now or datetime.utcnow()
    age_days = max(0.0, (now - last_seen).total_seconds() / 86400.0)
    damped = float(decay_rate) / (1.0 + math.log(1.0 + max(0, int(recurrence))))
    return clamp(float(confidence) * math.exp(-damped * age_days))


IRREGULAR_3SG = {
    "is": "are", "was": "were", "has": "have", "does": "do", "goes": "go",
    "prefers": "prefer", "says": "say", "needs": "need", "uses": "use", "wants": "want",
}


def to_second_person(sentence: str) -> str:
    """Rewrite a stored third-person claim as an answer to the user.

    "User prefers concise technical communication." -> "You prefer concise ..."
    "User's manager is Ravi."                       -> "Your manager is Ravi."

    This is English morphology, not domain knowledge: third-person singular
    verbs end in -s / -es / -ies, plus a short irregular list. It replaces the
    answerer's table of one canned sentence per predicate, so a memory with a
    predicate nobody anticipated still reads as a sentence.
    """
    text = sentence.strip()
    if not text:
        return text

    lowered = text.lower()
    if lowered.startswith("user's "):
        return "Your " + text[7:]
    if lowered.startswith("the user's "):
        return "Your " + text[11:]
    if not (lowered.startswith("user ") or lowered.startswith("the user ")):
        return text

    rest = text[5:] if lowered.startswith("user ") else text[9:]
    parts = rest.split(" ", 1)
    verb = parts[0]
    tail = f" {parts[1]}" if len(parts) > 1 else ""
    key = verb.lower()

    if key in IRREGULAR_3SG:
        verb = IRREGULAR_3SG[key]
    elif key.endswith("ies") and len(key) > 4:
        verb = verb[:-3] + "y"
    elif key.endswith(("ches", "shes", "sses", "xes", "zes")):
        verb = verb[:-2]
    elif key.endswith("s") and not key.endswith(("ss", "us", "is")):
        verb = verb[:-1]

    return f"You {verb}{tail}"


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "have", "he", "her", "hey", "i", "in", "into", "is", "it", "kivi",
    "me", "my", "of", "on", "or", "our", "please", "should", "that", "the",
    "this", "to", "use", "was", "we", "what", "when", "where", "who", "with",
    "you",
}

# Optional hand-written term mappings, applied during tokenisation.
#
# Shipped empty on purpose. This used to carry corpus-flavoured entries
# (`dictation -> transcript`, `prototype -> project`, plus several no-op
# identity mappings like `rust -> rust`) that quietly encoded the seed corpus's
# vocabulary into the retriever. Ablation after the tokenizer fix in
# `expand_identifier`: 11/11 with the table, 11/11 with it emptied - it was
# carrying nothing. The hook stays so a deployment can add real domain
# synonyms; the default is to let embeddings do this job.
SYNONYMS: dict[str, str] = {}

# Single-valued relations, decided structurally rather than by enumeration.
#
# The old version was a literal set of six predicate names. That cannot work
# once predicates are synthesised from the sentence: a corpus that produces
# `design_review_cadence_is` or `prefers_feedback` gets no conflict detection
# at all, because nobody put those names in a list. The shape of the predicate
# already says whether the relation can hold more than one value at a time.
EXCLUSIVE_PREDICATE_NAMES = {"works_at", "based_in", "is_a", "uses_tool"}

# Relations that are naturally multi-valued: two of them are not in conflict.
MULTI_VALUED_PREFIXES = ("has_", "avoids_", "needs_", "mentioned_", "instructed_")
MULTI_VALUED_NAMES = {"needs_to_do", "mentioned_event", "prefers_assistant_behaviour"}


def is_exclusive(predicate: str) -> bool:
    """True when a second value for this relation replaces the first.

    `manager_is` -> True   (one manager at a time)
    `prefers_feedback` -> True   (the topic is in the predicate, so a new value
                                  for the same topic is a change of mind)
    `needs_to_do` -> False  (two to-dos are not a contradiction)
    `has_task` -> False     (a person can have several)
    """
    name = (predicate or "").strip().lower()
    if not name or name in MULTI_VALUED_NAMES:
        return False
    if name.startswith(MULTI_VALUED_PREFIXES):
        return False
    if name in EXCLUSIVE_PREDICATE_NAMES:
        return True
    return name.endswith("_is") or name.startswith("prefers_")


# Kept for backwards compatibility with anything importing the old name.
EXCLUSIVE_PREDICATES = EXCLUSIVE_PREDICATE_NAMES


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part).strip().lower() for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def expand_identifier(value: str) -> list[str]:
    """Split a snake_case / colon-scoped identifier into its words.

    `prefers_response_style` -> ['prefers_response_style', 'prefers', 'response', 'style']
    `project:golden_goose`   -> ['project', 'golden', 'goose']

    This matters more than it looks. Predicate and scope names are written by
    whoever (or whatever) produced the candidate, and they already contain the
    words a user would ask with - "what response style do I prefer" against
    `prefers_response_style`. The original tokenizer kept `_` inside its
    character class, so those names stayed single opaque tokens and never
    matched anything, which is why retrieval needed a hand-written table of
    query-keyword -> predicate bonuses. Splitting them makes that association
    fall out of the text for every predicate, including ones nobody enumerated.
    """
    parts = [value]
    parts.extend(re.split(r"[_:\-./]+", value))
    return [part for part in parts if part]


def stem(word: str) -> str:
    """Strip regular English inflection.

    Grammar, not vocabulary: "prefers"/"prefer", "explanations"/"explanation",
    "reviewing"/"review". Without this, a query saying "what do I prefer"
    cannot match a memory whose predicate is `prefers_explanations`, which is
    the kind of gap that previously got papered over with a hand-written
    synonym table.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        base = word[:-3]
        return base[:-1] if len(base) > 3 and base[-1] == base[-2] else base
    if len(word) > 4 and word.endswith("ed") and not word.endswith("eed"):
        return word[:-2]
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is", "as")):
        return word[:-1]
    return word


def tokens(text: str) -> set[str]:
    values = set()
    for raw in re.findall(r"[a-zA-Z0-9_:.\-]+", text.lower()):
        for token in expand_identifier(raw):
            mapped = SYNONYMS.get(token, token)
            if len(mapped) > 2 and mapped not in STOPWORDS:
                values.add(mapped)
                stemmed = stem(mapped)
                if stemmed != mapped and len(stemmed) > 2 and stemmed not in STOPWORDS:
                    values.add(stemmed)
    return values


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def snippet(text: str, needle: str | None = None, limit: int = 220) -> str:
    clean = normalize_text(text)
    if len(clean) <= limit:
        return clean
    if needle:
        idx = clean.lower().find(needle.lower())
        if idx >= 0:
            start = max(0, idx - limit // 3)
            end = min(len(clean), start + limit)
            return clean[start:end].strip()
    return clean[:limit].rstrip() + "..."

@dataclass
class Candidate:
    """A structured observation produced by a semantic sensor.

    Sensors (rule-based or LLM-backed) fill in everything except `utility` and
    `status`, which the controller derives. Keeping this object identical
    across backends is what lets the extractor be swapped without touching the
    controller, the schema, or the audit trail.
    """

    memory_type: str
    subject: str
    predicate: str
    object: str
    scope: str
    canonical_text: str
    evidence: str
    importance: float
    confidence: float
    utility: float
    decay_rate: float
    status: str = "active"
    reason: str = ""
    source: str = "rule"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return stable_id(
            "mem",
            self.memory_type,
            self.subject,
            self.predicate,
            self.scope,
            self.object,
        )

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def embedding_text(self) -> str:
        return " | ".join([
            self.canonical_text,
            self.memory_type,
            self.predicate,
            self.object,
            self.scope,
        ])
