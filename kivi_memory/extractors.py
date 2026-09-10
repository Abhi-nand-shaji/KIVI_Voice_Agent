"""Choosing and combining semantic sensors.

Two sensors produce the *same* `Candidate` objects - `RuleExtractor` in
`rule_sensor.py` and `LlmExtractor` in `llm_sensor.py`. This module is only
about how they are combined, and that turns out to be the interesting part:

    rule      frames only, 0 model calls
    llm       model only, frames never run
    hybrid    both on every record, unioned, frames win ties
    cascade   frames first; the model only where they demonstrably fell short

`cascade` is the one that matters, because *when to spend the model* is a
controller decision like any other: its thresholds live in `KiviConfig`
alongside the admission thresholds, and `tests/cascade_curve.py` sweeps them.
"""

from __future__ import annotations

from typing import Any

from .config import KiviConfig
from .core import Candidate, normalize_text, strip_disfluency, tokens
from .llm import OllamaClient
from .llm_sensor import DEFAULT_DECAY, EXTRACTION_SCHEMA, EXTRACTION_SYSTEM, LlmExtractor
from .rule_sensor import (
    IGNORED_EPHEMERAL,
    IGNORED_NO_CLAIM,
    IGNORED_SUPPRESSED,
    SETTLED_REASONS,
    RuleExtractor,
)

# Re-exported so `from .extractors import RuleExtractor` keeps working: this
# module is the package's front door for sensors, whichever file defines them.
__all__ = [
    "RuleExtractor", "LlmExtractor", "HybridExtractor", "CascadeExtractor",
    "get_extractor", "IGNORED_SUPPRESSED", "IGNORED_EPHEMERAL", "IGNORED_NO_CLAIM",
    "SETTLED_REASONS", "EXTRACTION_SYSTEM", "EXTRACTION_SCHEMA", "DEFAULT_DECAY",
]


def _merge(
    rule_candidates: list[Candidate],
    rule_ignored: list[str],
    llm_candidates: list[Candidate],
    llm_ignored: list[str],
) -> tuple[list[Candidate], list[str]]:
    """Union two sensors' output, deduplicated on candidate identity.

    Frames win ties: where both fire on the same (type, predicate, scope,
    object), the deterministic candidate is kept, so the reproducible path
    stays authoritative and the model only adds recall.
    """
    merged: dict[str, Candidate] = {c.id: c for c in llm_candidates}
    merged.update({c.id: c for c in rule_candidates})
    ignored = rule_ignored + [item for item in llm_ignored if item not in rule_ignored]
    if merged:
        # "nothing here" notes from either sensor are wrong once the other found
        # something; drop them rather than showing a contradictory audit line.
        ignored = [
            item for item in ignored
            if "No durable" not in item and item != IGNORED_NO_CLAIM
        ]
    return list(merged.values()), ignored


class HybridExtractor:
    """Both sensors on every record, unioned. One model call per transcript."""

    def __init__(self, rule: RuleExtractor, llm: LlmExtractor) -> None:
        self.rule = rule
        self.llm = llm

    def describe(self) -> dict[str, Any]:
        return {"backend": "hybrid(rule+llm)", **self.llm.describe()}

    def extract(self, record: dict[str, Any]) -> tuple[list[Candidate], list[str]]:
        return _merge(*self.rule.extract(record), *self.llm.extract(record))


class CascadeExtractor:
    """Frames first; the model only where the frames demonstrably fell short.

    `HybridExtractor` pays for a model call on every transcript, including the
    large majority that a frame parses cleanly. That is the dominant cost in
    ingestion and almost all of it buys nothing.

    This escalates instead, on three signals - and the *controller* owns the
    thresholds, exactly as it owns the admission thresholds:

      1. no frame matched at all;
      2. the best matching frame was not confident (hedged speech, a vague
         frame like `event_with`) - `escalate_below`;
      3. the frames matched, but their candidates account for less than
         `escalate_coverage` of the transcript's content words, i.e. most of
         what was said went unparsed.

    (3) is the one a naive confidence gate misses. "My manager is Neha and I
    prefer detailed written feedback" can match `possessive` at confidence 0.84
    while the entire second clause is dropped on the floor. High confidence,
    half the sentence lost.

    `escalation_rate` is recorded so the cost/recall trade-off is measurable
    rather than asserted: sweeping `escalate_below` from 0 to 1 traces a curve
    from zero model calls to one per record.
    """

    def __init__(self, rule: RuleExtractor, llm: LlmExtractor, config: KiviConfig) -> None:
        self.rule = rule
        self.llm = llm
        self.config = config
        self.seen = 0
        self.escalated = 0

    def describe(self) -> dict[str, Any]:
        described = dict(self.llm.describe())
        described.pop("backend", None)   # the cascade's backend, not the model's
        return {
            "backend": "cascade(rule->llm)",
            "model": self.llm.describe().get("backend"),
            "escalate_below": self.config.escalate_below,
            "escalate_coverage": self.config.escalate_coverage,
            "records_seen": self.seen,
            "records_escalated": self.escalated,
            "escalation_rate": round(self.escalated / self.seen, 3) if self.seen else 0.0,
            **described,
        }

    @staticmethod
    def coverage(text: str, candidates: list[Candidate]) -> float:
        """Fraction of the transcript's content words the candidates account for.

        Compared on stemmed content tokens rather than character spans, because
        a candidate's `evidence` is a snippet that may be truncated, and its
        predicate carries words the object does not ("prefers_feedback" covers
        both "prefer" and "feedback").
        """
        text_tokens = tokens(text)
        if not text_tokens:
            return 1.0
        covered: set[str] = set()
        for candidate in candidates:
            covered |= tokens(candidate.object) | tokens(candidate.predicate)
        return len(text_tokens & covered) / len(text_tokens)

    def _escalation_reason(
        self, text: str, candidates: list[Candidate], ignored: list[str] | None = None
    ) -> str | None:
        # A settled reason is the frames saying "there is nothing here", not
        # "I could not parse this". Escalating on those is wrong twice over: it
        # spends a model call to learn nothing, and on IGNORED_SUPPRESSED it
        # would let the model extract from a transcript the person explicitly
        # asked not to be remembered. Measured on the 500-record corpus, these
        # account for 136 of the 324 no-candidate records.
        if any(reason in SETTLED_REASONS for reason in (ignored or [])):
            return None
        if not candidates:
            return "no frame matched"
        best = max(c.confidence for c in candidates)
        if best < self.config.escalate_below:
            return f"best frame confidence {best:.2f} < {self.config.escalate_below}"
        covered = self.coverage(text, candidates)
        if covered < self.config.escalate_coverage:
            return f"frames covered {covered:.0%} of the transcript < {self.config.escalate_coverage:.0%}"
        return None

    def extract(self, record: dict[str, Any]) -> tuple[list[Candidate], list[str]]:
        text = strip_disfluency(normalize_text(
            record.get("formatted_text") or record.get("raw_asr") or ""))
        rule_candidates, rule_ignored = self.rule.extract(record)

        self.seen += 1
        reason = self._escalation_reason(text, rule_candidates, rule_ignored)
        if reason is None:
            return rule_candidates, rule_ignored

        self.escalated += 1
        llm_candidates, llm_ignored = self.llm.extract(record)
        for candidate in llm_candidates:
            candidate.metadata["escalated_because"] = reason
        return _merge(rule_candidates, rule_ignored, llm_candidates, llm_ignored)


def get_extractor(config: KiviConfig, client: OllamaClient | None):
    mode = config.extractor
    if mode == "rule" or client is None:
        return RuleExtractor()
    if mode == "llm":
        return LlmExtractor(client, config)
    if mode == "hybrid":
        return HybridExtractor(RuleExtractor(), LlmExtractor(client, config))
    if mode == "cascade":
        return CascadeExtractor(RuleExtractor(), LlmExtractor(client, config), config)
    # auto
    if client.available() and client.has_model(config.llm_model):
        return LlmExtractor(client, config)
    return RuleExtractor()
