"""The model-backed sensor: an instruct model reading a transcript.

Emits exactly the same `Candidate` objects as `RuleExtractor`, which is the
constraint that makes the two interchangeable. The model proposes typed
observations with an evidence span and a self-reported confidence; `utility`,
`status` and every ADD/UPDATE/REJECT decision are computed afterwards by
`controller.py`. Swapping sensors therefore cannot change the memory policy.

Generation is schema-constrained at temperature 0, and anything that does not
validate is dropped rather than repaired - a malformed candidate is a sensor
fault, and guessing at what it meant would put the model back in the position
this architecture takes it out of.

If the daemon is unreachable the sensor falls back to the frames rather than
returning nothing, because an unavailable sensor must never be read as "this
transcript contained nothing worth remembering".
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import KiviConfig
from .core import (
    Candidate,
    clamp,
    normalize_text,
    snippet,
    utility_score,
)
from .llm import OllamaClient
from .rule_sensor import RuleExtractor


# ---------------------------------------------------------------------------
# LLM-backed sensor
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You extract durable memory candidates from a single voice-dictation transcript.

You are a SENSOR, not a memory. You do not decide what gets stored; a separate
deterministic controller does that. Your only job is to report what the
transcript says, in structure.

Return JSON with exactly two arrays: "candidates" and "rejected".

Each candidate:
  type          one of: preference, fact, project_state, task, event
  predicate     snake_case relation, reused across transcripts so the same
                fact always lands on the same predicate. Prefer these when they
                fit: prefers_response_style, prefers_response_format,
                prefers_language, uses_language, works_at, based_in,
                manager_is, team_is, has_assignment, focuses_on, needs_to_do,
                mentioned_event
  object        the value, lowercase, minimal (e.g. "python", "bangalore",
                "concise technical communication"). No sentences.
  scope         "communication" | "programming" | "profile" | "product" |
                "current_tasks" | "recent_events" | "project:<slug>"
  claim         one plain sentence a person could read back, e.g.
                "User prefers concise technical communication."
  evidence      the exact substring of the transcript that supports it
  importance    0.0-1.0, how much this should shape future assistance
  confidence    0.0-1.0, how certain you are the transcript actually says it
  hedged        true if phrased with maybe/might/considering/thinking about
  correction    true if it overrides something previously true ("actually",
                "no longer", "switched to", "now my")

Each rejected item: {"text": ..., "reason": ...} - things a naive system would
store but that have no future utility (small talk, weather, meals, one-off
chatter, transient state). Be explicit; these are shown to the user.

Rules:
- Never invent content that is not in the transcript.
- Do not emit a candidate for ephemeral chatter; put it in "rejected".
- If the user says not to remember something, reject it and say so.
- Return {"candidates": [], "rejected": [...]} when nothing durable is present.
Reply with JSON only."""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["preference", "fact", "project_state", "task", "event"]},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "scope": {"type": "string"},
                    "claim": {"type": "string"},
                    "evidence": {"type": "string"},
                    "importance": {"type": "number"},
                    "confidence": {"type": "number"},
                    "hedged": {"type": "boolean"},
                    "correction": {"type": "boolean"},
                },
                "required": ["type", "predicate", "object", "scope", "claim", "evidence", "importance", "confidence"],
            },
        },
        "rejected": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
    "required": ["candidates", "rejected"],
}

DEFAULT_DECAY = {
    "preference": 0.015,
    "fact": 0.005,
    "project_state": 0.05,
    "task": 0.14,
    "event": 0.18,
}


def _as_float(value: Any, default: float) -> float:
    try:
        return clamp(float(value))
    except (TypeError, ValueError):
        return default


class LlmExtractor:
    """Local-LLM semantic sensor. Emits the same Candidate objects as RuleExtractor."""

    def __init__(self, client: OllamaClient, config: KiviConfig) -> None:
        self.client = client
        self.config = config
        self.degraded = False
        self.fallback = RuleExtractor()

    def describe(self) -> dict[str, Any]:
        return {
            "backend": f"ollama:{self.config.llm_model}",
            "degraded_to_rules": self.degraded,
        }

    def extract(self, record: dict[str, Any]) -> tuple[list[Candidate], list[str]]:
        text = normalize_text(record.get("formatted_text") or record.get("raw_asr") or "")
        if not text:
            return [], ["Empty transcript."]

        payload = self.client.chat_json(
            EXTRACTION_SYSTEM,
            self._prompt(record, text),
            schema=EXTRACTION_SCHEMA,
            cache_kind="extract",
        )
        if payload is None:
            # The daemon is down or returned nothing parseable. Falling back is
            # correct here: an unavailable sensor must not be read as "this
            # transcript contained nothing worth remembering".
            self.degraded = True
            candidates, ignored = self.fallback.extract(record)
            return candidates, ignored + ["LLM extractor unavailable; fell back to deterministic rules."]

        self.degraded = False
        candidates = []
        for raw in payload.get("candidates") or []:
            candidate = self._to_candidate(raw, text, record)
            if candidate is not None:
                candidates.append(candidate)

        ignored = []
        for raw in payload.get("rejected") or []:
            if isinstance(raw, dict):
                reason = str(raw.get("reason") or "").strip()
                sample = str(raw.get("text") or "").strip()
                if reason:
                    ignored.append(f"{reason}{f' Source: {snippet(sample, limit=90)}' if sample else ''}")
        if not candidates and not ignored:
            ignored.append("Model found no durable fact, preference, task, project state, or useful episode.")
        return candidates, ignored

    def _prompt(self, record: dict[str, Any], text: str) -> str:
        # Deliberately excludes `created_at`: the timestamp is not something the
        # sensor should reason about (the controller owns first_seen/last_seen
        # and decay), and leaving it out makes the prompt content-addressable,
        # so repeated dictations hit the response cache instead of the model.
        # On the 500-record seed corpus that is roughly a 5x reduction in calls.
        return json.dumps(
            {
                "app": record.get("app"),
                "raw_asr": snippet(record.get("raw_asr") or "", limit=700),
                "formatted_text": text,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _to_candidate(self, raw: Any, text: str, record: dict[str, Any]) -> Candidate | None:
        if not isinstance(raw, dict):
            return None
        memory_type = str(raw.get("type", "")).strip().lower()
        if memory_type not in DEFAULT_DECAY:
            return None
        predicate = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("predicate", "")).strip().lower()).strip("_")
        object_ = normalize_text(str(raw.get("object", ""))).strip().lower()
        if not predicate or not object_:
            return None

        scope = str(raw.get("scope") or "global").strip().lower() or "global"
        scope = re.sub(r"[^a-z0-9_:]+", "_", scope).strip("_") or "global"
        claim = normalize_text(str(raw.get("claim") or "")) or f"User: {predicate} = {object_}."
        evidence = normalize_text(str(raw.get("evidence") or ""))
        if evidence.lower() not in text.lower():
            # The model paraphrased its own evidence. Keep provenance honest by
            # anchoring on the transcript instead of trusting the quote.
            evidence = snippet(text, object_)

        importance = _as_float(raw.get("importance"), 0.6)
        confidence = _as_float(raw.get("confidence"), 0.6)
        hedged = bool(raw.get("hedged"))
        correction = bool(raw.get("correction"))

        # Same modifiers the rule path applies, so the two sensors are
        # calibrated against one controller rather than two policies.
        if hedged:
            confidence -= 0.34
        if correction:
            confidence += 0.12
            importance += 0.05
        confidence = clamp(confidence)
        importance = clamp(importance)

        return Candidate(
            memory_type=memory_type,
            subject="user",
            predicate=predicate,
            object=object_,
            scope=scope,
            canonical_text=claim,
            evidence=snippet(evidence or text, object_),
            importance=round(importance, 3),
            confidence=round(confidence, 3),
            utility=round(utility_score(importance, confidence, memory_type), 3),
            decay_rate=DEFAULT_DECAY[memory_type],
            status="tentative" if hedged and confidence < self.config.tentative_confidence else "active",
            reason=str(raw.get("reason") or "Structured observation from local LLM sensor."),
            source="llm",
            metadata={
                "app": record.get("app"),
                "hedged": hedged,
                "correction": correction,
                "model": self.config.llm_model,
            },
        )
