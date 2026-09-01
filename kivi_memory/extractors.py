"""Semantic sensors.

Two backends produce the *same* `Candidate` objects:

* `RuleExtractor`  - the original deterministic regex/keyword sensors. Fast,
  reproducible, and honest about being narrow.
* `LlmExtractor`   - a locally hosted instruct model that reads an arbitrary
  transcript and returns structured observations.

The important constraint: the model never decides what is remembered. It
proposes typed observations with an evidence span and a self-reported
confidence; `utility`, `status`, and every ADD/UPDATE/REJECT decision are
computed by the controller from `core.utility_score` and the thresholds in
`KiviConfig`. Swapping sensors therefore cannot change the memory policy.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from typing import Any

from .config import KiviConfig
from .core import (
    Candidate,
    clamp,
    future_value,
    normalize_text,
    parse_dt,
    snippet,
    utility_score,
)
from .frames import (
    CORRECTION_MARKERS,
    EMPHASIS_MARKERS,
    FRAMES,
    HEDGE_MARKERS,
    Frame,
    head_noun,
)
from .llm import OllamaClient


class RuleExtractor:
    """Domain-free deterministic sensor.

    Runs the grammatical frames in `frames.py` over each sentence. It knows
    nothing about any particular corpus: predicates are synthesised from the
    sentence ("my manager is Neha" -> `manager_is`, "I prefer long written
    feedback" -> `prefers_feedback`), so the same code learns from a designer's
    dictations and an engineer's without editing.

    It is still a rule system and still the weaker of the two sensors. What it
    is not any more is a recogniser for one specific test corpus.
    """

    SENTENCE = re.compile(r"(?<=[.!?])\s+")

    def extract(self, record: dict[str, Any]) -> tuple[list[Candidate], list[str]]:
        text = normalize_text(record.get("formatted_text") or record.get("raw_asr") or "")
        lower = text.lower()
        if self._explicitly_low_value(lower):
            return [], ["The user explicitly asked for this not to be remembered."]

        candidates: dict[str, Candidate] = {}
        for sentence in self._sentences(text):
            claimed_families: set[str] = set()
            for frame in FRAMES:
                if frame.family and frame.family in claimed_families:
                    continue  # a more specific frame in this family already matched
                matched = False
                for match in frame.compiled().finditer(sentence):
                    built = frame.build(match)
                    if not built:
                        continue
                    for payload in [built] + list(built.get("extra") or []):
                        candidate = self._candidate_from_frame(frame, payload, sentence, record)
                        if candidate is not None:
                            candidates.setdefault(candidate.id, candidate)
                            matched = True
                if matched and frame.family:
                    claimed_families.add(frame.family)

        ignored: list[str] = []
        if not candidates:
            if self._looks_ephemeral(lower):
                ignored.append("Ephemeral personal remark with no durable claim in it.")
            else:
                ignored.append("No first-person claim, preference, commitment, or event was stated.")
        return list(candidates.values()), ignored

    def _sentences(self, text: str) -> list[str]:
        return [part.strip() for part in self.SENTENCE.split(text) if part.strip()]

    def _candidate_from_frame(
        self,
        frame: Frame,
        payload: dict[str, Any],
        sentence: str,
        record: dict[str, Any],
    ) -> Candidate | None:
        predicate = str(payload.get("predicate") or "").strip()
        object_ = str(payload.get("object") or "").strip()
        if not predicate or not object_ or len(object_) < 2:
            return None

        scope = str(payload.get("scope") or frame.scope or "general")
        lower = sentence.lower()
        hedged = any(marker in lower for marker in HEDGE_MARKERS)
        correction = any(marker in lower for marker in CORRECTION_MARKERS)

        importance = frame.importance
        confidence = frame.confidence
        if hedged:
            confidence -= 0.34
        if correction:
            confidence += 0.12
            importance += 0.05
        if any(marker in lower for marker in EMPHASIS_MARKERS):
            confidence += 0.06
        confidence = clamp(confidence)
        importance = clamp(importance)

        metadata: dict[str, Any] = {"frame": frame.name, "app": record.get("app")}
        if frame.memory_type == "task":
            due = self._extract_due_date(lower, record.get("created_at"))
            if due:
                metadata["due_hint"] = due

        return Candidate(
            memory_type=frame.memory_type,
            subject="user",
            predicate=predicate,
            object=object_,
            scope=scope,
            canonical_text=self._canonical(frame, predicate, object_, scope),
            evidence=snippet(sentence, object_),
            importance=round(importance, 3),
            confidence=round(confidence, 3),
            utility=round(utility_score(importance, confidence, frame.memory_type), 3),
            decay_rate=frame.decay_rate,
            status="tentative" if hedged and confidence < 0.62 else "active",
            reason=frame.reason,
            source="rule",
            metadata=metadata,
        )

    def _canonical(self, frame: Frame, predicate: str, object_: str, scope: str) -> str:
        """Render the claim as a sentence, from the predicate's own shape."""
        readable = object_[0].upper() + object_[1:] if object_ else object_
        if predicate == "is_a":
            return f"User is a {object_}."
        if predicate == "works_at":
            return f"User works at {readable}."
        if predicate == "based_in":
            return f"User is based in {readable}."
        if predicate == "needs_to_do":
            return f"User needs to {object_}."
        if predicate == "mentioned_event":
            return f"User mentioned a {object_}."
        if predicate == "uses_tool":
            project = scope.replace("project:", "").replace("_", " ")
            return f"For {project}, user is using {readable}."
        if predicate == "current_focus_is":
            return f"User's current focus is {object_}."
        if predicate == "prefers_assistant_behaviour":
            return f"User asked the assistant to {object_}."
        if predicate.startswith("prefers_"):
            return f"User prefers {object_}."
        if predicate.startswith("avoids_"):
            return f"User avoids {object_}."
        if predicate.startswith("has_"):
            return f"User has a {object_}."
        if predicate.endswith("_is"):
            attribute = predicate[:-3].replace("_", " ")
            return f"User's {attribute} is {readable}."
        return f"User: {predicate.replace('_', ' ')} {object_}."

    def _extract_due_date(self, lower: str, created_at: str | None) -> str | None:
        if not created_at:
            return None
        try:
            base = parse_dt(created_at)
        except Exception:
            return None
        if "tomorrow" in lower:
            return (base + timedelta(days=1)).date().isoformat()
        if "next week" in lower:
            return (base + timedelta(days=7)).date().isoformat()
        match = re.search(r"\bby (?:the )?(\d{1,2})(?:st|nd|rd|th)?\b", lower) or \
            re.search(r"\bbefore (?:the )?(\d{1,2})(?:st|nd|rd|th)?\b", lower)
        if match:
            day = int(match.group(1))
            if 1 <= day <= 31:
                return f"{base.year}-{base.month:02d}-{day:02d}"
        return None

    def _looks_ephemeral(self, lower: str) -> bool:
        """Structural, not topical: a short past-tense remark with no commitment."""
        if len(lower) > 240:
            return False
        past_tense = re.search(r"\b(was|were|had|went|got|did|felt|saw|ate|took|ran|\w+ed)\b", lower)
        forward_looking = re.search(
            r"\b(need|needs|must|should|will|going to|prefer|prefers|use|using|deadline|remind)\b", lower
        )
        return bool(past_tense) and not forward_looking

    def _explicitly_low_value(self, lower: str) -> bool:
        """An explicit instruction from the user, not a guess about the topic."""
        return any(
            phrase in lower
            for phrase in (
                "nothing important", "remember nothing", "nothing from this",
                "do not remember this", "don't remember this", "no need to remember",
                "random note", "ignore this", "forget this",
            )
        )


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


class HybridExtractor:
    """Union of both sensors, deduplicated on candidate identity.

    Rules win ties: where both fire on the same (type, predicate, scope,
    object), the deterministic candidate is kept so the reproducible path
    stays authoritative and the model only adds recall.
    """

    def __init__(self, rule: RuleExtractor, llm: LlmExtractor) -> None:
        self.rule = rule
        self.llm = llm

    def describe(self) -> dict[str, Any]:
        return {"backend": "hybrid(rule+llm)", **self.llm.describe()}

    def extract(self, record: dict[str, Any]) -> tuple[list[Candidate], list[str]]:
        rule_candidates, rule_ignored = self.rule.extract(record)
        llm_candidates, llm_ignored = self.llm.extract(record)
        merged: dict[str, Candidate] = {c.id: c for c in llm_candidates}
        merged.update({c.id: c for c in rule_candidates})
        ignored = rule_ignored + [item for item in llm_ignored if item not in rule_ignored]
        if merged:
            ignored = [item for item in ignored if "No durable" not in item]
        return list(merged.values()), ignored


# Backwards-compatible alias: the original class name.
SemanticExtractor = RuleExtractor


def get_extractor(config: KiviConfig, client: OllamaClient | None):
    mode = config.extractor
    if mode == "rule" or client is None:
        return RuleExtractor()
    if mode == "llm":
        return LlmExtractor(client, config)
    if mode == "hybrid":
        return HybridExtractor(RuleExtractor(), LlmExtractor(client, config))
    # auto
    if client.available() and client.has_model(config.llm_model):
        return LlmExtractor(client, config)
    return RuleExtractor()
