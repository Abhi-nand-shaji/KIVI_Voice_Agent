"""The deterministic sensor: grammatical frames over a transcript.

Runs the 25 frames in `frames.py` over each sentence and turns every match into
a `Candidate`. It knows nothing about any particular corpus - predicates are
synthesised from the sentence ("my manager is Neha" -> `manager_is`, "I prefer
long written feedback" -> `prefers_feedback`) - so the same code learns from a
designer's dictations and an engineer's without editing.

It is still a rule system and still the narrower of the two sensors. What it is
not any more is a recogniser for one specific test corpus.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

from .core import (
    Candidate,
    clamp,
    normalize_text,
    parse_dt,
    snippet,
    strip_disfluency,
    utility_score,
)
from .frames import (
    CORRECTION_MARKERS,
    EMPHASIS_MARKERS,
    FRAMES,
    HEDGE_MARKERS,
    Frame,
)


# reason the second sensor is not consulted. Only an explicit instruction is.
IGNORED_SUPPRESSED = "The user explicitly asked for this not to be remembered."
IGNORED_EPHEMERAL = "Ephemeral personal remark with no durable claim in it."
IGNORED_NO_CLAIM = "No first-person claim, preference, commitment, or event was stated."

SETTLED_REASONS = (IGNORED_SUPPRESSED,)


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
        text = strip_disfluency(normalize_text(
            record.get("formatted_text") or record.get("raw_asr") or ""))
        lower = text.lower()
        if self._explicitly_low_value(lower):
            return [], [IGNORED_SUPPRESSED]

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
                ignored.append(IGNORED_EPHEMERAL)
            else:
                ignored.append(IGNORED_NO_CLAIM)
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
        if predicate.startswith("instructed_not_to_"):
            return f"User asked the assistant not to {object_}."
        if predicate.startswith("instructed_always_"):
            return f"User asked the assistant to always {object_}."
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
        # Anything carrying a date, a weekday or a scheduling verb is about
        # future coordination, however it is phrased. "The migration has been
        # shifted to Thursday" is past tense but it is not small talk.
        scheduled = re.search(
            r"\b(monday|tuesday|wednesday|thursday|thurs|friday|saturday|sunday|today|tomorrow|"
            r"tonight|next week|this week|deadline|due|shifted|moved|postponed|rescheduled|"
            r"meeting|call|standup|review|\d{1,2}[/-]\d{1,2}|\d{1,2}\s*(?:am|pm))\b", lower
        )
        # A change of state is durable however far in the past it happened.
        # "We switched the prototype to Rust last week" is past tense with no
        # scheduling word, and the two tests above therefore both missed it -
        # yet the claim it makes is true until something switches back.
        state_change = re.search(
            r"\b(switch\w*|migrat\w*|adopt\w*|replac\w*|dropp\w*|abandon\w*|"
            r"rewrote|rewritten|ported|upgrad\w*|renamed|promoted|joined|left|"
            r"started using|stopped using|no longer)\b", lower
        )
        return bool(past_tense) and not forward_looking and not scheduled and not state_change

    # A retraction is grammar, not vocabulary. The literal-phrase list this
    # replaces missed "do not remember any of this" while catching "do not
    # remember this", which is the kind of gap a person finds immediately and a
    # test corpus never does. A negated retention verb with any of its ordinary
    # objects is the whole pattern.
    SUPPRESSION = re.compile(
        r"\b(?:do ?n[o']?t|dont|no need to|never)\s+"
        r"(?:remember|save|store|keep|note|record|log)\b"
        r"(?:\s+(?:any\s+)?(?:of\s+)?(?:this|that|these|it|any\s+of\s+it))?",
        re.IGNORECASE,
    )

    def _explicitly_low_value(self, lower: str) -> bool:
        """An explicit instruction from the user, not a guess about the topic."""
        if self.SUPPRESSION.search(lower):
            return True
        return any(
            phrase in lower
            for phrase in (
                "nothing important", "remember nothing", "nothing from this",
                "random note", "ignore this", "forget this", "off the record",
            )
        )
