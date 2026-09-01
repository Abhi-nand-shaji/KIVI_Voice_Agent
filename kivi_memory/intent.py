"""Query understanding.

The first version routed every question through a chain of substring tests:

    if "deliberately ignore" in q or "ignored" in q: ...
    elif "find" in q and ("dictation" in q or "slack" in q): ...

and then pulled slots out with more of the same - `"slack" if "slack" in q
else "email" if "email" in q else None`, and a date parser that understood
"yesterday" and "today" and nothing else. Ask "show me what you threw away"
and it silently answered the wrong question.

This replaces that with two things that generalise:

* `LlmIntentRouter` - the local model classifies the query into one of the
  intents below and fills a small slot schema (app, date, hour, target).
* `EmbeddingIntentRouter` - the offline fallback. Each intent carries a few
  natural-language exemplars; the query is matched against them by embedding
  similarity plus lexical overlap. That is a description of each intent rather
  than a keyword table, so paraphrases the author never wrote still land.

Slot filling is data-driven where it can be: the set of apps comes from
`SELECT DISTINCT app FROM transcripts`, not from a literal list, so a corpus
with Notion and Linear in it is searchable by name the day it is imported.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .config import KiviConfig
from .core import cosine, normalize_text, tokens
from .llm import OllamaClient


INTENTS = ("memory_question", "memory_overview", "ignored_audit", "find_transcript", "correction")

EXEMPLARS: dict[str, list[str]] = {
    "memory_question": [
        "what response style do I prefer",
        "which language am I using for this project",
        "who is my manager",
        "what am I working on",
        "where am I based",
    ],
    "memory_overview": [
        "what do you remember about me",
        "show me everything you have learned",
        "list your memories",
        "what is stored so far",
    ],
    "ignored_audit": [
        "what did you deliberately ignore",
        "show me what you threw away",
        "which things did you decide not to remember",
        "what got rejected and why",
        "what did you skip",
    ],
    "find_transcript": [
        "find the dictation I did yesterday afternoon and polish it",
        "pull up the note I recorded this morning",
        "get the message I dictated around 5 pm and clean it up",
        "search my transcripts for the one about the meeting",
    ],
    "correction": [
        "forget that",
        "that is wrong, drop it",
        "stop remembering that I prefer bullets",
        "that is no longer true",
        "remove that memory",
    ],
}

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# Coarse parts of the day, in hours. Not app- or corpus-specific.
DAYPARTS = {"morning": 9.0, "noon": 12.0, "afternoon": 15.0, "evening": 19.0, "night": 21.0}


@dataclass
class Intent:
    name: str
    confidence: float
    backend: str
    slots: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"intent": self.name, "confidence": round(self.confidence, 3),
                "backend": self.backend, "slots": self.slots}


ROUTER_SYSTEM = """You classify a user's request to a personal memory assistant, and extract slots.

Intents:
- memory_question   asking about a specific thing the assistant may have learned
- memory_overview   asking for everything the assistant has stored
- ignored_audit     asking what the assistant chose NOT to remember, and why
- find_transcript   asking to locate a past dictation/note/message, possibly to clean it up
- correction        telling the assistant something is wrong, or to forget it

Slots (all optional, omit when the query does not say):
- app          the application named in the query, lowercase, exactly as the user said it
- date         absolute date as YYYY-MM-DD if stated
- relative_days  integer offset from today if the query says today/yesterday/two days ago (0, -1, -2)
- weekday      a weekday name if the query names one
- hour         hour of day as a 0-23 number if a time is given
- daypart      morning | noon | afternoon | evening | night if the query says one
- target       what the user wants forgotten or asked about, as a short phrase

Reply with JSON only:
{"intent": "...", "confidence": 0.0-1.0, "slots": {...}}"""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "confidence": {"type": "number"},
        "slots": {"type": "object"},
    },
    "required": ["intent", "confidence"],
}


class BaseIntentRouter:
    name = "base"

    def route(self, query: str, known_apps: list[str]) -> Intent:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name}

    # -- slot normalisation, shared by both backends ------------------------

    @staticmethod
    def resolve_app(query: str, known_apps: list[str], hinted: str | None = None) -> str | None:
        """Match against the apps actually present in the corpus."""
        lower = query.lower()
        candidates = [app for app in known_apps if app]
        if hinted:
            hint = hinted.strip().lower()
            for app in candidates:
                if hint == app or hint in app or app in hint:
                    return app
        # Several app names are also ordinary words ("dictation", "docs",
        # "calendar"), so presence alone is not enough: "find the dictation I
        # did in Slack" mentions two. Prefer the one the user pointed *at* with
        # a preposition, then the last one mentioned.
        matches: list[tuple[int, bool, str]] = []
        for app in candidates:
            for match in re.finditer(rf"\b{re.escape(app)}\b", lower):
                prefix = lower[max(0, match.start() - 14):match.start()]
                prepositional = bool(re.search(r"\b(in|on|from|via|inside|within|using)\s*$", prefix))
                matches.append((match.start(), prepositional, app))
        if not matches:
            return None
        matches.sort(key=lambda item: (item[1], item[0]))
        return matches[-1][2]

    @staticmethod
    def resolve_datetime(query: str, slots: dict[str, Any], base: datetime) -> tuple[datetime | None, float | None]:
        """Turn whatever the router found into (target_date, target_hour).

        `base` is the most recent transcript in the corpus, so "yesterday"
        means yesterday relative to the data rather than to the wall clock.
        """
        lower = query.lower()
        target_date: datetime | None = None

        if slots.get("date"):
            try:
                target_date = datetime.fromisoformat(str(slots["date"])[:10])
            except ValueError:
                target_date = None
        if target_date is None and slots.get("relative_days") is not None:
            try:
                target_date = base + timedelta(days=int(slots["relative_days"]))
            except (TypeError, ValueError):
                target_date = None
        if target_date is None:
            match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", lower)
            if match:
                target_date = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if target_date is None:
            match = re.search(r"\b(\d+)\s+days?\s+ago\b", lower)
            if match:
                target_date = base - timedelta(days=int(match.group(1)))
            elif "day before yesterday" in lower:
                target_date = base - timedelta(days=2)
            elif "yesterday" in lower or "last night" in lower:
                target_date = base - timedelta(days=1)
            elif "today" in lower or "this morning" in lower or "this afternoon" in lower or "this evening" in lower:
                target_date = base
        if target_date is None:
            weekday = str(slots.get("weekday") or "").strip().lower()
            if weekday not in WEEKDAYS:
                weekday = next((day for day in WEEKDAYS if day in lower), "")
            if weekday:
                delta = (base.weekday() - WEEKDAYS.index(weekday)) % 7
                target_date = base - timedelta(days=delta or 7 if "last" in lower else delta)

        target_hour: float | None = None
        if slots.get("hour") is not None:
            try:
                target_hour = float(slots["hour"])
            except (TypeError, ValueError):
                target_hour = None
        if target_hour is None:
            match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lower) or re.search(
                r"\baround\s+(\d{1,2})(?::(\d{2}))?\b", lower
            )
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2) or 0)
                meridiem = match.group(3) if match.lastindex and match.lastindex >= 3 else None
                if meridiem == "pm" and hour < 12:
                    hour += 12
                if meridiem == "am" and hour == 12:
                    hour = 0
                target_hour = hour + minute / 60
        if target_hour is None:
            daypart = str(slots.get("daypart") or "").strip().lower()
            if daypart not in DAYPARTS:
                daypart = next((part for part in DAYPARTS if part in lower), "")
            if daypart:
                target_hour = DAYPARTS[daypart]

        return target_date, target_hour


class EmbeddingIntentRouter(BaseIntentRouter):
    """Offline router: nearest intent by exemplar similarity + lexical overlap."""

    name = "exemplar"

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self._vectors = {
            intent: [embedder.embed_one(text) for text in examples]
            for intent, examples in EXEMPLARS.items()
        }
        self._tokens = {
            intent: [tokens(text) for text in examples]
            for intent, examples in EXEMPLARS.items()
        }

    def route(self, query: str, known_apps: list[str]) -> Intent:
        query_vector = self.embedder.embed_one(query)
        query_tokens = tokens(query)
        scores: dict[str, float] = {}
        for intent in INTENTS:
            best = 0.0
            for vector, example_tokens in zip(self._vectors[intent], self._tokens[intent]):
                similarity = max(0.0, cosine(query_vector, vector))
                if example_tokens:
                    overlap = len(query_tokens & example_tokens) / len(example_tokens)
                else:
                    overlap = 0.0
                best = max(best, 0.6 * similarity + 0.4 * overlap)
            scores[intent] = best

        name = max(scores, key=lambda key: scores[key])
        top = scores[name]
        runner_up = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
        # A weak or ambiguous match means "just answer the question".
        if top < 0.18 or (top - runner_up) < 0.03:
            name, top = "memory_question", max(top, 0.2)
        return Intent(name, top, self.name, {"app": self.resolve_app(query, known_apps)})

    def describe(self) -> dict[str, Any]:
        return {"backend": f"exemplar({self.embedder.describe().get('backend')})"}


class LlmIntentRouter(BaseIntentRouter):
    name = "llm"

    def __init__(self, client: OllamaClient, config: KiviConfig, fallback: BaseIntentRouter) -> None:
        self.client = client
        self.config = config
        self.fallback = fallback
        self.degraded = False

    def route(self, query: str, known_apps: list[str]) -> Intent:
        payload = self.client.chat_json(
            ROUTER_SYSTEM,
            json.dumps({"query": normalize_text(query), "known_apps": known_apps}, ensure_ascii=False),
            schema=ROUTER_SCHEMA,
            cache_kind="intent",
        )
        if not payload or payload.get("intent") not in INTENTS:
            self.degraded = True
            return self.fallback.route(query, known_apps)
        self.degraded = False
        slots = payload.get("slots") if isinstance(payload.get("slots"), dict) else {}
        slots = dict(slots or {})
        slots["app"] = self.resolve_app(query, known_apps, slots.get("app"))
        try:
            confidence = float(payload.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        return Intent(str(payload["intent"]), max(0.0, min(1.0, confidence)),
                      f"{self.name}:{self.config.llm_model}", slots)

    def describe(self) -> dict[str, Any]:
        return {"backend": f"ollama:{self.config.llm_model}", "degraded_to_exemplar": self.degraded}


def get_router(config: KiviConfig, client: OllamaClient | None, embedder) -> BaseIntentRouter:
    fallback = EmbeddingIntentRouter(embedder)
    if client is None or config.answerer == "template" and config.extractor == "rule":
        return fallback
    if config.extractor == "llm" or (
        config.extractor == "auto" and client.available() and client.has_model(config.llm_model)
    ):
        return LlmIntentRouter(client, config, fallback)
    return fallback
