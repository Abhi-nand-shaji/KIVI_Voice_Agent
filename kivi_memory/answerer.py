"""Answer generation and text polishing.

`TemplateAnswerer` renders a stored claim with a grammatical person-shift and
nothing else - no per-predicate table, so it degrades gracefully to predicates
nobody anticipated.

`LlmAnswerer` is a grounded generator under a hard constraint:
it may only use the memories it is handed, it must cite them by id, and it must
abstain when they do not contain the answer. Any answer citing an id that was
not supplied is discarded and the template path is used instead - the model is
not trusted to police itself.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import KiviConfig
from .core import normalize_text, to_second_person
from .llm import OllamaClient


class TemplateAnswerer:
    """Backend-free answer rendering.

    This used to be a table of one canned sentence per predicate - seven
    `if predicate == ...` branches, several of which hardcoded the seed
    corpus's own answers ("You prefer concise technical communication.").
    A memory with any other predicate fell through to a bulleted dump.

    Every candidate already carries `canonical_text`: a complete sentence
    written by whichever sensor produced it. So rendering is just a
    person-shift (`core.to_second_person`), which is grammar rather than
    domain knowledge and therefore works for predicates nobody anticipated.
    """

    name = "template"

    def compose(self, query: str, rows: list[Any], style: dict[str, Any] | None = None) -> dict[str, Any]:
        primary = rows[0]
        answer = to_second_person(str(primary["canonical_text"]).strip())
        if not answer.endswith((".", "!", "?")):
            answer += "."

        # Deliberately one memory, not a list. An earlier version appended an
        # "Also relevant" block, which read badly (three answers to one
        # question) and, worse, let a substring-graded evaluation pass on a
        # memory that was not the one the retriever actually chose. If the top
        # memory is wrong, that should be visible.
        return {"answer": answer, "abstain": False, "cited_ids": [primary["id"]], "backend": self.name}

    def polish(self, text: str, style_claims: list[str] | None = None) -> str:
        """Backend-free cleanup: disfluencies and spacing only.

        Deliberately does NOT try to apply a style preference. The previous
        version ran `SELECT ... WHERE object LIKE '%concise%'` and truncated to
        two sentences if it matched - a hardcoded reading of one specific
        preference string. Applying a stated style to prose is a generation
        task; when no model is available this returns clean text and says so
        rather than guessing at what the user meant by it.
        """
        clean = normalize_text(text)
        clean = re.sub(r"\b(um+|uh+|erm+|hmm+|you know|i mean|sort of|kind of)\b", "", clean,
                       flags=re.IGNORECASE)
        clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
        clean = re.sub(r"\s+", " ", clean).strip(" ,;:")
        if not clean:
            return text
        if not clean.endswith((".", "!", "?")):
            clean += "."
        return clean[0].upper() + clean[1:]

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "generated": False}


POLISH_SYSTEM = """You clean up a raw voice dictation so it can be sent as written text.

Hard rules:
- Preserve every fact, name, number and technical term. Do not add anything.
- Remove disfluencies and false starts; fix punctuation and capitalisation.
- Apply the user's stated style preferences if any are given.
- Do not summarise unless a style preference asks for brevity.

Reply with JSON only: {"polished": "..."}"""

POLISH_SCHEMA = {
    "type": "object",
    "properties": {"polished": {"type": "string"}},
    "required": ["polished"],
}


ANSWER_SYSTEM = """You answer questions about a user from a fixed set of retrieved memories.

Hard rules:
- Use ONLY the numbered memories provided. They are your entire world.
- If they do not contain the answer, abstain. Do not guess, infer beyond them,
  or fill gaps from general knowledge.
- Cite the id of every memory you used, exactly as given.
- A memory marked "tentative" is uncertain: say so in the answer if you use it.
- Match the user's stated response-style preference if one is provided.
- Answer in one or two sentences unless the question asks for a list.

Reply with JSON only:
{"answer": "...", "abstain": false, "used_memory_ids": ["mem_..."]}"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "abstain": {"type": "boolean"},
        "used_memory_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "abstain", "used_memory_ids"],
}


class LlmAnswerer:
    name = "llm"

    def __init__(self, client: OllamaClient, config: KiviConfig) -> None:
        self.client = client
        self.config = config
        self.template = TemplateAnswerer()
        self.degraded = False

    def describe(self) -> dict[str, Any]:
        return {
            "backend": f"ollama:{self.config.llm_model}",
            "generated": True,
            "degraded_to_template": self.degraded,
        }

    def compose(self, query: str, rows: list[Any], style: dict[str, Any] | None = None) -> dict[str, Any]:
        allowed = {row["id"] for row in rows}
        payload = self.client.chat_json(
            ANSWER_SYSTEM,
            self._prompt(query, rows, style),
            schema=ANSWER_SCHEMA,
            cache_kind="answer",
        )
        if not payload:
            self.degraded = True
            result = self.template.compose(query, rows, style)
            result["backend"] = "template (llm unavailable)"
            return result

        self.degraded = False
        answer = normalize_text(str(payload.get("answer") or ""))
        abstain = bool(payload.get("abstain"))
        cited = [str(item) for item in (payload.get("used_memory_ids") or [])]
        unknown = [item for item in cited if item not in allowed]

        if abstain:
            return {"answer": answer or "I do not have enough supported memory to answer that.",
                    "abstain": True, "cited_ids": [], "backend": self.name}

        if not answer or unknown or not cited:
            # Ungrounded or malformed. Fall back rather than surface it.
            result = self.template.compose(query, rows, style)
            result["backend"] = "template (ungrounded llm output rejected)"
            result["rejected_citations"] = unknown
            return result

        return {"answer": answer, "abstain": False, "cited_ids": cited, "backend": self.name}

    def polish(self, text: str, style_claims: list[str] | None = None) -> str:
        payload = self.client.chat_json(
            POLISH_SYSTEM,
            json.dumps({"dictation": text, "style_preferences": style_claims or []}, ensure_ascii=False),
            schema=POLISH_SCHEMA,
            cache_kind="polish",
        )
        polished = normalize_text(str((payload or {}).get("polished") or ""))
        if not polished:
            return self.template.polish(text, style_claims)

        # Guard against the model quietly dropping content. Polishing may lose
        # filler words; it may not lose half the substance.
        source_terms = set(re.findall(r"[a-z0-9]{4,}", text.lower()))
        kept = source_terms & set(re.findall(r"[a-z0-9]{4,}", polished.lower()))
        asked_for_brevity = any("concise" in claim.lower() or "brief" in claim.lower() or "short" in claim.lower()
                                for claim in (style_claims or []))
        floor = 0.45 if asked_for_brevity else 0.7
        if source_terms and len(kept) / len(source_terms) < floor:
            return self.template.polish(text, style_claims)
        return polished

    def _prompt(self, query: str, rows: list[Any], style: dict[str, Any] | None) -> str:
        memories = [
            {
                "id": row["id"],
                "claim": row["canonical_text"],
                "type": row["memory_type"],
                "predicate": row["predicate"],
                "object": row["object"],
                "scope": row["scope"],
                "status": row["status"],
                "confidence": round(float(row["confidence"]), 2),
                "last_seen_at": row["last_seen_at"],
            }
            for row in rows
        ]
        return json.dumps(
            {
                "question": query,
                "response_style_preference": (style or {}).get("style"),
                "memories": memories,
            },
            ensure_ascii=False,
            indent=2,
        )


def get_answerer(config: KiviConfig, client: OllamaClient | None):
    mode = config.answerer
    if mode == "template" or client is None:
        return TemplateAnswerer()
    if mode == "llm":
        return LlmAnswerer(client, config)
    if client.available() and client.has_model(config.llm_model):
        return LlmAnswerer(client, config)
    return TemplateAnswerer()
