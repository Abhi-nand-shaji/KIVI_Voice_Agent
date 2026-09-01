"""Domain-free extraction frames.

The deterministic sensor used to match this corpus by name - "golden goose",
"sarvam", "predictive maintenance", `\\bi work at (...)`. On a different user's
dictations it learned 8 memories out of 500 and abstained on almost every
question, which is exactly the condition the reviewer's own corpus creates.

What is here instead are *grammatical* frames. They key on how English
first-person statements are shaped ("my X is Y", "I prefer X", "for the X I am
using Y", "I need to X"), never on what X and Y are about. Predicates are
synthesised from the sentence rather than chosen from a fixed list, so a
corpus about design systems produces `manager_is` and `prefers_feedback` the
same way a corpus about compilers produces `manager_is` and `prefers_answers`.

Every frame declares the memory type, how to build the predicate and object,
and a base importance/confidence for that *shape* of statement - not for that
topic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


ARTICLES = ("the ", "a ", "an ", "my ", "our ", "some ", "any ")

# Clause boundaries: an object ends where the sentence changes direction.
CLAUSE_BREAK = re.compile(
    r"\s+(?:and|but|because|so that|so|while|although|though|since|which|that\s+is|"
    r"when|before|after|unless|however)\s+",
    re.IGNORECASE,
)

TRAILING_NOISE = re.compile(
    r"\s+(?:for now|right now|these days|at the moment|today|currently|please|thanks)\b\.?$",
    re.IGNORECASE,
)

# Words that never make a useful predicate topic.
WEAK_HEADS = {
    "thing", "things", "stuff", "one", "ones", "it", "them", "this", "that",
    "work", "lot", "bit", "kind", "sort", "way", "time", "times",
}

HEDGE_MARKERS = ("maybe", "might", "thinking about", "considering", "probably", "perhaps", "not sure")
CORRECTION_MARKERS = (
    "actually", "no longer", "instead", "switched", "now my", "now using",
    "moved off", "we moved", "changed to", "from now on", "correction",
)
EMPHASIS_MARKERS = ("must", "always", "never", "definitely", "make sure")


def clean_object(value: str, max_words: int = 12) -> str:
    """Trim a captured phrase down to the thing itself."""
    text = value.strip().strip("\"'")
    text = CLAUSE_BREAK.split(text)[0]
    text = TRAILING_NOISE.sub("", text)
    text = re.sub(r"[.,;:!?]+$", "", text).strip()
    lowered = text.lower()
    for article in ARTICLES:
        if lowered.startswith(article):
            text = text[len(article):]
            lowered = text.lower()
            break
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return re.sub(r"\s+", " ", text).strip().lower()


def slug(value: str, max_words: int = 4) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in {"the", "a", "an", "my", "of", "for"}]
    return "_".join(words[:max_words]) or "general"


def head_noun(value: str) -> str:
    """Last content word of a phrase, singularised - the topic key for a predicate.

    Stemming matters here more than it looks. The head noun becomes part of the
    predicate name, and the predicate name is the memory's identity. Without it,
    "concise explanation" and "concise technical explanations" produce
    `prefers_explanation` and `prefers_explanations` - two separate memories for
    one preference, neither of which supersedes the other when the user changes
    their mind. Same word, same relation, same memory.
    """
    from .core import stem  # noqa: PLC0415  (core does not import frames)

    words = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if len(w) > 2]
    for word in reversed(words):
        if word not in WEAK_HEADS:
            return stem(word)
    return stem(words[-1]) if words else "general"


# Words that are never the name of a project or piece of work. Prevents
# "I am using Rust for now" from creating a project called "now".
NON_PROJECT = {
    "now", "today", "tomorrow", "this", "that", "it", "me", "you", "us", "them",
    "here", "there", "everything", "anything", "something", "work", "general",
    "the moment", "a while", "once", "week", "month", "year", "quarter",
}


@dataclass
class Frame:
    name: str
    pattern: str
    memory_type: str
    scope: str
    importance: float
    confidence: float
    decay_rate: float
    build: Callable[[re.Match], dict[str, Any] | None]
    reason: str
    # Frames in the same family compete: the first (most specific) one to match
    # a sentence wins, so "I am a designer at Zerodha" does not also produce the
    # vaguer "I am a designer at Zerodha" from the general identity frame.
    family: str = ""
    flags: int = re.IGNORECASE

    def compiled(self):
        if not hasattr(self, "_compiled"):
            self._compiled = re.compile(self.pattern, self.flags)
        return self._compiled


def _identity_at(match: re.Match) -> dict[str, Any] | None:
    role = clean_object(match.group(1))
    org = clean_object(match.group(2))
    if not role or not org:
        return None
    return {"predicate": "is_a", "object": role, "extra": [{"predicate": "works_at", "object": org}]}


def _simple(predicate: str, group: int = 1):
    def build(match: re.Match) -> dict[str, Any] | None:
        value = clean_object(match.group(group))
        return {"predicate": predicate, "object": value} if value else None
    return build


def _possessive(match: re.Match) -> dict[str, Any] | None:
    attribute = slug(match.group(1))
    value = clean_object(match.group(2))
    if not attribute or not value or attribute == "general":
        return None
    return {"predicate": f"{attribute}_is", "object": value}


def _has(match: re.Match) -> dict[str, Any] | None:
    value = clean_object(match.group(1))
    if not value:
        return None
    return {"predicate": f"has_{head_noun(value)}", "object": value}


INFINITIVE = re.compile(r"^to\s+\w+")


def _preference(prefix: str):
    def build(match: re.Match) -> dict[str, Any] | None:
        value = clean_object(match.group(1))
        # "I want to test a stricter controller" states an intention, not a
        # standing preference. Infinitive objects are purpose clauses.
        if not value or INFINITIVE.match(value):
            return None
        return {"predicate": f"{prefix}_{head_noun(value)}", "object": value}
    return build


def _quality(match: re.Match) -> dict[str, Any] | None:
    subject = clean_object(match.group(1))
    quality = match.group(2).lower()
    if not subject:
        return None
    return {"predicate": f"prefers_{head_noun(subject)}", "object": f"{quality} {subject}"}


def _valid_project(project: str) -> bool:
    return bool(project) and project.lower() not in NON_PROJECT and len(project) > 2


def _trim_tool(value: str) -> str:
    """A tool name ends where the sentence starts comparing it to something."""
    return re.split(r"\s+(?:with|instead of|rather than|over|and not)\s+", value, maxsplit=1)[0].strip()


def _project_tool(project_group: int, tool_group: int):
    def build(match: re.Match) -> dict[str, Any] | None:
        project = clean_object(match.group(project_group))
        tool = _trim_tool(clean_object(match.group(tool_group), max_words=8))
        if not _valid_project(project) or not tool:
            return None
        return {"predicate": "uses_tool", "object": tool, "scope": f"project:{slug(project)}"}
    return build


def _project_switch(match: re.Match) -> dict[str, Any] | None:
    project = clean_object(match.group(1))
    tool = _trim_tool(clean_object(match.group(match.lastindex), max_words=8))
    if not _valid_project(project) or not tool:
        return None
    return {"predicate": "uses_tool", "object": tool, "scope": f"project:{slug(project)}"}


def _focus(match: re.Match) -> dict[str, Any] | None:
    value = clean_object(match.group(1))
    return {"predicate": "current_focus_is", "object": value} if value else None


def _task(match: re.Match) -> dict[str, Any] | None:
    value = clean_object(match.group(1), max_words=18)
    return {"predicate": "needs_to_do", "object": value} if value else None


def _event(match: re.Match) -> dict[str, Any] | None:
    kind = match.group(1).lower()
    who = clean_object(match.group(2), max_words=8)
    if not who:
        return None
    return {"predicate": "mentioned_event", "object": f"{kind} with {who}"}


def _instruction(prefix: str):
    """A standing instruction to the assistant.

    Deliberately NOT named `prefers_*`. A single catch-all preference predicate
    matched every question containing the word "prefer", so "what response style
    do I prefer?" was answered with "you asked the assistant to avoid make me
    approve every memory". Naming the relation after what it is about keeps it
    out of unrelated queries, and reads as a sentence.
    """
    def build(match: re.Match) -> dict[str, Any] | None:
        value = clean_object(match.group(1), max_words=14)
        if not value or len(value) < 4:
            return None
        return {
            "predicate": f"instructed_{'not_to' if prefix == 'avoid' else 'always'}_{head_noun(value)}",
            "object": value,
        }
    return build


# ---------------------------------------------------------------------------
# The frame table. Ordered: more specific shapes first.
# ---------------------------------------------------------------------------

FRAMES: list[Frame] = [
    Frame("identity_at", r"\bi(?:'m| am) (?:a|an) ([^.,;]{3,60}?) at ([^.,;]{2,50})",
          "fact", "profile", 0.86, 0.88, 0.005, _identity_at, "Stated role and employer.", family="identity"),
    Frame("identity", r"\bi(?:'m| am) (?:a|an) ([^.,;]{3,60})",
          "fact", "profile", 0.82, 0.84, 0.005, _simple("is_a"), "Stated role or identity.", family="identity"),
    Frame("works_at", r"\bi work (?:at|for) (?:the )?([^.,;]{2,50})",
          "fact", "profile", 0.84, 0.86, 0.005, _simple("works_at"), "Stated employer."),
    Frame("works_out_of", r"\bi work (?:out of|from) (?:the )?([^.,;]{2,40}?)(?:\s+office)?(?:[.,;]|$)",
          "fact", "profile", 0.8, 0.84, 0.005, _simple("based_in"), "Stated work location.", family="location"),
    Frame("based_in", r"\bi (?:live|am based|am located|am currently) in (?:the )?([^.,;]{2,40})",
          "fact", "profile", 0.82, 0.86, 0.005, _simple("based_in"), "Stated location.", family="location"),
    Frame("moved_to", r"\bi(?:'ve| have)? ?(?:just )?moved to ([^.,;]{2,40})",
          "fact", "profile", 0.84, 0.88, 0.005, _simple("based_in"), "Stated relocation.", family="location"),
    Frame("possessive", r"\bmy ([a-z][a-z ]{2,28}?) (?:is|are) ([^.,;]{2,60})",
          "fact", "profile", 0.8, 0.84, 0.006, _possessive, "Stated attribute of the user."),
    Frame("has", r"\bi have (?:a|an) ([^.,;]{3,60})",
          "fact", "profile", 0.76, 0.82, 0.02, _has, "Something the user says they have."),

    Frame("prefers", r"\bi (?:prefer|like|love|want|always want) ([^.,;]{3,70})",
          "preference", "communication", 0.86, 0.84, 0.012, _preference("prefers"),
          "Explicitly stated preference."),
    Frame("avoids", r"\bi (?:hate|dislike|do not like|don't like|avoid|do not want|don't want) ([^.,;]{3,70})",
          "preference", "communication", 0.82, 0.82, 0.012, _preference("avoids"),
          "Explicitly stated dispreference."),
    Frame("quality", r"\b(?:keep|make|write) (?:the |my |your )?([a-z][a-z ]{2,28}?) (short|brief|concise|detailed|long|simple|formal|casual|thorough)\b",
          "preference", "communication", 0.84, 0.82, 0.012, _quality, "Stated quality preference."),
    Frame("use_when", r"\b(?:use|give me|send) ([^.,;]{3,50}?) when (?:you|i|we) [^.,;]{3,60}",
          "preference", "communication", 0.8, 0.8, 0.014, _preference("prefers"),
          "Conditional formatting preference."),
    Frame("never", r"\b(?:do not|don't|never) ([^.,;]{4,60})",
          "preference", "product", 0.78, 0.78, 0.014, _instruction("avoid"),
          "Instruction to the assistant."),
    Frame("always", r"\balways ([^.,;]{4,60})",
          "preference", "product", 0.78, 0.8, 0.014, _instruction("always"),
          "Standing instruction to the assistant."),

    Frame("project_tool_for", r"\bfor (?:the )?([^.,;]{3,50}?)[, ]+i(?:'m| am)? (?:now )?(?:using|prototyping in|building in|writing in|working in) ([^.,;]{2,50})",
          "project_state", "", 0.84, 0.84, 0.05, _project_tool(1, 2), "Tool choice on a named piece of work.", family="project_tool"),
    Frame("tool_for_project", r"\bi(?:'m| am)? (?:using|use) ([^.,;]{2,40}?) for (?:the |my )?([^.,;]{3,50})",
          "project_state", "", 0.82, 0.82, 0.05, _project_tool(2, 1), "Tool choice on a named piece of work.", family="project_tool"),
    Frame("moved_project_tool", r"\bwe (?:moved|switched|took) (?:the )?([^.,;]{3,50}?) (?:off|from) [^.,;]{2,40}? to ([^.,;]{2,50})",
          "project_state", "", 0.88, 0.9, 0.05, _project_switch, "Explicit change of tool on a named piece of work.", family="project_tool"),
    Frame("switched_project", r"\bi (?:switched|moved) (?:the )?([^.,;]{3,50}?) to ([^.,;]{2,50})",
          "project_state", "", 0.88, 0.9, 0.05, _project_switch, "Explicit change on a named piece of work.", family="project_tool"),
    Frame("focus", r"\b(?:the )?([^.,;]{3,50}?) is (?:the main thing on my plate|my main focus|my priority|the priority)",
          "project_state", "current_focus", 0.84, 0.86, 0.06, _focus, "Stated current focus."),

    Frame("need_to", r"\bi (?:need to|have to|must|should) ([^.,;]{4,90})",
          "task", "current_tasks", 0.74, 0.78, 0.14, _task, "Action item stated by the user.", family="task"),
    Frame("remind", r"\bremind me (?:that |to |about )?([^.,;]{4,90})",
          "task", "current_tasks", 0.78, 0.82, 0.12, _task, "Explicit reminder request.", family="task"),
    Frame("follow_up", r"\b(?:follow up|check in|circle back) with ([^.,;]{2,60})",
          "task", "current_tasks", 0.7, 0.76, 0.16, _task, "Follow-up commitment.", family="task"),
    Frame("ship_by", r"\b(?:ship|send|submit|finish|deliver|complete) ([^.,;]{4,70}?) (?:before|by) [^.,;]{2,30}",
          "task", "current_tasks", 0.78, 0.8, 0.13, _task, "Deadline-bearing commitment.", family="task"),

    Frame("event_with", r"\b(meeting|call|demo|sync|review|presentation|interview|standup) with ([^.,;]{2,50})",
          "event", "recent_events", 0.6, 0.72, 0.18, _event, "Scheduled or mentioned event."),
]
