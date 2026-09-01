"""A stand-in for `ollama serve`, for testing the model-backed paths.

Why this exists: the LLM extractor, the NLI layer, the embedder and the
grounded answerer all talk to the same local HTTP API. Wiring bugs in that
layer (bad JSON handling, wrong endpoint, ungrounded citations slipping
through, a dead daemon being read as "nothing to remember") are exactly the
bugs that are expensive to find by hand with a 5 GB model in the loop.

This server implements /api/tags, /api/chat and /api/embed with small
deterministic heuristics. It is NOT a language model and is not used by the
application - it is only here so `tests/run_llm_path.py` can exercise every
model-backed branch in about a second, on any machine, with nothing installed.

Run standalone:  python3 -m tests.fake_ollama --port 11500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


MODELS = ["qwen2.5:7b-instruct", "nomic-embed-text"]

SINGLE_VALUED = {
    "language": {"python", "rust", "typescript", "javascript", "go"},
    "city": {"bangalore", "chennai", "mumbai", "delhi", "pune", "hyderabad"},
}


# --------------------------------------------------------------------------
# "extraction"
# --------------------------------------------------------------------------

def fake_extract(record: dict[str, Any]) -> dict[str, Any]:
    text = str(record.get("formatted_text") or "")
    lower = text.lower()
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def add(type_, predicate, object_, scope, claim, importance, confidence, **kwargs):
        candidates.append({
            "type": type_, "predicate": predicate, "object": object_, "scope": scope,
            "claim": claim, "evidence": text, "importance": importance,
            "confidence": confidence, "hedged": kwargs.get("hedged", False),
            "correction": kwargs.get("correction", False),
        })

    correction = any(word in lower for word in ("actually", "switched", "no longer", "now my"))

    if re.search(r"\b(concise|short|brief)\b", lower) and re.search(
        r"\b(answer|answers|response|explanation|explanations|update|updates)\b", lower
    ):
        add("preference", "prefers_response_style", "concise technical communication", "communication",
            "User prefers concise technical communication.", 0.9, 0.86)

    if "bullet" in lower and re.search(r"\b(summar|notes|review)\w*", lower):
        add("preference", "prefers_response_format", "bulleted summaries", "communication",
            "User prefers bulleted summaries when reviewing notes.", 0.78, 0.78)

    if "audit trail" in lower or "approve every memory" in lower:
        add("preference", "prefers_memory_control", "low-friction control with auditability", "product",
            "User prefers low-friction memory control with inspectable audit trails.", 0.84, 0.77)

    language = re.search(r"\bprefer\w*\s+(python|rust|typescript|javascript|go)\b", lower)
    if language:
        value = language.group(1)
        add("preference", "prefers_language", value, "programming",
            f"User prefers {value.title()} for programming work.", 0.78, 0.79, correction=correction)

    if "golden goose" in lower or "kivi memory" in lower or "sarvam" in lower:
        project = re.search(r"\b(python|rust)\b", lower)
        if project and re.search(r"\b(using|switch\w*|build\w*|prototype|implement\w*)\b", lower):
            value = project.group(1)
            claim = (
                f"The Golden Goose / Kivi memory prototype is being built in {value.title()}."
            )
            add("project_state", "uses_language", value, "project:golden_goose", claim,
                0.85, 0.9 if correction else 0.83, correction=correction)
        if "semantic memory" in lower:
            add("project_state", "focuses_on", "selective semantic memory", "project:golden_goose",
                "The Golden Goose project focuses on selective semantic memory for Kivi.", 0.82, 0.84)
        if "internship task" in lower or "golden goose task" in lower or "sarvam ai" in lower:
            add("fact", "has_assignment", "sarvam ai golden goose internship task", "project:golden_goose",
                "User is working on the Sarvam AI Golden Goose internship task.", 0.8, 0.86)

    if "predictive maintenance" in lower:
        add("project_state", "topic", "predictive maintenance", "project:predictive_maintenance",
            "A current work thread is about predictive maintenance.", 0.7, 0.72)

    if re.search(r"\b(need to|have to|deadline|submit|finish|follow up|remind me)\b", lower):
        obj = text[:140].strip().lower()
        add("task", "needs_to_do", obj, "current_tasks", f"User needs to do: {text[:140].strip()}", 0.7, 0.72)

    if re.search(r"\b(meeting|call|demo|presentation)\b", lower):
        obj = text[:130].strip().lower()
        add("event", "mentioned_event", obj, "recent_events", f"User mentioned an event: {text[:130].strip()}",
            0.58, 0.68)

    if not candidates:
        if any(word in lower for word in ("chai", "coffee", "weather", "traffic", "playlist", "walked", "chair", "lunch", "dinner")):
            rejected.append({"text": text, "reason": "Ephemeral personal detail with low future utility."})
        else:
            rejected.append({"text": text, "reason": "No durable fact, preference, task, project state, or useful episode detected."})

    return {"candidates": candidates, "rejected": rejected}


# --------------------------------------------------------------------------
# "NLI"
# --------------------------------------------------------------------------

def fake_nli(premise: str, hypothesis: str) -> dict[str, Any]:
    p, h = premise.lower(), hypothesis.lower()
    for _, values in SINGLE_VALUED.items():
        in_p = {value for value in values if re.search(rf"\b{value}\b", p)}
        in_h = {value for value in values if re.search(rf"\b{value}\b", h)}
        if in_p and in_h and not (in_p & in_h):
            return {"label": "contradiction", "probability": 0.91,
                    "rationale": f"single-valued attribute: {sorted(in_p)} vs {sorted(in_h)}"}

    # A real NLI model reads word order and syntax. This stub only compares
    # content words, which is enough to stand in for "same claim, reordered".
    filler = {"the", "that", "are", "and", "for", "with", "was", "its", "his", "her", "you", "any"}
    p_tokens = set(re.findall(r"[a-z]{3,}", p)) - filler
    h_tokens = set(re.findall(r"[a-z]{3,}", h)) - filler
    if not p_tokens or not h_tokens:
        return {"label": "neutral", "probability": 0.5, "rationale": "empty"}
    coverage = len(p_tokens & h_tokens) / max(1, min(len(p_tokens), len(h_tokens)))
    jaccard = len(p_tokens & h_tokens) / len(p_tokens | h_tokens)
    if coverage >= 0.95 and jaccard >= 0.6:
        return {"label": "entailment", "probability": 0.88,
                "rationale": f"content words match (coverage={coverage:.2f})"}
    return {"label": "neutral", "probability": 0.6, "rationale": f"jaccard={jaccard:.2f}"}


# --------------------------------------------------------------------------
# "answering"
# --------------------------------------------------------------------------

def fake_answer(payload: dict[str, Any]) -> dict[str, Any]:
    memories = payload.get("memories") or []
    if not memories:
        return {"answer": "I do not have enough supported memory to answer that.",
                "abstain": True, "used_memory_ids": []}
    top = memories[0]
    claim = str(top.get("claim") or "").rstrip(".")
    prefix = "Based on what I have stored: " if len(memories) > 1 else ""
    answer = f"{prefix}{claim}."
    if top.get("status") == "tentative":
        answer += " I am not fully certain about this one."
    return {"answer": answer, "abstain": False, "used_memory_ids": [top["id"]]}


# --------------------------------------------------------------------------
# "intent routing"
# --------------------------------------------------------------------------

def fake_route(payload: dict[str, Any]) -> dict[str, Any]:
    query = str(payload.get("query") or "")
    lower = query.lower()
    slots: dict[str, Any] = {}

    for app in payload.get("known_apps") or []:
        if re.search(rf"\b(?:in|on|from|via)\s+{re.escape(str(app))}\b", lower):
            slots["app"] = app
            break

    if "yesterday" in lower:
        slots["relative_days"] = -1
    elif "today" in lower or "this morning" in lower:
        slots["relative_days"] = 0
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        if day in lower:
            slots["weekday"] = day
    for part in ("morning", "noon", "afternoon", "evening", "night"):
        if part in lower:
            slots["daypart"] = part
    match = re.search(r"\b(\d{1,2})\s*(am|pm)\b", lower)
    if match:
        hour = int(match.group(1)) % 12
        slots["hour"] = hour + (12 if match.group(2) == "pm" else 0)

    if re.search(r"\b(forget|wrong|incorrect|stop remembering|no longer true|drop)\b", lower):
        intent = "correction"
        target = re.sub(r"\b(hey kivi|forget|that|this|please|drop|stop remembering)\b", " ", lower)
        slots["target"] = re.sub(r"\s+", " ", target).strip(" .,")
    elif re.search(r"\b(ignore|ignored|threw away|throw away|not to keep|not to remember|rejected|skip)\b", lower):
        intent = "ignored_audit"
    elif re.search(r"\b(find|pull up|search|locate|get me)\b", lower) and re.search(
        r"\b(dictation|transcript|note|message|recording|draft)\b", lower
    ):
        intent = "find_transcript"
    elif re.search(r"\b(remember|learned|stored|know about me)\b", lower) and re.search(
        r"\b(what|show|list|tell|everything)\b", lower
    ):
        intent = "memory_overview"
    else:
        intent = "memory_question"

    return {"intent": intent, "confidence": 0.9, "slots": slots}


# --------------------------------------------------------------------------
# "embeddings"
# --------------------------------------------------------------------------

def fake_embed(text: str, dim: int = 128) -> list[float]:
    vector = [0.0] * dim
    for token in set(re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())):
        digest = hashlib.sha1(token.encode()).digest()
        vector[int.from_bytes(digest[:4], "big") % dim] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class FakeOllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path.startswith("/api/tags"):
            return self._json({"models": [{"name": name} for name in MODELS]})
        return self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        body = self._body()
        if self.path.startswith("/api/chat"):
            return self._json(self._chat(body))
        if self.path.startswith("/api/embed"):
            inputs = body.get("input") or []
            if isinstance(inputs, str):
                inputs = [inputs]
            return self._json({"embeddings": [fake_embed(text) for text in inputs]})
        if self.path.startswith("/api/embeddings"):
            return self._json({"embedding": fake_embed(body.get("prompt", ""))})
        return self._json({"error": "not found"}, 404)

    def _chat(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages") or []
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        if "extract durable memory candidates" in system:
            content = fake_extract(json.loads(user))
        elif "natural language inference" in system:
            premise = re.search(r"PREMISE: (.*)\nHYPOTHESIS:", user, re.DOTALL)
            hypothesis = re.search(r"HYPOTHESIS: (.*)$", user, re.DOTALL)
            content = fake_nli(premise.group(1) if premise else "", hypothesis.group(1) if hypothesis else "")
        elif "answer questions about a user" in system:
            content = fake_answer(json.loads(user))
        elif "classify a user's request" in system:
            content = fake_route(json.loads(user))
        elif "clean up a raw voice dictation" in system:
            payload = json.loads(user)
            cleaned = re.sub(r"\b(um+|uh+|you know|i mean)\b", "", str(payload.get("dictation") or ""),
                             flags=re.IGNORECASE)
            content = {"polished": re.sub(r"\s+", " ", cleaned).strip()}
        else:
            content = {"error": "unrecognised system prompt"}
        return {"model": body.get("model"), "message": {"role": "assistant", "content": json.dumps(content)},
                "done": True}

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode()) if length else {}

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def start(port: int = 11500) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", port), FakeOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11500)
    args = parser.parse_args()
    server, _ = start(args.port)
    print(f"fake ollama on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
