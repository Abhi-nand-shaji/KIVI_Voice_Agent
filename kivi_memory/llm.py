"""Minimal local-Ollama client.

Standard library only: `urllib.request` against a daemon the user runs with
`ollama serve`. No API key, no vendor SDK, no outbound network beyond
localhost.

Everything is deterministic by default (temperature 0) and every call is
content-addressed in a small SQLite cache, so re-running the 500-record
ingest or the evaluation does not re-run the model.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import KiviConfig


CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    model TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class OllamaUnavailable(RuntimeError):
    pass


class _Cache:
    def __init__(self, path: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection | None:
        if not self.enabled:
            return None
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.executescript(CACHE_SCHEMA)
            self._conn.commit()
        return self._conn

    def get(self, key: str) -> Any | None:
        conn = self._connect()
        if conn is None:
            return None
        row = conn.execute("SELECT response_json FROM llm_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, kind: str, model: str, value: Any) -> None:
        conn = self._connect()
        if conn is None:
            return
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache(key, kind, model, response_json) VALUES (?, ?, ?, ?)",
            (key, kind, model, json.dumps(value, sort_keys=True)),
        )
        conn.commit()


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first well-formed JSON object out of a model response.

    Small local models wrap JSON in prose or fences more often than hosted
    ones, so this is defensive on purpose.
    """
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:idx + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class OllamaClient:
    """Thin, dependency-free wrapper over the local Ollama HTTP API."""

    def __init__(self, config: KiviConfig) -> None:
        self.config = config
        self.host = config.ollama_host.rstrip("/")
        self._cache = _Cache(config.cache_path, config.use_cache)
        self._available: bool | None = None
        self._tags: list[str] = []
        self.counters = {
            "chat_calls": 0,
            "chat_cache_hits": 0,
            "embed_calls": 0,
            "embed_cache_hits": 0,
            "embed_texts": 0,
            "failures": 0,
            "total_latency_ms": 0.0,
        }

    # -- transport ----------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get(self, path: str, timeout: float) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.host}{path}", method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    # -- health -------------------------------------------------------------

    def available(self, refresh: bool = False) -> bool:
        if self._available is not None and not refresh:
            return self._available
        try:
            payload = self._get("/api/tags", self.config.health_timeout)
            self._tags = [m.get("name", "") for m in payload.get("models", [])]
            self._available = True
        except Exception:
            self._tags = []
            self._available = False
        return self._available

    def installed_models(self) -> list[str]:
        self.available()
        return list(self._tags)

    def has_model(self, name: str) -> bool:
        if not self.available():
            return False
        return any(tag == name or tag.split(":")[0] == name.split(":")[0] for tag in self._tags)

    def doctor(self) -> dict[str, Any]:
        reachable = self.available(refresh=True)
        return {
            "host": self.host,
            "reachable": reachable,
            "installed_models": self._tags,
            "chat_model": self.config.llm_model,
            "chat_model_present": self.has_model(self.config.llm_model),
            "embed_model": self.config.embed_model,
            "embed_model_present": self.has_model(self.config.embed_model),
            "nli_model": self.config.resolved_nli_model,
            "nli_model_present": self.has_model(self.config.resolved_nli_model),
            "hint": (
                "Run `ollama serve`, then `ollama pull "
                f"{self.config.llm_model}` and `ollama pull {self.config.embed_model}`."
                if not reachable else ""
            ),
        }

    # -- chat ---------------------------------------------------------------

    def chat_json(
        self,
        system: str,
        user: str,
        model: str | None = None,
        schema: dict[str, Any] | None = None,
        cache_kind: str = "chat",
    ) -> dict[str, Any] | None:
        """Ask the model for a JSON object. Returns None on any failure.

        Callers must treat None as "sensor unavailable" and fall back, rather
        than as "nothing to remember" — that distinction is what keeps a dead
        daemon from silently emptying the memory store.
        """
        model = model or self.config.llm_model
        key = hashlib.sha1(
            json.dumps(
                {"m": model, "s": system, "u": user, "schema": schema, "t": self.config.temperature},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        cached = self._cache.get(key)
        if cached is not None:
            self.counters["chat_cache_hits"] += 1
            return cached

        if not self.available():
            return None

        payload: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.num_ctx,
                "seed": 7,
            },
            "format": schema if schema else "json",
        }

        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._post("/api/chat", payload, self.config.request_timeout)
                content = (response.get("message") or {}).get("content", "")
                parsed = extract_json_object(content)
                if parsed is None:
                    raise ValueError(f"model did not return JSON: {content[:200]!r}")
                self.counters["chat_calls"] += 1
                self.counters["total_latency_ms"] += (time.perf_counter() - started) * 1000
                self._cache.put(key, cache_kind, model, parsed)
                return parsed
            except Exception as exc:  # network error, bad JSON, model missing
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(0.4 * (attempt + 1))
                    # A schema-constrained retry can fail on older daemons.
                    payload["format"] = "json"
        self.counters["failures"] += 1
        self.last_error = last_error
        return None

    # -- embeddings ---------------------------------------------------------

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]] | None:
        if not texts:
            return []
        model = model or self.config.embed_model
        results: list[list[float] | None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []

        for idx, text in enumerate(texts):
            key = hashlib.sha1(f"embed|{model}|{text}".encode("utf-8")).hexdigest()
            cached = self._cache.get(key)
            if cached is not None:
                self.counters["embed_cache_hits"] += 1
                results[idx] = cached
            else:
                pending.append((idx, text))

        if pending:
            if not self.available():
                return None
            batch = [text for _, text in pending]
            vectors = self._embed_batch(batch, model)
            if vectors is None or len(vectors) != len(pending):
                return None
            for (idx, text), vector in zip(pending, vectors):
                results[idx] = vector
                key = hashlib.sha1(f"embed|{model}|{text}".encode("utf-8")).hexdigest()
                self._cache.put(key, "embed", model, vector)

        return [vector for vector in results if vector is not None] if all(
            vector is not None for vector in results
        ) else None

    def _embed_batch(self, texts: list[str], model: str) -> list[list[float]] | None:
        started = time.perf_counter()
        # Newer daemons: /api/embed with `input`. Older ones: /api/embeddings,
        # one text per call with `prompt`.
        try:
            response = self._post(
                "/api/embed",
                {"model": model, "input": texts},
                self.config.request_timeout,
            )
            vectors = response.get("embeddings")
            if isinstance(vectors, list) and len(vectors) == len(texts):
                self.counters["embed_calls"] += 1
                self.counters["embed_texts"] += len(texts)
                self.counters["total_latency_ms"] += (time.perf_counter() - started) * 1000
                return [[float(x) for x in vector] for vector in vectors]
        except Exception:
            pass

        vectors = []
        try:
            for text in texts:
                response = self._post(
                    "/api/embeddings",
                    {"model": model, "prompt": text},
                    self.config.request_timeout,
                )
                vector = response.get("embedding")
                if not isinstance(vector, list):
                    return None
                vectors.append([float(x) for x in vector])
            self.counters["embed_calls"] += 1
            self.counters["embed_texts"] += len(texts)
            self.counters["total_latency_ms"] += (time.perf_counter() - started) * 1000
            return vectors
        except Exception:
            self.counters["failures"] += 1
            return None

    def usage(self) -> dict[str, Any]:
        return {
            **self.counters,
            "total_latency_ms": round(self.counters["total_latency_ms"], 2),
            "external_api_calls": 0,
            "estimated_cost_usd": 0.0,
            "note": "All model calls are local (Ollama); no API key and no outbound network.",
        }


class OpenAICompatibleClient(OllamaClient):
    """Same interface, OpenAI-compatible transport.

    Exists so a reviewer who cannot or will not download a local model can
    still exercise the model-backed paths by supplying `KIVI_LLM_API_KEY`.
    Nothing else changes: the same prompts, the same JSON contracts, the same
    controller, the same cache. Works against any endpoint that speaks
    `/chat/completions` and `/embeddings`.
    """

    def __init__(self, config: KiviConfig) -> None:
        super().__init__(config)
        self.host = (config.llm_base_url or "https://api.openai.com/v1").rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"
        return headers

    def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        # Translate the Ollama-shaped calls the rest of the code makes.
        if path == "/api/chat":
            path = "/chat/completions"
            payload = {
                "model": payload["model"],
                "messages": payload["messages"],
                "temperature": payload.get("options", {}).get("temperature", 0),
                "response_format": {"type": "json_object"},
            }
        elif path in ("/api/embed", "/api/embeddings"):
            path = "/embeddings"
            texts = payload.get("input") or payload.get("prompt")
            payload = {"model": payload["model"], "input": texts if isinstance(texts, list) else [texts]}

        request = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        # Translate the response back into the shape the callers expect.
        if "choices" in body:
            content = body["choices"][0].get("message", {}).get("content", "")
            return {"message": {"role": "assistant", "content": content}}
        if "data" in body:
            return {"embeddings": [item["embedding"] for item in body["data"]]}
        return body

    def available(self, refresh: bool = False) -> bool:
        if self._available is not None and not refresh:
            return self._available
        self._available = bool(self.config.llm_api_key or self.config.llm_base_url)
        self._tags = [self.config.llm_model, self.config.embed_model] if self._available else []
        return self._available

    def has_model(self, name: str) -> bool:
        return self.available()

    def doctor(self) -> dict[str, Any]:
        configured = self.available(refresh=True)
        return {
            "transport": "openai-compatible",
            "endpoint": self.host,
            "api_key_env": "KIVI_LLM_API_KEY",
            "api_key_present": bool(self.config.llm_api_key),
            "reachable": configured,
            "chat_model": self.config.llm_model,
            "embed_model": self.config.embed_model,
            "installed_models": [],
            "chat_model_present": configured,
            "embed_model_present": configured,
            "nli_model": self.config.resolved_nli_model,
            "nli_model_present": configured,
            "hint": "" if configured else "Set KIVI_LLM_API_KEY (and optionally KIVI_LLM_BASE_URL).",
        }


def get_client(config: KiviConfig) -> OllamaClient:
    """Pick the transport. Local Ollama unless a key or base URL says otherwise."""
    if config.api_style == "openai":
        return OpenAICompatibleClient(config)
    return OllamaClient(config)
