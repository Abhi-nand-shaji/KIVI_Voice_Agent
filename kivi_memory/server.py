"""Local HTTP server for the Kivi interface.

The API is split the way the product is: `/api/dictate` is regular dictation
(text in, written text out, memory only shapes the writing and learns quietly),
`/api/ask` is Hey Kivi (retrieval, answers, refusals, corrections). The
inspection endpoints exist but sit behind the interface's engineering view -
they are for an engineer asking why, not part of ordinary use.
"""

from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import KiviConfig
from .engine import KiviMemoryEngine
from .seed import read_jsonl


class KiviHandler(SimpleHTTPRequestHandler):
    engine: KiviMemoryEngine
    web_root = Path("web")

    # -- GET ----------------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        route = parsed.path

        if route == "/api/state":
            return self._json({
                "stats": self.engine.stats(),
                "backends": self.engine.backends(),
                "growth": self.engine.growth(),
                "usage": self.engine.usage(),
            })
        if route == "/api/summary":  # kept for compatibility
            return self._json({
                "stats": self.engine.stats(),
                "memories": self.engine.memories(12),
                "decisions": self.engine.decisions(12),
                "backends": self.engine.backends(),
            })
        if route == "/api/backends":
            return self._json(self.engine.backends())
        if route == "/api/growth":
            return self._json(self.engine.growth())
        if route == "/api/memories":
            limit = self._int(query, "limit", 100)
            return self._json({"memories": self.engine.memories(limit)})
        if route == "/api/decisions":
            limit = self._int(query, "limit", 100)
            return self._json({"decisions": self.engine.decisions(limit)})
        if route == "/api/transcripts":
            limit = self._int(query, "limit", 100)
            search = query.get("search", [None])[0]
            return self._json({"transcripts": self.engine.transcripts(limit, search)})
        if route == "/":
            self.path = "/index.html"
        return super().do_GET()

    # -- POST ---------------------------------------------------------------

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        body = self._body()

        if route == "/api/ask":
            query = str(body.get("query") or "").strip()
            if not query:
                return self._json({"error": "Ask me something."}, status=400)
            return self._json(self.engine.ask(query))

        if route == "/api/dictate":
            text = str(body.get("text") or "").strip()
            if not text:
                return self._json({"error": "Nothing to write."}, status=400)
            return self._json(self.engine.dictate(text, str(body.get("app") or "dictation")))

        if route == "/api/memories/forget":
            memory_id = str(body.get("id") or "").strip()
            if not memory_id:
                return self._json({"error": "Which memory?"}, status=400)
            result = self.engine.forget(memory_id)
            return self._json(result, status=200 if result.get("ok") else 404)

        if route == "/api/ingest":
            records = body.get("records")
            if not records and body.get("path"):
                records = read_jsonl(body["path"])
            if not isinstance(records, list):
                return self._json({"error": "Provide records[] or path"}, status=400)
            return self._json(self.engine.ingest_records(records))

        return self._json({"error": "not found"}, status=404)

    # -- plumbing -----------------------------------------------------------

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        clean = parsed.path.lstrip("/") or "index.html"
        return str((self.web_root / clean).resolve())

    def log_message(self, format: str, *args) -> None:
        return

    @staticmethod
    def _int(query: dict, key: str, default: int) -> int:
        try:
            return int(query.get(key, [str(default)])[0])
        except (TypeError, ValueError):
            return default

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)


def serve(
    db_path: str = "data/kivi.db",
    host: str = "127.0.0.1",
    port: int = 8000,
    config: KiviConfig | None = None,
) -> None:
    KiviHandler.engine = KiviMemoryEngine(db_path, config)
    backends = KiviHandler.engine.backends()
    server = ThreadingHTTPServer((host, port), KiviHandler)
    print(f"Kivi running at http://{host}:{port}")
    print(f"  extractor: {backends['extractor'].get('backend')}")
    print(f"  nli:       {backends['nli'].get('backend')}")
    print(f"  embedder:  {backends['embedder'].get('backend')}")
    print(f"  answerer:  {backends['answerer'].get('backend')}")
    print(f"  router:    {backends['router'].get('backend')}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()
