"""Read-only views over the database, for `inspect`, `metrics` and the UI.

Everything here is a projection of state that already exists - counts, the
memory list, the decision log, transcripts. No decisions, no model calls.

The decision log is the one that matters: every ADD / UPDATE / REINFORCE /
REJECT / NO_OP / FORGET ever taken, with the candidate that prompted it, the
reason, both scalars, which sensor proposed it and what NLI said. That table is
why "why do you know this?" is a join rather than a generated rationalisation.
"""

from __future__ import annotations

import json
from typing import Any

from .db import connect, migrate


def stats(engine) -> dict[str, Any]:
    with connect(engine.db_path) as conn:
        migrate(conn)
        counts = {}
        for table in ("transcripts", "memories", "memory_evidence", "decisions"):
            counts[table] = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]
        by_status = {
            row["status"]: row["n"]
            for row in conn.execute("SELECT status, count(*) AS n FROM memories GROUP BY status")
        }
        by_action = {
            row["action"]: row["n"]
            for row in conn.execute("SELECT action, count(*) AS n FROM decisions GROUP BY action")
        }
        by_source = {
            row["source"]: row["n"]
            for row in conn.execute("SELECT source, count(*) AS n FROM memories GROUP BY source")
        }
        nli_used = conn.execute(
            "SELECT count(*) AS n FROM decisions WHERE nli_label IS NOT NULL"
        ).fetchone()["n"]
        return {
            "counts": counts,
            "memories_by_status": by_status,
            "memories_by_source": by_source,
            "decisions_by_action": by_action,
            "decisions_with_nli": nli_used,
            "backends": engine.backends(),
        }

def memories(engine, limit: int = 100) -> list[dict[str, Any]]:
    with connect(engine.db_path) as conn:
        migrate(conn)
        rows = conn.execute(
            """
            SELECT * FROM memories
            ORDER BY status, utility DESC, confidence DESC, updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [engine._memory_payload(conn, row) for row in rows]

def decisions(engine, limit: int = 100) -> list[dict[str, Any]]:
    with connect(engine.db_path) as conn:
        migrate(conn)
        rows = conn.execute(
            """
            SELECT d.*, t.formatted_text, t.created_at, t.app
            FROM decisions d
            LEFT JOIN transcripts t ON t.id = d.transcript_id
            ORDER BY d.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["candidate"] = json.loads(item.pop("candidate_json"))
            result.append(item)
        return result

def transcripts(engine, limit: int = 100, search: str | None = None) -> list[dict[str, Any]]:
    with connect(engine.db_path) as conn:
        migrate(conn)
        if search:
            q = f"%{search.lower()}%"
            rows = conn.execute(
                """
                SELECT * FROM transcripts
                WHERE lower(formatted_text) LIKE ? OR lower(raw_asr) LIKE ? OR lower(app) LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (q, q, q, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM transcripts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
