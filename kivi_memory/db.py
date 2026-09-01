from __future__ import annotations

import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path("data/kivi.db")


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS transcripts (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    app TEXT NOT NULL,
    raw_asr TEXT NOT NULL,
    formatted_text TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transcripts_created_at ON transcripts(created_at);
CREATE INDEX IF NOT EXISTS idx_transcripts_app ON transcripts(app);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    canonical_text TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 'tentative', 'archived')),
    confidence REAL NOT NULL,
    utility REAL NOT NULL,
    importance REAL NOT NULL,
    recurrence INTEGER NOT NULL DEFAULT 1,
    decay_rate REAL NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    supersedes_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (supersedes_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_key
    ON memories(memory_type, subject, predicate, scope, object);

CREATE TABLE IF NOT EXISTS memory_evidence (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    transcript_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    snippet TEXT NOT NULL,
    contribution REAL NOT NULL,
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memories(id),
    FOREIGN KEY (transcript_id) REFERENCES transcripts(id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_memory ON memory_evidence(memory_id);
CREATE INDEX IF NOT EXISTS idx_evidence_transcript ON memory_evidence(transcript_id);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    transcript_id TEXT,
    action TEXT NOT NULL,
    target_memory_id TEXT,
    candidate_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0,
    utility REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(id),
    FOREIGN KEY (target_memory_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_decisions_action ON decisions(action);
CREATE INDEX IF NOT EXISTS idx_decisions_transcript ON decisions(transcript_id);

CREATE TABLE IF NOT EXISTS embeddings (
    owner_kind TEXT NOT NULL CHECK(owner_kind IN ('memory', 'transcript')),
    owner_id TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (owner_kind, owner_id, model)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_owner ON embeddings(owner_kind, owner_id);

-- Database growth over time. The assignment asks for growth to be reported
-- "wherever it matters"; a single final row count cannot show whether memory
-- grows with the transcript log (bad) or sublinearly (the product claim).
CREATE TABLE IF NOT EXISTS growth_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sampled_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    transcripts INTEGER NOT NULL,
    memories INTEGER NOT NULL,
    active_memories INTEGER NOT NULL,
    decisions INTEGER NOT NULL,
    evidence_rows INTEGER NOT NULL,
    embedding_rows INTEGER NOT NULL,
    db_bytes INTEGER NOT NULL
);
"""


# Columns added after the first version of the schema shipped. Applied with
# ALTER TABLE so an existing data/kivi.db keeps its rows.
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "decisions": {
        "extractor": "TEXT NOT NULL DEFAULT 'rule'",
        "nli_label": "TEXT",
        "nli_probability": "REAL",
    },
    "memories": {
        "source": "TEXT NOT NULL DEFAULT 'rule'",
    },
}


def _ensure_columns(conn: sqlite3.Connection) -> None:
    for table, columns in ADDED_COLUMNS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, declaration in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _ensure_columns(conn)
    conn.commit()


def reset_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    path = Path(db_path)
    if path.exists():
        path.unlink()
    conn = connect(path)
    try:
        migrate(conn)
    finally:
        conn.close()
