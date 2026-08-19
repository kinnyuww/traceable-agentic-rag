from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS knowledge_bases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    active_index_version_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    object_path TEXT NOT NULL,
    artifact_path TEXT,
    status TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(knowledge_base_id, sha256, version)
);

CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(knowledge_base_id);

CREATE TABLE IF NOT EXISTS index_versions (
    id TEXT PRIMARY KEY,
    knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    document_manifest_json TEXT NOT NULL DEFAULT '[]',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    activated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_index_versions_kb ON index_versions(knowledge_base_id);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    index_version_id TEXT NOT NULL REFERENCES index_versions(id) ON DELETE CASCADE,
    knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    contextual_text TEXT NOT NULL DEFAULT '',
    source_json TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(index_version_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    index_version_id UNINDEXED,
    text,
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    locked_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);

CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress REAL NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(job_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id, sequence);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    index_version_id TEXT,
    conversation_id TEXT,
    question TEXT NOT NULL,
    route TEXT,
    status TEXT NOT NULL,
    answer TEXT,
    citations_json TEXT NOT NULL DEFAULT '[]',
    rounds INTEGER NOT NULL DEFAULT 0,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_kb_created ON runs(knowledge_base_id, created_at);

CREATE TABLE IF NOT EXISTS trace_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    stage TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, sequence)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    examples_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    completed_at TEXT
);
"""


class Database:
    """Small SQLite boundary with one connection per operation.

    Connections are intentionally short-lived so the API and worker containers
    can share the same WAL database without keeping fork-unsafe global handles.
    """

    def __init__(self, path: Path):
        self.path = path
        self._init_lock = threading.Lock()
        self._initialized = False

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as connection:
                connection.executescript(SCHEMA)
                run_columns = {
                    row["name"] for row in connection.execute("PRAGMA table_info(runs)")
                }
                if "conversation_id" not in run_columns:
                    connection.execute("ALTER TABLE runs ADD COLUMN conversation_id TEXT")
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_runs_conversation_created
                    ON runs(knowledge_base_id, conversation_id, created_at)
                    """
                )
            self._initialized = True

    def execute(self, sql: str, parameters: Iterable[Any] = ()) -> None:
        self.initialize()
        with self.connect() as connection:
            connection.execute(sql, tuple(parameters))
            connection.commit()

    def execute_many(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        self.initialize()
        with self.connect() as connection:
            connection.executemany(sql, rows)
            connection.commit()

    def fetch_one(self, sql: str, parameters: Iterable[Any] = ()) -> sqlite3.Row | None:
        self.initialize()
        with self.connect() as connection:
            return connection.execute(sql, tuple(parameters)).fetchone()

    def fetch_all(self, sql: str, parameters: Iterable[Any] = ()) -> list[sqlite3.Row]:
        self.initialize()
        with self.connect() as connection:
            return list(connection.execute(sql, tuple(parameters)).fetchall())

    def transaction(self) -> sqlite3.Connection:
        self.initialize()
        connection = self.connect()
        connection.execute("BEGIN IMMEDIATE")
        return connection


def decode_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
