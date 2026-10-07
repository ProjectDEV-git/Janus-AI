"""SQLite-backed persistence: audit log, episodic runs, and learned lessons.

One database (settings.db_path) holds everything Janus needs across runs:
  - runs:    one row per task (goal, status, metrics)
  - events:  append-only audit log of every step within a run
  - lessons: free-text strategy notes Janus writes to improve future behavior
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    goal        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    started_at  REAL NOT NULL,
    ended_at    REAL,
    iterations  INTEGER DEFAULT 0,
    tokens      INTEGER DEFAULT 0,
    result      TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   INTEGER NOT NULL,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,          -- thought | action | observation | decision | note
    payload  TEXT NOT NULL,          -- JSON
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE TABLE IF NOT EXISTS lessons (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    tags     TEXT NOT NULL DEFAULT '',
    text     TEXT NOT NULL
);
"""


@dataclass
class Memory:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- runs ---
    def start_run(self, goal: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs (goal, status, started_at) VALUES (?, 'running', ?)",
            (goal, time.time()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def finish_run(
        self, run_id: int, *, status: str, iterations: int, tokens: int, result: str
    ) -> None:
        self._conn.execute(
            "UPDATE runs SET status=?, ended_at=?, iterations=?, tokens=?, result=? WHERE id=?",
            (status, time.time(), iterations, tokens, result, run_id),
        )
        self._conn.commit()

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()

    def recent_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    # --- events (audit log) ---
    def log_event(self, run_id: int, kind: str, payload: dict) -> None:
        self._conn.execute(
            "INSERT INTO events (run_id, ts, kind, payload) VALUES (?, ?, ?, ?)",
            (run_id, time.time(), kind, json.dumps(payload, default=str)),
        )
        self._conn.commit()

    def events_for(self, run_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()

    # --- lessons (semantic memory) ---
    def add_lesson(self, text: str, tags: str = "") -> None:
        self._conn.execute(
            "INSERT INTO lessons (ts, tags, text) VALUES (?, ?, ?)",
            (time.time(), tags, text),
        )
        self._conn.commit()

    def recall_lessons(self, query: str = "", limit: int = 8) -> list[str]:
        """Return lesson texts, keyword-filtered when a query is given, newest first."""
        if query.strip():
            like = f"%{query.strip()}%"
            rows = self._conn.execute(
                "SELECT text FROM lessons WHERE text LIKE ? OR tags LIKE ? "
                "ORDER BY id DESC LIMIT ?",
                (like, like, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT text FROM lessons ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [r["text"] for r in rows]

    def close(self) -> None:
        self._conn.close()
