"""Model lineage registry.

Tracks every Janus model version produced by self-training: its Ollama tag, its
parent, the base weights and dataset it came from, its benchmark scorecard, and
whether it has been adopted as the active model. This is what makes Janus "its
own model" — a versioned lineage descended from the base Gemma weights.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    version     INTEGER NOT NULL,
    tag         TEXT NOT NULL UNIQUE,
    parent_tag  TEXT,
    base_model  TEXT NOT NULL,
    dataset_hash TEXT,
    adapter_dir TEXT,
    created_at  REAL NOT NULL,
    scorecard   TEXT,
    adopted     INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'built'
);
"""


@dataclass
class ModelRecord:
    version: int
    tag: str
    parent_tag: str | None
    base_model: str
    adopted: bool
    status: str
    scorecard: dict | None


class ModelRegistry:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def next_version(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(version),0) AS v FROM models").fetchone()
        return int(row["v"]) + 1

    def current_tag(self, default: str) -> str:
        """The adopted model with the highest version, else the given default."""
        row = self._conn.execute(
            "SELECT tag FROM models WHERE adopted=1 ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return row["tag"] if row else default

    def register(self, *, version: int, tag: str, parent_tag: str | None,
                 base_model: str, dataset_hash: str, adapter_dir: str,
                 scorecard: dict | None = None, status: str = "built") -> None:
        self._conn.execute(
            "INSERT INTO models (version, tag, parent_tag, base_model, dataset_hash, "
            "adapter_dir, created_at, scorecard, adopted, status) "
            "VALUES (?,?,?,?,?,?,?,?,0,?)",
            (version, tag, parent_tag, base_model, dataset_hash, adapter_dir,
             time.time(), json.dumps(scorecard) if scorecard else None, status),
        )
        self._conn.commit()

    def set_scorecard(self, tag: str, scorecard: dict) -> None:
        self._conn.execute("UPDATE models SET scorecard=? WHERE tag=?",
                           (json.dumps(scorecard), tag))
        self._conn.commit()

    def adopt(self, tag: str) -> None:
        self._conn.execute("UPDATE models SET adopted=0")
        self._conn.execute("UPDATE models SET adopted=1, status='adopted' WHERE tag=?", (tag,))
        self._conn.commit()

    def set_status(self, tag: str, status: str) -> None:
        self._conn.execute("UPDATE models SET status=? WHERE tag=?", (status, tag))
        self._conn.commit()

    def list(self) -> list[ModelRecord]:
        rows = self._conn.execute("SELECT * FROM models ORDER BY version").fetchall()
        return [
            ModelRecord(
                version=r["version"], tag=r["tag"], parent_tag=r["parent_tag"],
                base_model=r["base_model"], adopted=bool(r["adopted"]), status=r["status"],
                scorecard=json.loads(r["scorecard"]) if r["scorecard"] else None,
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
