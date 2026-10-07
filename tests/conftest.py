"""Shared test fixtures: a scriptable fake LLM and an isolated settings object."""
from __future__ import annotations

from pathlib import Path

import pytest

from janus.config import Settings


class FakeLLM:
    """Stands in for janus.llm.LLM. Returns scripted JSON dicts in order."""

    def __init__(self, script: list[dict]):
        self._script = list(script)
        self.tokens_used = 0
        self.calls = 0

    def chat_json(self, messages, *, retries: int = 2) -> dict:
        self.calls += 1
        self.tokens_used += 10
        if self._script:
            return self._script.pop(0)
        # Default: keep taking a harmless action so loops hit budget, not StopIteration.
        return {"thought": "idle", "action": {"tool": "list_dir", "args": {"path": "."}}}

    def ping(self):
        return True, "ok (fake)"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    ws = tmp_path / "ws"
    ws.mkdir()
    s = Settings(
        workspace=ws,
        max_iterations=40,
        max_seconds=1800,
        max_tokens=200_000,
        auto_approve_safe=True,
        trust=False,
    )
    return s


@pytest.fixture(autouse=True)
def _pin_root(monkeypatch, tmp_path):
    """Force config._find_config_dir to resolve to tmp so db/stopfile stay isolated."""
    import janus.config as cfg
    monkeypatch.setattr(cfg, "_find_config_dir", lambda start=None: tmp_path)
