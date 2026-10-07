"""Common types for self-improvement strategies.

A Strategy inspects recent runs and proposes exactly one change, applying it in
a reversible way and returning an AppliedChange whose `revert` undoes it. The
orchestrator (run_improve) then measures the benchmark and either keeps the
change or calls `revert`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


def _noop() -> None:
    return None


@dataclass
class AppliedChange:
    description: str
    revert: Callable[[], None]
    keep: Callable[[], None] = _noop  # called when the change is accepted (e.g. merge a branch)


@dataclass
class ImproveContext:
    settings: object      # Settings
    llm: object           # LLM
    memory: object        # Memory
    console: object | None = None

    def say(self, msg: str) -> None:
        if self.console is not None:
            self.console.print(msg)


class Strategy(Protocol):
    name: str

    def propose(self, ctx: ImproveContext) -> AppliedChange | None:
        """Apply a candidate change reversibly, or return None if nothing to do."""
        ...
