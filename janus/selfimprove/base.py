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


def approve_self_edit(gate, kind: str, args: dict) -> tuple[bool, str]:
    """Route a self-edit through the approval gate as a RISKY action.

    `args` is what the approver sees, so it should include the actual change
    (the diff, the module source), not just a summary. No gate means no check
    was requested (e.g. unit tests)."""
    if gate is None:
        return True, "no gate"
    from pydantic import BaseModel, ConfigDict

    from janus.tools.base import Risk, Tool

    class _Args(BaseModel):
        model_config = ConfigDict(extra="allow")

    sentinel = Tool(kind, "edit Janus itself", _Args, lambda a, *, settings: None, Risk.RISKY)
    res = gate.decide(sentinel, args)
    return res.decision.value == "approved", res.reason


class Strategy(Protocol):
    name: str

    def propose(self, ctx: ImproveContext) -> AppliedChange | None:
        """Apply a candidate change reversibly, or return None if nothing to do."""
        ...
