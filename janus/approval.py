"""The approval gate — Janus's primary safety control.

Every tool call passes through `decide`, which classifies it as SAFE or RISKY
(using the tool's own risk assessment) and then applies policy:

  - SAFE   -> approved automatically when auto_approve_safe is set.
  - RISKY  -> approved automatically only when trust is set; otherwise the
              `prompter` callback is asked (interactively, a y/n). If there is
              no prompter (non-interactive run), RISKY calls are BLOCKED.

The decision, its reason, and the policy that produced it are returned so the
caller can log every gate decision to the audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from janus.tools.base import Risk, Tool

# prompter(tool_name, args, reason) -> True to allow, False to deny.
Prompter = Callable[[str, dict, str], bool]


class Decision(str, Enum):
    APPROVED = "approved"
    BLOCKED = "blocked"


@dataclass
class GateResult:
    decision: Decision
    risk: Risk
    reason: str
    policy: str  # which rule decided it (for the audit log)


class ApprovalGate:
    def __init__(self, settings, prompter: Prompter | None = None) -> None:
        self.settings = settings
        self.prompter = prompter

    def decide(self, tool: Tool, args: dict) -> GateResult:
        risk, reason = tool.risk_for(args, self.settings)

        if risk == Risk.SAFE:
            if self.settings.auto_approve_safe:
                return GateResult(Decision.APPROVED, risk, reason or "safe action",
                                  "auto_approve_safe")
            # Even "safe" needs confirmation when auto-approve is off.
            return self._ask_or_block(tool, args, risk, reason or "safe action")

        # RISKY
        if self.settings.trust:
            return GateResult(Decision.APPROVED, risk,
                              reason or "risky action", "trust_mode")
        return self._ask_or_block(tool, args, risk, reason or "risky action")

    def _ask_or_block(self, tool: Tool, args: dict, risk: Risk, reason: str) -> GateResult:
        if self.prompter is None:
            return GateResult(Decision.BLOCKED, risk,
                              f"{reason} (no interactive approver available)",
                              "blocked_noninteractive")
        allowed = self.prompter(tool.name, args, reason)
        if allowed:
            return GateResult(Decision.APPROVED, risk, reason, "user_approved")
        return GateResult(Decision.BLOCKED, risk, reason, "user_denied")
