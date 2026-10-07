"""Layer 3 — patch Janus's own source code.

The model proposes a unified diff against the janus/ package. It is applied on a
throwaway git branch (via PatchTransaction), then the test suite runs. If tests
fail the transaction is reverted immediately and nothing is kept. If tests pass,
an AppliedChange is returned whose `keep` merges the branch and whose `revert`
discards it — the orchestrator decides based on the benchmark. Always routed
through the approval gate before any of this happens.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from janus.selfimprove.base import AppliedChange, ImproveContext
from janus.selfimprove.gittx import PatchTransaction, has_commit

_PATCH = """You may improve Janus's own source to make it more efficient (fewer tokens,
fewer steps, less memory) WITHOUT reducing correctness.

Here is the current source of a module you may change:

FILE: {path}
```
{source}
```

Propose a SMALL, safe improvement as a unified diff (git apply format) with
correct `--- a/{path}` and `+++ b/{path}` headers and @@ hunks. Do not reformat
unrelated code. Return JSON:
{{"rationale": "<what and why>", "diff": "<unified diff>"}}
If you see no safe improvement, return {{"diff": ""}}."""

# Candidate files the patcher is allowed to consider (never config/approval/cli safety paths
# are excluded here deliberately so the gate+tests guard them instead of a blocklist).
_CANDIDATES = ["janus/agent.py", "janus/llm.py", "janus/tools/builtins.py"]


class CodePatcherStrategy:
    name = "code_patcher"

    def __init__(self, gate=None) -> None:
        self.gate = gate  # optional ApprovalGate; if set, ask before patching

    def propose(self, ctx: ImproveContext) -> AppliedChange | None:
        root = ctx.settings.root
        if not has_commit(root):
            ctx.say("[dim]code_patcher: repo has no commits yet; skipping[/]")
            return None

        target_rel = _pick_target(root)
        if target_rel is None:
            return None
        source = (root / target_rel).read_text()

        try:
            out = ctx.llm.chat_json([
                {"role": "user", "content": _PATCH.format(path=target_rel, source=source)}
            ])
        except Exception as e:  # noqa: BLE001
            ctx.say(f"[dim]code_patcher: generation failed: {e}[/]")
            return None
        diff = (out.get("diff") or "").strip()
        if not diff:
            ctx.say("[dim]code_patcher: model proposed no change[/]")
            return None

        # Approval gate: self-code edits are always risky.
        if self.gate is not None:
            from janus.tools.base import Risk, Tool
            from pydantic import BaseModel

            class _PatchArgs(BaseModel):
                path: str
                rationale: str = ""

            sentinel = Tool("self_code_patch", "edit Janus's own source", _PatchArgs,
                            lambda a, *, settings: None, Risk.RISKY)
            decision = self.gate.decide(sentinel, {"path": target_rel,
                                                   "rationale": out.get("rationale", "")})
            if decision.decision.value != "approved":
                ctx.say(f"[yellow]code_patcher: blocked by approval gate ({decision.reason})[/]")
                return None

        tx = PatchTransaction(root=root)
        ok, detail = tx.begin_and_apply(diff)
        if not ok:
            ctx.say(f"[dim]code_patcher: {detail}[/]")
            return None

        tests_ok, test_detail = _run_tests(root)
        if not tests_ok:
            tx.revert()
            ctx.say(f"[yellow]code_patcher: tests failed, reverted ({test_detail})[/]")
            return None

        ctx.say(f"[cyan]code_patcher[/]: candidate patch to {target_rel} passed tests")
        return AppliedChange(f"patched {target_rel}", revert=tx.revert, keep=tx.keep)


def _pick_target(root: Path) -> str | None:
    for rel in _CANDIDATES:
        if (root / rel).is_file():
            return rel
    return None


def _run_tests(root: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header"],
            cwd=str(root), capture_output=True, text=True, timeout=600,
        )
        return proc.returncode == 0, (proc.stdout or proc.stderr)[-300:]
    except Exception as e:  # noqa: BLE001
        return False, str(e)
