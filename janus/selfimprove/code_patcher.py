"""Layer 3 — patch Janus's own source code.

The model proposes a unified diff against the janus/ package. It is applied on a
throwaway git branch (via PatchTransaction), then the test suite runs. If tests
fail the transaction is reverted immediately and nothing is kept. If tests pass,
an AppliedChange is returned whose `keep` merges the branch and whose `revert`
discards it — the orchestrator decides based on the benchmark. Always routed
through the approval gate before any of this happens.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from janus import sandbox
from janus.selfimprove.base import AppliedChange, ImproveContext, approve_self_edit
from janus.selfimprove.gittx import PatchTransaction, has_commit, patch_paths

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

        # The patch may only touch the file it was shown. In particular it must not
        # touch tests/, which are what judge it.
        touched = patch_paths(root, diff)
        if touched != {target_rel}:
            ctx.say(f"[yellow]code_patcher: rejected patch touching {sorted(touched) or 'nothing'} "
                    f"(only {target_rel} is allowed)[/]")
            return None

        # Approval gate: self-code edits are always risky. Show the approver the diff.
        approved, reason = approve_self_edit(self.gate, "self_code_patch", {
            "path": target_rel, "rationale": out.get("rationale", ""), "diff": diff})
        if not approved:
            ctx.say(f"[yellow]code_patcher: blocked by approval gate ({reason})[/]")
            return None

        tx = PatchTransaction(root=root)
        ok, detail = tx.begin_and_apply(diff, paths=[target_rel])
        if not ok:
            ctx.say(f"[dim]code_patcher: {detail}[/]")
            return None

        tests_ok, test_detail = _run_tests(root, ctx.settings)
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


def _run_tests(root: Path, settings) -> tuple[bool, str]:
    """Run the suite against the patched code, sandboxed when possible: the repo is
    read-only there, so the patched code can't touch anything but a scratch dir."""
    scratch = Path(tempfile.mkdtemp(prefix="janus-selftest-"))
    try:
        proc = sandbox.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
             "--basetemp", str(scratch / "pytest")],
            settings=settings, writable=scratch, cwd=root, extra_ro=(root,), timeout=600,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return proc.returncode == 0, (proc.stdout or proc.stderr)[-300:]
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
