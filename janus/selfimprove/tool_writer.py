"""Layer 2 — write a new tool for itself.

Asks the model to author a new tool module for janus/tools/dynamic/, writes it,
then validates it in a sandboxed subprocess (import + the model-provided
self-test). Because a dynamic tool is imported into Janus's own process, the
module is a self-edit: it goes through the approval gate (with its source shown)
before it is written. Kept only if it imports and self-tests pass; reversible by
deleting the module file.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from janus import sandbox
from janus.selfimprove.base import AppliedChange, ImproveContext, approve_self_edit

_AUTHOR = """Write a new tool for the Janus agent as a Python module.

A tool module must define a top-level `TOOL = Tool(...)` using this API:

    from janus.tools.base import Tool, ToolResult, Risk
    from pydantic import BaseModel, Field

    class Args(BaseModel):
        x: str = Field(description="...")

    def _run(a: Args, *, settings) -> ToolResult:
        return ToolResult(True, "result text")

    TOOL = Tool("tool_name", "what it does", Args, _run, Risk.SAFE)

Return JSON:
{{"filename": "snake_case.py", "module_code": "<full module source>",
  "selftest": "<python that imports the module and asserts it works, printing OK>"}}

The tool should be genuinely useful for future tasks (e.g. a text or math helper).
Keep it SAFE (no filesystem writes outside args, no shell)."""


class ToolWriterStrategy:
    name = "tool_writer"

    def __init__(self, gate=None) -> None:
        self.gate = gate  # optional ApprovalGate; if set, ask before adding the tool

    def propose(self, ctx: ImproveContext) -> AppliedChange | None:
        dyn_dir = Path(__file__).resolve().parent.parent / "tools" / "dynamic"
        try:
            out = ctx.llm.chat_json([{"role": "user", "content": _AUTHOR}])
        except Exception as e:  # noqa: BLE001
            ctx.say(f"[dim]tool_writer: generation failed: {e}[/]")
            return None

        filename = str(out.get("filename", "")).strip()
        code = out.get("module_code", "")
        selftest = out.get("selftest", "")
        if not (filename.endswith(".py") and code and selftest) or filename.startswith("_"):
            ctx.say("[dim]tool_writer: malformed proposal[/]")
            return None

        target = dyn_dir / Path(filename).name
        if target.exists():
            ctx.say(f"[dim]tool_writer: {filename} already exists, skipping[/]")
            return None

        approved, reason = approve_self_edit(self.gate, "self_add_tool", {
            "filename": target.name, "module_code": code, "selftest": selftest})
        if not approved:
            ctx.say(f"[yellow]tool_writer: blocked by approval gate ({reason})[/]")
            return None

        target.write_text(code)
        ok, detail = _validate(target, selftest, ctx.settings)
        if not ok:
            target.unlink(missing_ok=True)
            ctx.say(f"[dim]tool_writer: {filename} failed validation, discarded: {detail}[/]")
            return None

        ctx.say(f"[cyan]tool_writer[/]: added tool module {filename}")

        def revert() -> None:
            target.unlink(missing_ok=True)

        return AppliedChange(f"added dynamic tool {filename}", revert)


def _validate(module_path: Path, selftest: str, settings) -> tuple[bool, str]:
    """Import the module and run its self-test in a (sandboxed) subprocess."""
    repo_root = module_path.resolve().parents[3]
    mod_name = f"janus.tools.dynamic.{module_path.stem}"
    script = (
        f"import importlib\n"
        f"m = importlib.import_module({mod_name!r})\n"
        f"assert hasattr(m, 'TOOL'), 'no TOOL defined'\n"
        f"{selftest}\n"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="janus-tooltest-") as scratch:
            proc = sandbox.run(
                [sys.executable, "-B", "-c", script], settings=settings,
                writable=Path(scratch), cwd=repo_root, extra_ro=(repo_root,), timeout=60,
            )
        if proc.returncode == 0 and "OK" in proc.stdout:
            return True, "ok"
        return False, (proc.stderr or proc.stdout)[:300]
    except Exception as e:  # noqa: BLE001
        return False, str(e)
