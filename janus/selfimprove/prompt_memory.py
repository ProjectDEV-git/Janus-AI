"""Layer 1 — rewrite prompts/memory: distill lessons from recent runs.

Cheapest self-improvement. Reflects on recent runs (especially failures) and
writes short, actionable lessons into memory, which are injected into future
runs. Reversible by deleting the lessons added this round.
"""
from __future__ import annotations

import json

from janus.selfimprove.base import AppliedChange, ImproveContext

_REFLECT = """You are improving an agent by writing LESSONS it will read before future tasks.
Here are summaries of recent runs (goal, status, result):

{runs}

Write 1-3 short, concrete, reusable lessons that would make the agent more
efficient or more reliable (fewer steps, fewer tokens, avoid a past mistake).
Return JSON: {{"lessons": ["...", "..."]}}. Lessons must be general, not about one task."""


class PromptMemoryStrategy:
    name = "prompt_memory"

    def propose(self, ctx: ImproveContext) -> AppliedChange | None:
        runs = ctx.memory.recent_runs(limit=10)
        if not runs:
            ctx.say("[dim]prompt_memory: no runs to reflect on[/]")
            return None
        summary = "\n".join(
            f"- goal={r['goal'][:80]!r} status={r['status']} result={(r['result'] or '')[:80]!r}"
            for r in runs
        )
        try:
            out = ctx.llm.chat_json([
                {"role": "user", "content": _REFLECT.format(runs=summary)}
            ])
            lessons = [str(x).strip() for x in out.get("lessons", []) if str(x).strip()]
        except Exception as e:  # noqa: BLE001
            ctx.say(f"[dim]prompt_memory: reflection failed: {e}[/]")
            return None
        if not lessons:
            return None

        before_max = _max_lesson_id(ctx.memory)
        for text in lessons[:3]:
            ctx.memory.add_lesson(text, tags="self-improve")
        ctx.say(f"[cyan]prompt_memory[/]: added {len(lessons[:3])} lesson(s)")

        def revert() -> None:
            ctx.memory._conn.execute("DELETE FROM lessons WHERE id > ?", (before_max,))
            ctx.memory._conn.commit()

        return AppliedChange(f"added {len(lessons[:3])} lessons", revert)


def _max_lesson_id(memory) -> int:
    row = memory._conn.execute("SELECT COALESCE(MAX(id),0) AS m FROM lessons").fetchone()
    return int(row["m"])
