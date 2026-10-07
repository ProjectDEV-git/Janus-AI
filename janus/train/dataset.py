"""Mine the audit log into a supervised fine-tuning dataset.

Each successful (status='finished') run is reconstructed into a chat trajectory:
system prompt + goal, then the agent's emitted JSON actions as assistant turns
interleaved with observations as user turns. Training on these assistant turns
is behavioral self-distillation: Janus learns to reproduce what worked.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from janus.agent import SYSTEM_PROMPT
from janus.memory import Memory
from janus.tools import build_registry


@dataclass
class DatasetStats:
    path: Path
    examples: int
    runs_used: int
    assistant_turns: int


def _system_prompt() -> str:
    return SYSTEM_PROMPT.format(catalog=build_registry().catalog())


def _reconstruct(events: list) -> list[dict] | None:
    """Rebuild a [{'role','content'}, ...] trajectory from one run's events."""
    by_iter: dict[int, dict] = {}
    for e in events:
        payload = json.loads(e["payload"])
        i = payload.get("iter")
        if i is None:
            continue
        slot = by_iter.setdefault(i, {})
        if e["kind"] == "thought":
            slot["thought"] = payload.get("text", "")
        elif e["kind"] == "action":
            slot["action"] = {"tool": payload.get("tool"), "args": payload.get("args", {})}
        elif e["kind"] == "observation":
            slot["observation"] = {"ok": payload.get("ok"), "output": payload.get("output", "")}
        elif e["kind"] == "decision" and payload.get("finish"):
            slot["finish"] = {"done": True, "evidence": payload.get("evidence", "")}

    messages: list[dict] = []
    has_assistant = False
    for i in sorted(by_iter):
        slot = by_iter[i]
        thought = slot.get("thought", "")
        if "finish" in slot:
            messages.append({"role": "assistant",
                             "content": json.dumps({"thought": thought, "finish": slot["finish"]})})
            has_assistant = True
            break
        if "action" in slot:
            messages.append({"role": "assistant",
                             "content": json.dumps({"thought": thought, "action": slot["action"]})})
            has_assistant = True
            obs = slot.get("observation")
            if obs is not None:
                messages.append({"role": "user",
                                 "content": f"OBSERVATION (ok={obs['ok']}):\n{obs['output']}\n\n"
                                            "Emit your next JSON object."})
    return messages if has_assistant else None


def build_dataset(memory: Memory, settings, *, limit_runs: int = 500) -> DatasetStats:
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.datasets_dir / "sft.jsonl"
    system = _system_prompt()

    runs = memory.recent_runs(limit=limit_runs)
    examples = 0
    runs_used = 0
    assistant_turns = 0
    with out_path.open("w") as fh:
        for r in runs:
            if r["status"] != "finished":
                continue
            traj = _reconstruct(memory.events_for(r["id"]))
            if not traj:
                continue
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"GOAL: {r['goal']}\n\nBegin. Emit your first JSON action."},
                *traj,
            ]
            fh.write(json.dumps({"messages": msgs}) + "\n")
            examples += 1
            runs_used += 1
            assistant_turns += sum(1 for m in traj if m["role"] == "assistant")

    return DatasetStats(out_path, examples, runs_used, assistant_turns)
