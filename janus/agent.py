"""The core agent loop: Plan -> Act -> Observe -> Reflect.

The agent asks the model for one action at a time as strict JSON, routes each
action through the approval gate, records the observation, and repeats until the
model declares the goal finished (with evidence) or a stop/budget limit is hit.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from janus.approval import ApprovalGate, Decision
from janus.config import Settings
from janus.llm import LLM
from janus.memory import Memory
from janus.tools import Registry

SYSTEM_PROMPT = """You are Janus, an autonomous agent working toward a goal.

You act by emitting ONE JSON object per turn and nothing else. Two shapes:

1. Take an action:
   {{"thought": "<brief reasoning>", "action": {{"tool": "<name>", "args": {{...}}}}}}

2. Finish, only when the goal is genuinely achieved and you have evidence:
   {{"thought": "<why it is done>", "finish": {{"done": true, "evidence": "<what proves it>"}}}}

Rules:
- Use tools to gather evidence; do not claim success you have not verified.
- Prefer small, checkable steps. After producing an artifact, test it.
- Keep working until the goal is verifiably met. Do not give up early.
- Output ONLY the JSON object. No prose, no markdown fences.

Available tools:
{catalog}
"""


@dataclass
class StopController:
    """Lets the loop be halted via an in-process event or a stopfile on disk."""
    stopfile: "object"  # Path
    _event: threading.Event = field(default_factory=threading.Event)

    def request_stop(self) -> None:
        self._event.set()

    def should_stop(self) -> bool:
        if self._event.is_set():
            return True
        try:
            return self.stopfile.exists()
        except OSError:
            return False

    def clear(self) -> None:
        self._event.clear()
        try:
            if self.stopfile.exists():
                self.stopfile.unlink()
        except OSError:
            pass


@dataclass
class Outcome:
    run_id: int
    status: str          # finished | stopped | budget | error | blocked_out
    iterations: int
    tokens: int
    evidence: str = ""
    error: str = ""


# Reporter callbacks for UI; all optional.
@dataclass
class Reporter:
    on_thought: Callable[[int, str], None] = lambda i, t: None
    on_action: Callable[[int, str, dict], None] = lambda i, n, a: None
    on_gate: Callable[[int, str, str], None] = lambda i, d, r: None
    on_observation: Callable[[int, bool, str], None] = lambda i, ok, o: None
    on_status: Callable[[str], None] = lambda s: None


class Agent:
    def __init__(
        self,
        settings: Settings,
        llm: LLM,
        registry: Registry,
        memory: Memory,
        gate: ApprovalGate,
        stop: StopController,
        reporter: Reporter | None = None,
    ) -> None:
        self.s = settings
        self.llm = llm
        self.reg = registry
        self.mem = memory
        self.gate = gate
        self.stop = stop
        self.report = reporter or Reporter()

    def _system(self) -> str:
        return SYSTEM_PROMPT.format(catalog=self.reg.catalog())

    def _seed_messages(self, goal: str) -> list[dict]:
        lessons = self.mem.recall_lessons(goal)
        lesson_block = ""
        if lessons:
            lesson_block = "\n\nLessons from past runs:\n" + "\n".join(f"- {x}" for x in lessons)
        return [
            {"role": "system", "content": self._system()},
            {"role": "user", "content": f"GOAL: {goal}{lesson_block}\n\nBegin. Emit your first JSON action."},
        ]

    def run(self, goal: str) -> Outcome:
        run_id = self.mem.start_run(goal)
        messages = self._seed_messages(goal)
        start = time.monotonic()
        i = 0
        self.report.on_status(f"run #{run_id} started")

        try:
            while i < self.s.max_iterations:
                if self.stop.should_stop():
                    return self._end(run_id, "stopped", i, "")
                if time.monotonic() - start > self.s.max_seconds:
                    return self._end(run_id, "budget", i, "", note="time budget exceeded")
                if self.llm.tokens_used > self.s.max_tokens:
                    return self._end(run_id, "budget", i, "", note="token budget exceeded")

                i += 1
                try:
                    step = self.llm.chat_json(messages)
                except Exception as e:  # noqa: BLE001
                    self.mem.log_event(run_id, "note", {"iter": i, "error": str(e)})
                    messages.append({"role": "user",
                                     "content": f"Your output was not valid JSON ({e}). "
                                                "Re-emit exactly one JSON object."})
                    continue

                # Keep the model's own turn in the history so it can see what it already did.
                messages.append({"role": "assistant", "content": json.dumps(step)})
                thought = str(step.get("thought", ""))
                self.report.on_thought(i, thought)
                self.mem.log_event(run_id, "thought", {"iter": i, "text": thought})

                if "finish" in step:
                    evidence = str((step.get("finish") or {}).get("evidence", ""))
                    self.mem.log_event(run_id, "decision", {"iter": i, "finish": True,
                                                            "evidence": evidence})
                    return self._end(run_id, "finished", i, evidence)

                action = step.get("action") or {}
                tool_name = str(action.get("tool", ""))
                args = action.get("args") or {}
                self.report.on_action(i, tool_name, args)
                self.mem.log_event(run_id, "action", {"iter": i, "tool": tool_name, "args": args})

                tool = self.reg.get(tool_name)
                if tool is None:
                    obs = f"No such tool '{tool_name}'. Available: {[t.name for t in self.reg.all()]}"
                    self._observe(run_id, messages, i, False, obs)
                    continue

                try:
                    tool.validate(args)
                except Exception as e:  # noqa: BLE001
                    self._observe(run_id, messages, i, False, f"invalid args for {tool_name}: {e}")
                    continue

                gate_res = self.gate.decide(tool, args)
                self.report.on_gate(i, gate_res.decision.value, gate_res.reason)
                self.mem.log_event(run_id, "decision", {"iter": i, "gate": gate_res.decision.value,
                                                        "risk": gate_res.risk.value,
                                                        "reason": gate_res.reason,
                                                        "policy": gate_res.policy})
                if gate_res.decision is Decision.BLOCKED:
                    self._observe(run_id, messages, i, False,
                                  f"BLOCKED by approval gate: {gate_res.reason}. "
                                  "Choose a different, safer approach.")
                    continue

                result = tool.run(args, settings=self.s)
                self._observe(run_id, messages, i, result.ok, result.output)

            return self._end(run_id, "budget", i, "", note="iteration budget exceeded")
        except Exception as e:  # noqa: BLE001
            self.mem.log_event(run_id, "note", {"fatal": str(e)})
            return self._end(run_id, "error", i, "", err=str(e))

    def _observe(self, run_id: int, messages: list[dict], i: int, ok: bool, output: str) -> None:
        self.report.on_observation(i, ok, output)
        self.mem.log_event(run_id, "observation", {"iter": i, "ok": ok, "output": output})
        messages.append({"role": "user",
                         "content": f"OBSERVATION (ok={ok}):\n{output}\n\nEmit your next JSON object."})

    def _end(self, run_id: int, status: str, iters: int, evidence: str,
             note: str = "", err: str = "") -> Outcome:
        if note:
            self.mem.log_event(run_id, "note", {"status": status, "note": note})
        self.mem.finish_run(run_id, status=status, iterations=iters,
                            tokens=self.llm.tokens_used, result=evidence or note or err)
        self.report.on_status(f"run #{run_id} {status} after {iters} iters, "
                              f"{self.llm.tokens_used} tokens")
        return Outcome(run_id, status, iters, self.llm.tokens_used, evidence, err)
