"""The efficiency benchmark — Janus's fitness function.

Runs a fixed suite of small tasks and measures, per task: success (an automated
check), tokens used, wall-clock seconds, and peak RSS of the agent process. The
aggregate scorecard is what every self-improvement must beat to be kept.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

from janus.agent import Agent, Reporter, StopController
from janus.approval import ApprovalGate
from janus.config import Settings, load_settings
from janus.llm import LLM
from janus.memory import Memory
from janus.tools import build_registry

TASKS_FILE = Path(__file__).resolve().parent.parent / "benchmark" / "tasks" / "tasks.json"


@dataclass
class TaskScore:
    id: str
    success: bool
    tokens: int
    seconds: float
    status: str


@dataclass
class Scorecard:
    tasks: list[TaskScore]

    @property
    def n(self) -> int:
        return len(self.tasks)

    @property
    def successes(self) -> int:
        return sum(1 for t in self.tasks if t.success)

    @property
    def success_rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(t.tokens for t in self.tasks)

    @property
    def total_seconds(self) -> float:
        return sum(t.seconds for t in self.tasks)

    def is_better_than(self, other: "Scorecard | None") -> tuple[bool, str]:
        """A change is kept only if success does not drop AND cost falls."""
        if other is None:
            return True, "no baseline"
        if self.success_rate < other.success_rate:
            return False, f"success dropped {other.success_rate:.2f} -> {self.success_rate:.2f}"
        cheaper_tokens = self.total_tokens < other.total_tokens
        cheaper_time = self.total_seconds < other.total_seconds
        if self.success_rate > other.success_rate:
            return True, f"success up {other.success_rate:.2f} -> {self.success_rate:.2f}"
        if cheaper_tokens or cheaper_time:
            return True, (f"same success, cheaper (tokens {other.total_tokens}->{self.total_tokens}, "
                          f"time {other.total_seconds:.0f}->{self.total_seconds:.0f}s)")
        return False, "no improvement in success or cost"

    def to_dict(self) -> dict:
        return {
            "success_rate": self.success_rate,
            "successes": self.successes,
            "n": self.n,
            "total_tokens": self.total_tokens,
            "total_seconds": self.total_seconds,
            "tasks": [asdict(t) for t in self.tasks],
        }


def load_tasks() -> list[dict]:
    return json.loads(TASKS_FILE.read_text())


def _check(check: dict, settings: Settings) -> bool:
    """Run a task's success check. Currently supports type 'file_runs'."""
    if check.get("type") == "file_runs":
        target = settings.workspace_abs / check["path"]
        if not target.is_file():
            return False
        snippet = check["snippet"]
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", snippet],
                cwd=str(settings.workspace_abs),
                capture_output=True, text=True, timeout=30,
            )
            return proc.returncode == 0 and "OK" in proc.stdout
        except Exception:  # noqa: BLE001
            return False
    return False


def run_benchmark(trust: bool = True, console=None, settings: Settings | None = None) -> Scorecard:
    s = settings or load_settings()
    s.trust = trust
    s.workspace_abs.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process(os.getpid())

    scores: list[TaskScore] = []
    for task in load_tasks():
        # Fresh clients per task so token accounting is per-task.
        llm = LLM(s)
        reg = build_registry()
        mem = Memory(s.db_path)
        gate = ApprovalGate(s, prompter=None)  # non-interactive; trust decides risky
        stop = StopController(stopfile=s.stopfile)
        agent = Agent(s, llm, reg, mem, gate, stop, reporter=Reporter())

        start = time.monotonic()
        out = agent.run(task["goal"])
        elapsed = time.monotonic() - start
        success = _check(task["check"], s)
        scores.append(TaskScore(task["id"], success, out.tokens, elapsed, out.status))
        mem.close()
        if console:
            mark = "[green]PASS[/]" if success else "[red]FAIL[/]"
            console.print(f"{mark} {task['id']}: {out.status}, {out.tokens} tok, {elapsed:.1f}s")

    card = Scorecard(scores)
    if console:
        console.print(f"\n[bold]Scorecard[/]: {card.successes}/{card.n} "
                      f"({card.success_rate:.0%}), {card.total_tokens} tokens, "
                      f"{card.total_seconds:.0f}s, peak RSS {proc.memory_info().rss // (1024*1024)} MB")
    return card


def measure_subprocess(trust: bool = True, model: str | None = None) -> Scorecard:
    """Run the benchmark in a fresh subprocess and parse its JSON scorecard.

    Used by self-improvement so a change to Janus's own source/tools/memory is
    measured against freshly-imported code, not the already-loaded modules. Pass
    `model` to A/B a specific Ollama tag (e.g. a newly trained janus:vN).
    """
    env = dict(os.environ)
    if model:
        env["JANUS_MODEL"] = model
    proc = subprocess.run(
        [sys.executable, "-m", "janus.benchmark", "--json", ("--trust" if trust else "--no-trust")],
        capture_output=True, text=True, timeout=60 * 60, env=env,
    )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
    data = json.loads(line)
    tasks = [TaskScore(**t) for t in data.get("tasks", [])]
    return Scorecard(tasks)


if __name__ == "__main__":
    _trust = "--no-trust" not in sys.argv
    _as_json = "--json" in sys.argv
    _card = run_benchmark(trust=_trust, console=None)
    if _as_json:
        print(json.dumps(_card.to_dict()))
    else:
        print(json.dumps(_card.to_dict(), indent=2))
