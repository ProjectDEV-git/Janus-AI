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

import shutil
import tempfile

import psutil

from janus import sandbox
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

    @property
    def tokens_per_task(self) -> float:
        return self.total_tokens / self.n if self.n else 0.0

    def is_better_than(self, other: "Scorecard | None",
                       min_gain: float = 0.05) -> tuple[bool, str]:
        """Keep a change only if success rises, or success holds and tokens/task
        fall by at least `min_gain`. Wall-clock time is reported but never decides:
        it moves with machine load, so it would let noise through as "improvement"."""
        if other is None:
            return True, "no baseline"
        if self.success_rate < other.success_rate:
            return False, f"success dropped {other.success_rate:.2f} -> {self.success_rate:.2f}"
        if self.success_rate > other.success_rate:
            return True, f"success up {other.success_rate:.2f} -> {self.success_rate:.2f}"
        cost = (f"tokens/task {other.tokens_per_task:.0f}->{self.tokens_per_task:.0f}, "
                f"time {other.total_seconds:.0f}->{self.total_seconds:.0f}s")
        if other.tokens_per_task and \
                self.tokens_per_task <= other.tokens_per_task * (1 - min_gain):
            return True, f"same success, cheaper ({cost})"
        return False, f"no improvement beyond the {min_gain:.0%} margin ({cost})"

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
    """Run a task's success check (sandboxed when possible). Supports type 'file_runs'."""
    if check.get("type") == "file_runs":
        target = settings.workspace_abs / check["path"]
        if not target.is_file():
            return False
        snippet = check["snippet"]
        try:
            # -E -s, not -I: -I drops the cwd from sys.path, so the import would always fail.
            proc = sandbox.run([sys.executable, "-E", "-s", "-c", snippet], settings=settings,
                               writable=settings.workspace_abs, timeout=30)
            return proc.returncode == 0 and "OK" in proc.stdout
        except Exception:  # noqa: BLE001
            return False
    return False


def preflight(settings: Settings, trust: bool) -> tuple[bool, str]:
    """The benchmark runs model-written code. Without a sandbox that is only allowed
    under explicit trust; otherwise every code step would be blocked and the
    scorecard would measure nothing."""
    if trust or sandbox.enabled(settings):
        return True, "ok"
    return False, ("the benchmark runs model-written code and no sandbox is available. "
                   f"To sandbox it, {sandbox.install_hint()}; or pass --trust to run it "
                   "unsandboxed on this machine.")


def run_benchmark(trust: bool = False, console=None, settings: Settings | None = None) -> Scorecard:
    s = settings or load_settings()
    s.trust = trust
    # Sample (near-)deterministically so an A/B difference reflects the change, not luck.
    s.temperature = s.bench_temperature
    s.seed = s.bench_seed
    proc = psutil.Process(os.getpid())

    scores: list[TaskScore] = []
    for task in load_tasks():
        for _ in range(max(1, s.bench_repeats)):
            scores.append(_run_task(task, s, console))

    card = Scorecard(scores)
    if console:
        console.print(f"\n[bold]Scorecard[/]: {card.successes}/{card.n} "
                      f"({card.success_rate:.0%}), {card.total_tokens} tokens, "
                      f"{card.total_seconds:.0f}s, peak RSS {proc.memory_info().rss // (1024*1024)} MB")
    return card


def _run_task(task: dict, s: Settings, console=None) -> TaskScore:
    """Run one task in a fresh, empty workspace so leftovers can't pass its check."""
    ws = Path(tempfile.mkdtemp(prefix=f"janus-bench-{task['id']}-"))
    ts = s.model_copy(update={"workspace": ws})
    # Fresh clients per task so token accounting is per-task.
    llm = LLM(ts)
    mem = Memory(ts.db_path)
    try:
        gate = ApprovalGate(ts, prompter=None)  # non-interactive; trust decides risky
        stop = StopController(stopfile=ts.stopfile)
        agent = Agent(ts, llm, build_registry(), mem, gate, stop, reporter=Reporter())
        start = time.monotonic()
        out = agent.run(task["goal"])
        elapsed = time.monotonic() - start
        success = _check(task["check"], ts)
    finally:
        mem.close()
        shutil.rmtree(ws, ignore_errors=True)
    if console:
        mark = "[green]PASS[/]" if success else "[red]FAIL[/]"
        console.print(f"{mark} {task['id']}: {out.status}, {out.tokens} tok, {elapsed:.1f}s")
    return TaskScore(task["id"], success, out.tokens, elapsed, out.status)


def measure_subprocess(trust: bool = False, model: str | None = None) -> Scorecard:
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
    _trust = "--trust" in sys.argv
    _as_json = "--json" in sys.argv
    _card = run_benchmark(trust=_trust, console=None)
    if _as_json:
        print(json.dumps(_card.to_dict()))
    else:
        print(json.dumps(_card.to_dict(), indent=2))
