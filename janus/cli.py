"""Janus command-line interface."""
from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from janus.agent import Agent, Reporter, StopController
from janus.approval import ApprovalGate
from janus.config import load_settings
from janus.llm import LLM
from janus.memory import Memory
from janus.tools import build_registry

app = typer.Typer(add_completion=False, help="Janus — a local, self-improving agent.")
console = Console()


def _build(trust: bool | None = None, interactive: bool = True):
    s = load_settings()
    if trust is not None:
        s.trust = trust
    s.workspace_abs.mkdir(parents=True, exist_ok=True)
    s.transcripts_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(s)
    reg = build_registry()
    mem = Memory(s.db_path)
    prompter = _make_prompter() if interactive else None
    gate = ApprovalGate(s, prompter=prompter)
    stop = StopController(stopfile=s.stopfile)
    return s, llm, reg, mem, gate, stop


def _make_prompter():
    def prompt(tool_name: str, args: dict, reason: str) -> bool:
        console.print(Panel(
            f"[bold yellow]Approval needed[/]\n"
            f"tool: [bold]{tool_name}[/]\nreason: {reason}\nargs: {json.dumps(args)[:400]}",
            border_style="yellow"))
        return Confirm.ask("Allow this action?", default=False)
    return prompt


def _reporter() -> Reporter:
    return Reporter(
        on_thought=lambda i, t: console.print(f"[dim]#{i}[/] [cyan]think:[/] {t}"),
        on_action=lambda i, n, a: console.print(f"[dim]#{i}[/] [magenta]act:[/] {n} {json.dumps(a)[:200]}"),
        on_gate=lambda i, d, r: console.print(f"[dim]#{i}[/] [blue]gate:[/] {d} ({r})"),
        on_observation=lambda i, ok, o: console.print(
            f"[dim]#{i}[/] [{'green' if ok else 'red'}]obs:[/] {o[:500]}"),
        on_status=lambda msg: console.print(f"[bold]{msg}[/]"),
    )


@app.command()
def config():
    """Show settings and check Ollama connectivity."""
    s = load_settings()
    t = Table(title="Janus config")
    t.add_column("key"); t.add_column("value")
    for k in ("model", "ollama_host", "workspace_abs", "max_iterations",
              "max_seconds", "max_tokens", "auto_approve_safe", "trust"):
        t.add_row(k, str(getattr(s, k)))
    console.print(t)
    ok, msg = LLM(s).ping()
    console.print(f"[{'green' if ok else 'red'}]connectivity:[/] {msg}")


@app.command()
def task(
    goal: str = typer.Argument(..., help="The goal to pursue."),
    trust: bool = typer.Option(False, "--trust", help="Auto-approve risky actions (watch it!)."),
):
    """Run a goal to completion (or until stopped / budget hit)."""
    s, llm, reg, mem, gate, stop = _build(trust=trust)
    stop.clear()
    ok, msg = llm.ping()
    if not ok:
        console.print(f"[red]Cannot start:[/] {msg}")
        raise typer.Exit(1)
    agent = Agent(s, llm, reg, mem, gate, stop, reporter=_reporter())
    out = agent.run(goal)
    console.print(Panel(f"status: [bold]{out.status}[/]\nevidence: {out.evidence or out.error}",
                        title=f"run #{out.run_id}", border_style="green" if out.status == "finished" else "red"))
    mem.close()


@app.command()
def stop():
    """Signal a running Janus to halt at its next step."""
    s = load_settings()
    s.stopfile.write_text("stop")
    console.print(f"[yellow]Stop requested[/] (wrote {s.stopfile})")


@app.command()
def log(task_id: int = typer.Option(None, "--task", help="Show events for one run.")):
    """Inspect the audit trail."""
    s = load_settings()
    mem = Memory(s.db_path)
    if task_id is None:
        t = Table(title="Recent runs")
        for c in ("id", "status", "iters", "tokens", "goal"):
            t.add_column(c)
        for r in mem.recent_runs():
            t.add_row(str(r["id"]), r["status"], str(r["iterations"]),
                      str(r["tokens"]), (r["goal"] or "")[:60])
        console.print(t)
    else:
        for e in mem.events_for(task_id):
            console.print(f"[dim]{e['kind']}[/] {e['payload'][:300]}")
    mem.close()


@app.command()
def replay(task_id: int = typer.Argument(..., help="Run id to replay.")):
    """Replay a past run's thoughts and actions in order."""
    s = load_settings()
    mem = Memory(s.db_path)
    run = mem.get_run(task_id)
    if run is None:
        console.print(f"[red]no such run {task_id}[/]"); raise typer.Exit(1)
    console.print(Panel(f"goal: {run['goal']}\nstatus: {run['status']}", title=f"run #{task_id}"))
    for e in mem.events_for(task_id):
        payload = json.loads(e["payload"])
        console.print(f"[cyan]{e['kind']}[/] {json.dumps(payload)[:300]}")
    mem.close()


@app.command()
def bench(trust: bool = typer.Option(True, "--trust/--no-trust",
                                     help="Benchmark runs trusted by default.")):
    """Run the efficiency benchmark and print a scorecard."""
    from janus.benchmark import run_benchmark
    run_benchmark(trust=trust, console=console)


@app.command()
def improve(
    rounds: int = typer.Option(1, help="How many improvement rounds to attempt."),
    trust: bool = typer.Option(False, "--trust", help="Auto-approve risky self-edits."),
):
    """Run the self-improvement cycle against the benchmark."""
    from janus.selfimprove import run_improve
    run_improve(rounds=rounds, trust=trust, console=console)


if __name__ == "__main__":
    app()
