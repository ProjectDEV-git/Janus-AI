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
    s.model = _active_model(s)
    s.workspace_abs.mkdir(parents=True, exist_ok=True)
    s.transcripts_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(s)
    reg = build_registry()
    mem = Memory(s.db_path)
    prompter = _make_prompter() if interactive else None
    gate = ApprovalGate(s, prompter=prompter)
    stop = StopController(stopfile=s.stopfile)
    return s, llm, reg, mem, gate, stop


def _active_model(s) -> str:
    """Prefer the adopted model from the lineage registry over the config default."""
    try:
        from janus.train.registry import ModelRegistry
        reg = ModelRegistry(s.db_path)
        tag = reg.current_tag(default=s.model)
        reg.close()
        return tag
    except Exception:  # noqa: BLE001
        return s.model


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
def bench(trust: bool = typer.Option(False, "--trust/--no-trust",
                                     help="Allow risky actions (needed to run code without a sandbox).")):
    """Run the efficiency benchmark and print a scorecard."""
    from janus.benchmark import preflight, run_benchmark
    from janus.config import load_settings
    ok, msg = preflight(load_settings(), trust)
    if not ok:
        console.print(f"[red]Cannot benchmark:[/] {msg}")
        raise typer.Exit(1)
    run_benchmark(trust=trust, console=console)


@app.command()
def improve(
    rounds: int = typer.Option(1, help="How many improvement rounds to attempt."),
    trust: bool = typer.Option(False, "--trust", help="Auto-approve risky self-edits."),
    train: bool = typer.Option(False, "--train", help="Also run a self-training cycle at the end."),
    adopt: bool = typer.Option(False, "--adopt", help="Adopt a winning trained model automatically."),
):
    """Run the self-improvement cycle (prompts/tools/code), optionally self-training too."""
    from janus.selfimprove import run_improve
    run_improve(rounds=rounds, trust=trust, train=train, adopt=adopt, console=console)


@app.command()
def train(
    adopt: bool = typer.Option(False, "--adopt", help="Adopt the new model if it wins the A/B."),
    trust: bool = typer.Option(False, "--trust",
                               help="Allow the A/B benchmark to run code without a sandbox."),
):
    """Train a new Janus model version (LoRA on the base weights) from winning runs."""
    from janus.config import load_settings
    from janus.memory import Memory
    from janus.train.pipeline import run_training_cycle

    s = load_settings()
    s.trust = trust
    s.model = _active_model(s)
    ok, msg = LLM(s).ping()
    if not ok:
        console.print(f"[yellow]note:[/] {msg} (A/B benchmarking needs the model running)")
    mem = Memory(s.db_path)
    outcome = run_training_cycle(s, mem, adopt=adopt, console=console)
    mem.close()
    color = {"adopted": "green", "built_not_adopted": "cyan",
             "dataset_only": "yellow", "skipped": "yellow", "error": "red"}.get(outcome.status, "white")
    console.print(Panel(f"status: [bold]{outcome.status}[/]\n{outcome.detail}",
                        title=f"train {outcome.tag or ''}", border_style=color))


@app.command()
def dataset():
    """Build the training dataset from winning runs (for training on Colab/elsewhere)."""
    from janus.config import load_settings
    from janus.memory import Memory
    from janus.train.dataset import build_dataset

    s = load_settings()
    mem = Memory(s.db_path)
    stats = build_dataset(mem, s)
    mem.close()
    console.print(Panel(
        f"examples: [bold]{stats.examples}[/] (from {stats.runs_used} runs, "
        f"{stats.assistant_turns} assistant turns)\npath: {stats.path}\n\n"
        f"Upload this file to the Colab notebook (notebooks/janus_train_colab.ipynb) "
        f"to train without a local GPU.",
        title="dataset", border_style="cyan"))


model_app = typer.Typer(help="Inspect and switch Janus model versions.")
app.add_typer(model_app, name="model")


@model_app.command("list")
def model_list():
    """List the Janus model lineage."""
    from janus.train.registry import ModelRegistry
    s = load_settings()
    reg = ModelRegistry(s.db_path)
    rows = reg.list()
    if not rows:
        console.print(f"No trained versions yet. Active model: [bold]{_active_model(s)}[/]")
        reg.close()
        return
    t = Table(title="Janus model lineage")
    for c in ("version", "tag", "parent", "status", "adopted", "success"):
        t.add_column(c)
    for r in rows:
        sr = f"{r.scorecard['success_rate']:.0%}" if r.scorecard else "-"
        t.add_row(str(r.version), r.tag, r.parent_tag or "(base)", r.status,
                  "yes" if r.adopted else "", sr)
    console.print(t)
    reg.close()


@model_app.command("use")
def model_use(tag: str = typer.Argument(..., help="Model tag to adopt, e.g. janus:v2.")):
    """Adopt a specific model version as the active model."""
    from janus.train.registry import ModelRegistry
    s = load_settings()
    reg = ModelRegistry(s.db_path)
    reg.adopt(tag)
    reg.close()
    console.print(f"[green]Active model set to[/] {tag}")


@model_app.command("import")
def model_import(
    gguf: str = typer.Argument(..., help="Path to a .gguf trained elsewhere (e.g. from Colab)."),
    adopt: bool = typer.Option(False, "--adopt", help="Adopt it as the active model."),
    parent: str = typer.Option(None, "--parent", help="Parent tag for the lineage."),
):
    """Register a GGUF trained on Colab (or any machine) into the lineage + Ollama."""
    from janus.config import load_settings
    from janus.train.pipeline import register_external_gguf

    s = load_settings()
    outcome = register_external_gguf(s, gguf_path=gguf, adopt=adopt, parent_tag=parent,
                                     console=console)
    color = {"adopted": "green", "built_not_adopted": "cyan", "error": "red"}.get(
        outcome.status, "white")
    console.print(Panel(f"status: [bold]{outcome.status}[/]\n{outcome.detail}",
                        title=f"import {outcome.tag or ''}", border_style=color))


if __name__ == "__main__":
    app()
