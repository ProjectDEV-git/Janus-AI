"""Self-improvement orchestration.

run_improve measures a baseline benchmark, then for each round runs the
strategies in order (cheapest first). Each strategy applies one reversible
change; the benchmark is re-measured in a subprocess, and the change is kept
only if the new scorecard beats the baseline, otherwise reverted.
"""
from __future__ import annotations

from janus.approval import ApprovalGate
from janus.benchmark import measure_subprocess, preflight
from janus.config import load_settings
from janus.llm import LLM
from janus.memory import Memory
from janus.selfimprove.base import ImproveContext
from janus.selfimprove.code_patcher import CodePatcherStrategy
from janus.selfimprove.finetune import FineTuneStrategy
from janus.selfimprove.prompt_memory import PromptMemoryStrategy
from janus.selfimprove.tool_writer import ToolWriterStrategy

__all__ = ["run_improve"]


def run_improve(rounds: int = 1, trust: bool = False, train: bool = False,
                adopt: bool = False, console=None) -> None:
    s = load_settings()
    s.trust = trust
    s.workspace_abs.mkdir(parents=True, exist_ok=True)
    llm = LLM(s)
    mem = Memory(s.db_path)

    ok, msg = llm.ping()
    if not ok:
        if console:
            console.print(f"[red]Cannot improve:[/] {msg}")
        return

    # Interactive prompter for risky self-edits unless trusting.
    prompter = None
    if console is not None and not trust:
        from rich.prompt import Confirm

        def prompter(tool_name, args, reason):  # noqa: ANN001
            console.print(f"[yellow]Self-edit approval[/] {tool_name}: {reason}")
            for key, val in args.items():
                console.print(f"[bold]{key}[/]:")
                console.print(str(val), markup=False, highlight=False)
            return Confirm.ask("Allow this self-edit?", default=False)

    gate = ApprovalGate(s, prompter=prompter)

    ok, msg = preflight(s, trust)
    if not ok:
        if console:
            console.print(f"[red]Cannot improve:[/] {msg}")
        return

    strategies = [
        PromptMemoryStrategy(),
        ToolWriterStrategy(gate=gate),
        CodePatcherStrategy(gate=gate),
        FineTuneStrategy(),
    ]

    if console:
        console.print("[bold]Measuring baseline...[/]")
    baseline = measure_subprocess(trust=trust)
    if console:
        console.print(f"baseline: {baseline.success_rate:.0%} success, "
                      f"{baseline.total_tokens} tokens, {baseline.total_seconds:.0f}s")

    ctx = ImproveContext(settings=s, llm=llm, memory=mem, console=console)

    for rnd in range(1, rounds + 1):
        if console:
            console.print(f"\n[bold]=== Round {rnd}/{rounds} ===[/]")
        for strat in strategies:
            change = strat.propose(ctx)
            if change is None:
                continue
            if console:
                console.print(f"[bold]Testing[/] {strat.name}: {change.description}")
            candidate = measure_subprocess(trust=trust)
            better, why = candidate.is_better_than(baseline, s.bench_min_gain)
            if better:
                change.keep()
                baseline = candidate
                if console:
                    console.print(f"[green]KEEP[/] {strat.name}: {why}")
            else:
                change.revert()
                if console:
                    console.print(f"[red]REVERT[/] {strat.name}: {why}")

    if train:
        if console:
            console.print("\n[bold]=== Self-training ===[/]")
        from janus.train.pipeline import run_training_cycle
        # Adopting a new model is a risky action; gate it unless trusting.
        do_adopt = adopt
        if adopt and not trust and gate.prompter is not None:
            do_adopt = gate.prompter("adopt_model", {"adopt": True},
                                     "adopt a newly trained model as the active model")
        outcome = run_training_cycle(s, mem, adopt=do_adopt, console=console)
        if console:
            console.print(f"[bold]train:[/] {outcome.status} — {outcome.detail}")

    mem.close()
