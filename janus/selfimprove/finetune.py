"""Layer 4 — self-training (LoRA) hook inside the improvement cycle.

The heavy training/export work lives in janus.train.pipeline and is driven by
`janus train` (or `janus improve --train`). As a Strategy this only reports
readiness, since training is not a small reversible micro-change like the other
layers — it produces a whole new model version with its own A/B gate.
"""
from __future__ import annotations

from janus.selfimprove.base import AppliedChange, ImproveContext
from janus.train import trainer as trainer_mod


class FineTuneStrategy:
    name = "finetune"

    def propose(self, ctx: ImproveContext) -> AppliedChange | None:
        ok, msg = trainer_mod.preflight()
        if ok:
            ctx.say("[dim]finetune: trainer ready — run `janus train` to build a new model "
                    "version (gated on an A/B benchmark).[/]")
        else:
            ctx.say(f"[dim]finetune: {msg}[/]")
        return None
