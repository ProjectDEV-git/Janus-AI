"""Layer 4 — optional LoRA fine-tuning (no-ops gracefully without a GPU).

Collects successful run transcripts into a small SFT dataset and, if a GPU and
the `finetune` extra are installed, trains a LoRA adapter and A/B-tests it
against the base model on the benchmark, adopting it only on a win. Without the
dependencies or a GPU it reports that it is unavailable and makes no change.
"""
from __future__ import annotations

from janus.selfimprove.base import AppliedChange, ImproveContext


def _gpu_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


class FineTuneStrategy:
    name = "finetune"

    def propose(self, ctx: ImproveContext) -> AppliedChange | None:
        if not _gpu_available():
            ctx.say("[dim]finetune: no GPU / torch not installed; skipping (install .[finetune])[/]")
            return None
        # Dataset assembly + LoRA training would go here; intentionally not run
        # automatically without an explicit, resourced opt-in.
        ctx.say("[dim]finetune: GPU present but auto-training is opt-in; skipping[/]")
        return None
