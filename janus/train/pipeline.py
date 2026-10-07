"""Orchestrate one self-training cycle, producing a new Janus model version.

Steps: mine dataset -> (preflight) -> LoRA train -> merge -> GGUF -> ollama
create -> register in the lineage -> A/B benchmark vs the current model -> adopt
only on a win, else keep the version on record but do not adopt. Degrades
gracefully: with no GPU/deps it still builds the dataset and reports the plan.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from janus.benchmark import measure_subprocess, preflight
from janus.train import gguf as gguf_mod
from janus.train.gguf import ollama_create, write_modelfile
from janus.train import trainer as trainer_mod
from janus.train.dataset import build_dataset
from janus.train.registry import ModelRegistry


@dataclass
class TrainOutcome:
    status: str          # adopted | built_not_adopted | skipped | dataset_only | error
    detail: str
    tag: str | None = None


def _hash_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_training_cycle(settings, memory, *, adopt: bool = False, console=None) -> TrainOutcome:
    def say(msg: str) -> None:
        if console is not None:
            console.print(msg)

    # 1. Mine the dataset from winning runs.
    stats = build_dataset(memory, settings)
    say(f"[cyan]dataset[/]: {stats.examples} examples from {stats.runs_used} runs "
        f"({stats.assistant_turns} assistant turns) -> {stats.path}")
    if stats.examples < settings.train_min_examples:
        return TrainOutcome(
            "skipped",
            f"only {stats.examples} training examples (< train_min_examples="
            f"{settings.train_min_examples}); run more successful tasks first.",
        )

    # 2. Preflight the trainer; without a GPU/deps, stop after the dataset.
    ok, msg = trainer_mod.preflight()
    if not ok:
        return TrainOutcome("dataset_only",
                            f"dataset ready at {stats.path}, but cannot train: {msg}")

    registry = ModelRegistry(settings.db_path)
    version = registry.next_version()
    tag = f"{settings.model_prefix}:v{version}"
    parent_tag = registry.current_tag(default=settings.model)
    adapter_dir = settings.adapters_dir / f"v{version}"
    dataset_hash = _hash_file(stats.path)

    try:
        # 3. Train the LoRA adapter.
        say(f"[cyan]train[/]: LoRA on {settings.base_model_id} -> {adapter_dir}")
        trainer_mod.train_lora(stats.path, adapter_dir, settings)

        # 4. Merge + convert + quantize into a GGUF.
        merged_dir = settings.adapters_dir / f"v{version}_merged"
        gguf_path = settings.adapters_dir / f"{settings.model_prefix}-v{version}.{settings.gguf_quant}.gguf"
        say("[cyan]export[/]: merging adapter and converting to GGUF")
        gguf_mod.merge_adapter(adapter_dir, merged_dir, settings)
        gguf_mod.convert_to_gguf(merged_dir, gguf_path, settings)

        # 5. Register with Ollama.
        modelfile = gguf_mod.write_modelfile(gguf_path, tag, settings, parent_tag)
        created, detail = gguf_mod.ollama_create(tag, modelfile)
        if not created:
            registry.register(version=version, tag=tag, parent_tag=parent_tag,
                              base_model=settings.base_model_id, dataset_hash=dataset_hash,
                              adapter_dir=str(adapter_dir), status="export_failed")
            registry.close()
            return TrainOutcome("error", f"ollama create failed: {detail}", tag)

        registry.register(version=version, tag=tag, parent_tag=parent_tag,
                          base_model=settings.base_model_id, dataset_hash=dataset_hash,
                          adapter_dir=str(adapter_dir), status="built")
    except Exception as e:  # noqa: BLE001
        registry.close()
        return TrainOutcome("error", f"training/export failed: {e}", tag)

    # 6. A/B benchmark: candidate vs current, adopt only on a measurable win.
    ok, msg = preflight(settings, settings.trust)
    if not ok:
        registry.close()
        return TrainOutcome("built_not_adopted", f"{tag} built but not benchmarked: {msg}", tag)
    say(f"[cyan]A/B[/]: benchmarking {tag} vs current ({parent_tag})")
    base_card = measure_subprocess(trust=settings.trust, model=parent_tag)
    cand_card = measure_subprocess(trust=settings.trust, model=tag)
    registry.set_scorecard(tag, cand_card.to_dict())
    better, why = cand_card.is_better_than(base_card, settings.bench_min_gain)

    if better and adopt:
        registry.adopt(tag)
        registry.close()
        return TrainOutcome("adopted",
                            f"{tag} beat {parent_tag} ({why}); adopted as the active model.", tag)
    if better:
        registry.close()
        return TrainOutcome("built_not_adopted",
                            f"{tag} beat current ({why}); not adopted (pass --adopt or "
                            f"`janus model use {tag}`).", tag)
    registry.set_status(tag, "rejected")
    registry.close()
    return TrainOutcome("built_not_adopted", f"{tag} did not beat current ({why}); kept on record.",
                        tag)


def register_external_gguf(settings, *, gguf_path, adopt=False, parent_tag=None,
                           console=None, create=ollama_create) -> TrainOutcome:
    """Register a GGUF trained elsewhere (e.g. Google Colab) into the lineage.

    Writes a Modelfile, creates the Ollama tag, records the version, and
    optionally adopts it. `create` is injectable for testing. This closes the
    no-local-GPU loop: train on Colab, download the GGUF, import it here.
    """
    def say(msg: str) -> None:
        if console is not None:
            console.print(msg)

    gguf_path = __import__("pathlib").Path(gguf_path)
    if not gguf_path.is_file():
        return TrainOutcome("error", f"GGUF not found: {gguf_path}")

    registry = ModelRegistry(settings.db_path)
    version = registry.next_version()
    tag = f"{settings.model_prefix}:v{version}"
    parent = parent_tag or registry.current_tag(default=settings.model)

    modelfile = write_modelfile(gguf_path, tag, settings, parent)
    ok, detail = create(tag, modelfile)
    if not ok:
        registry.register(version=version, tag=tag, parent_tag=parent,
                          base_model=settings.base_model_id, dataset_hash="(external)",
                          adapter_dir="(external)", status="export_failed")
        registry.close()
        return TrainOutcome("error", f"ollama create failed: {detail}", tag)

    registry.register(version=version, tag=tag, parent_tag=parent,
                      base_model=settings.base_model_id, dataset_hash="(external)",
                      adapter_dir="(external)", status="built")
    say(f"[cyan]import[/]: registered {tag} (from {gguf_path.name})")
    if adopt:
        registry.adopt(tag)
        registry.close()
        return TrainOutcome("adopted", f"{tag} imported and adopted as the active model.", tag)
    registry.close()
    return TrainOutcome("built_not_adopted",
                        f"{tag} imported; run `janus model use {tag}` to activate it.", tag)
