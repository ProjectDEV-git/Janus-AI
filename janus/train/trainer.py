"""LoRA/QLoRA trainer over the base Gemma weights.

Loads the full-precision base model (settings.base_model_id), applies a LoRA
adapter, and fine-tunes on the mined chat dataset using TRL's SFTTrainer with
the tokenizer's Gemma chat template (training on assistant turns). Requires the
`finetune` extra and a CUDA GPU; without either it raises TrainerUnavailable so
callers can degrade gracefully.
"""
from __future__ import annotations

from pathlib import Path


class TrainerUnavailable(RuntimeError):
    """Raised when training deps or a GPU are missing."""


def deps_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import trl  # noqa: F401
        import datasets  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def preflight() -> tuple[bool, str]:
    if not deps_available():
        return False, "training deps missing; install with: pip install -e '.[finetune]'"
    if not gpu_available():
        return False, "no CUDA GPU detected; LoRA training needs a GPU"
    return True, "ready"


def train_lora(dataset_path: Path, out_dir: Path, settings) -> Path:
    """Train a LoRA adapter and save it to out_dir. Returns out_dir."""
    ok, msg = preflight()
    if not ok:
        raise TrainerUnavailable(msg)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model_id, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    peft_cfg = LoraConfig(
        r=settings.lora_r, lora_alpha=settings.lora_alpha, lora_dropout=settings.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    def _format(batch):
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False)
                         for m in batch["messages"]]}

    ds = ds.map(_format, batched=True, remove_columns=ds.column_names)

    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=settings.train_epochs,
        per_device_train_batch_size=settings.train_batch_size,
        gradient_accumulation_steps=settings.train_grad_accum,
        learning_rate=settings.train_lr,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        dataset_text_field="text",
        max_seq_length=2048,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
