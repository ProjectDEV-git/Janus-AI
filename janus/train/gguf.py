"""Merge a LoRA adapter into the base weights, export to GGUF, register in Ollama.

This is what turns a trained adapter into a runnable Janus model version:
  merge (peft) -> convert to GGUF (llama.cpp) -> quantize -> `ollama create`.

The Modelfile writing and the `ollama create` invocation are pure/shell steps
and are unit-tested; merge/convert need the finetune extra + llama.cpp present.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


class ExportUnavailable(RuntimeError):
    pass


def write_modelfile(gguf_path: Path, tag: str, settings, parent_tag: str | None) -> Path:
    """Write an Ollama Modelfile pointing at the GGUF. Returns its path. Pure."""
    mf = gguf_path.parent / "Modelfile"
    lineage = f"# lineage: base={settings.base_model_id} parent={parent_tag or '(base)'} tag={tag}\n"
    body = (
        lineage
        + f"FROM {gguf_path.name}\n"
        + f"PARAMETER temperature {settings.temperature}\n"
    )
    mf.write_text(body)
    return mf


def _find_llama_cpp(settings) -> Path | None:
    if settings.llama_cpp_dir:
        p = Path(settings.llama_cpp_dir)
        return p if p.exists() else None
    for cand in (Path.home() / "llama.cpp", Path("/opt/llama.cpp"), Path("./llama.cpp")):
        if cand.exists():
            return cand
    return None


def merge_adapter(adapter_dir: Path, out_dir: Path, settings) -> Path:
    """Merge LoRA weights into the base model; save merged HF weights to out_dir."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # noqa: BLE001
        raise ExportUnavailable(f"merge needs the finetune extra: {e}") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(
        settings.base_model_id, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(settings.base_model_id).save_pretrained(str(out_dir))
    return out_dir


def convert_to_gguf(merged_dir: Path, gguf_path: Path, settings) -> Path:
    """Convert merged HF weights to a quantized GGUF via llama.cpp tooling."""
    llama = _find_llama_cpp(settings)
    if llama is None:
        raise ExportUnavailable(
            "llama.cpp not found; set llama_cpp_dir in janus.toml to a built checkout")
    convert = llama / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise ExportUnavailable(f"convert_hf_to_gguf.py not found under {llama}")

    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    f16 = gguf_path.with_suffix(".f16.gguf")
    subprocess.run(
        [sys.executable, str(convert), str(merged_dir), "--outfile", str(f16), "--outtype", "f16"],
        check=True,
    )
    quant_bin = _find_quantize_bin(llama)
    if quant_bin is None:
        # No quantize binary: ship the f16 GGUF as-is.
        shutil.move(str(f16), str(gguf_path))
        return gguf_path
    subprocess.run([str(quant_bin), str(f16), str(gguf_path), settings.gguf_quant], check=True)
    f16.unlink(missing_ok=True)
    return gguf_path


def _find_quantize_bin(llama: Path) -> Path | None:
    for name in ("llama-quantize", "quantize"):
        for sub in ("build/bin", "build", "."):
            cand = llama / sub / name
            if cand.exists() and os.access(cand, os.X_OK):
                return cand
    return None


def ollama_create(tag: str, modelfile: Path) -> tuple[bool, str]:
    """Register the GGUF with Ollama under `tag`. Returns (ok, detail)."""
    if shutil.which("ollama") is None:
        return False, "ollama CLI not found on PATH"
    proc = subprocess.run(
        ["ollama", "create", tag, "-f", str(modelfile)],
        capture_output=True, text=True, cwd=str(modelfile.parent),
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-400:]
