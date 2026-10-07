"""Configuration for Janus, loaded from janus.toml with JANUS_* env overrides."""
from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_FILENAME = "janus.toml"


class Settings(BaseSettings):
    """Runtime settings. Precedence: env (JANUS_*) > janus.toml > defaults."""

    model_config = SettingsConfigDict(env_prefix="JANUS_", extra="ignore")

    # Model
    provider: str = "ollama"         # ollama | openai (any OpenAI-compatible API) | anthropic
    model: str = "hf.co/HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive:IQ3_M"
    ollama_host: str = "http://localhost:11434"
    api_base: str = ""               # openai/anthropic endpoint; "" = the provider's official API
    api_key: str = ""                # prefer JANUS_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
    temperature: float = 0.7
    num_predict: int = 0
    num_ctx: int = 8192              # Ollama context window (0 = Ollama's default, often too small)
    seed: int | None = None          # fixed sampling seed (None = random)

    # Workspace
    workspace: Path = Field(default=Path("./janus_workspace"))

    # Budgets
    max_iterations: int = 40
    max_seconds: int = 1800
    max_tokens: int = 200_000

    # Approval policy
    auto_approve_safe: bool = True
    trust: bool = False
    # Where model-written code runs: "auto" (bubblewrap if usable), "bwrap" (required), "off".
    sandbox: str = "auto"

    # Benchmark (the fitness gate for self-improvement)
    bench_temperature: float = 0.0   # deterministic-ish runs so A/B differences are real
    bench_seed: int = 0
    bench_repeats: int = 1           # run each task this many times
    bench_min_gain: float = 0.05     # same success must cut tokens/task by at least this fraction

    # Self-training (LoRA on the base Gemma weights)
    base_model_id: str = "google/gemma-3n-E2B"   # full-precision source for training
    model_prefix: str = "janus"                   # produced Ollama tags: janus:v1, v2, ...
    gguf_quant: str = "Q4_K_M"                     # quant for the produced GGUF
    train_min_examples: int = 20                  # refuse to train below this
    train_epochs: int = 1
    train_lr: float = 2e-4
    train_batch_size: int = 1
    train_grad_accum: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    llama_cpp_dir: str = ""                        # path to llama.cpp (for GGUF convert); "" = autodetect

    # --- Derived paths (not read from config) ---
    @property
    def root(self) -> Path:
        """Project root: directory containing janus.toml, else cwd."""
        return _find_config_dir()

    @property
    def workspace_abs(self) -> Path:
        ws = self.workspace
        return ws if ws.is_absolute() else (self.root / ws).resolve()

    @property
    def db_path(self) -> Path:
        return self.root / "janus_state.sqlite"

    @property
    def stopfile(self) -> Path:
        return self.root / ".janus_stop"

    @property
    def transcripts_dir(self) -> Path:
        return self.root / "transcripts"

    @property
    def adapters_dir(self) -> Path:
        return self.root / "adapters"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"


def _find_config_dir(start: Path | None = None) -> Path:
    """Walk upward from `start` (or cwd) to find the dir holding janus.toml."""
    cur = (start or Path.cwd()).resolve()
    for d in [cur, *cur.parents]:
        if (d / CONFIG_FILENAME).is_file():
            return d
    return cur


def load_settings(start: Path | None = None) -> Settings:
    """Load settings from janus.toml (if present) then apply env overrides."""
    cfg_dir = _find_config_dir(start)
    file_values: dict = {}
    cfg_file = cfg_dir / CONFIG_FILENAME
    if cfg_file.is_file():
        with cfg_file.open("rb") as fh:
            file_values = tomllib.load(fh)
    # BaseSettings applies env overrides on top of the kwargs we pass.
    return Settings(**file_values)
