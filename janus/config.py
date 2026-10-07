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
    model: str = "hf.co/HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive:IQ3_M"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.7
    num_predict: int = 0

    # Workspace
    workspace: Path = Field(default=Path("./janus_workspace"))

    # Budgets
    max_iterations: int = 40
    max_seconds: int = 1800
    max_tokens: int = 200_000

    # Approval policy
    auto_approve_safe: bool = True
    trust: bool = False

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
