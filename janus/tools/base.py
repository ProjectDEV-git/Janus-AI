"""Tool abstraction and registry.

A Tool wraps a plain Python callable with:
  - a name + description (shown to the model),
  - a pydantic arg schema (validates the model's JSON args),
  - a base risk level, and an optional `assess` hook that can escalate risk
    based on the concrete arguments (e.g. a path outside the workspace).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel


class Risk(str, Enum):
    SAFE = "safe"      # read-only or confined to the workspace
    RISKY = "risky"    # destructive, out-of-workspace, or self-modifying


@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: dict = field(default_factory=dict)


# assess(args, settings) -> (Risk, reason). Optional per-tool risk escalation.
AssessFn = Callable[[dict, Any], tuple[Risk, str]]


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., ToolResult]
    base_risk: Risk = Risk.SAFE
    assess: AssessFn | None = None

    def validate(self, args: dict) -> BaseModel:
        return self.args_model(**args)

    def risk_for(self, args: dict, settings: Any) -> tuple[Risk, str]:
        if self.assess is not None:
            return self.assess(args, settings)
        return self.base_risk, ""

    def run(self, args: dict, *, settings: Any) -> ToolResult:
        parsed = self.validate(args)
        return self.func(parsed, settings=settings)

    def schema_hint(self) -> str:
        """Compact signature for the system prompt."""
        fields = ", ".join(
            f"{n}: {info.annotation.__name__ if hasattr(info.annotation, '__name__') else info.annotation}"
            for n, info in self.args_model.model_fields.items()
        )
        return f"{self.name}({fields}) [{self.base_risk.value}] - {self.description}"


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def catalog(self) -> str:
        return "\n".join(f"- {t.schema_hint()}" for t in self.all())
