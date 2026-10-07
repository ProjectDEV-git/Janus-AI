"""Tool registry assembly: built-ins plus auto-discovered dynamic tools.

A dynamic tool is any module in janus/tools/dynamic/ that exposes a top-level
`TOOL` (a janus.tools.base.Tool) or a `get_tools()` returning a list of them.
This is how Janus loads tools it has written for itself.
"""
from __future__ import annotations

import importlib
import pkgutil

from janus.tools.base import Registry, Risk, Tool, ToolResult
from janus.tools.builtins import builtin_tools

__all__ = ["Registry", "Risk", "Tool", "ToolResult", "build_registry"]


def build_registry(include_dynamic: bool = True) -> Registry:
    reg = Registry()
    for t in builtin_tools():
        reg.register(t)
    if include_dynamic:
        _load_dynamic(reg)
    return reg


def _load_dynamic(reg: Registry) -> None:
    from janus.tools import dynamic as dyn_pkg

    for mod_info in pkgutil.iter_modules(dyn_pkg.__path__):
        if mod_info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"janus.tools.dynamic.{mod_info.name}")
        except Exception:  # noqa: BLE001 - a broken self-written tool must not crash Janus
            continue
        if hasattr(mod, "TOOL") and isinstance(mod.TOOL, Tool):
            reg.register(mod.TOOL)
        elif hasattr(mod, "get_tools"):
            for t in mod.get_tools():
                if isinstance(t, Tool):
                    reg.register(t)
