"""OS-level sandbox for running model-written code (shell, Python, checks, tests).

Inspecting a command or a Python snippet cannot tell whether it is harmful, so
Janus confines execution instead. When bubblewrap (`bwrap`) is available, code
runs with:

  - the whole filesystem mounted read-only,
  - only the workspace (and a private /tmp) writable,
  - the home directory hidden behind an empty tmpfs (no ~/.ssh, tokens, etc.),
  - no network, and its own PID/IPC namespaces.

The `sandbox` setting picks the mode: "auto" (use bwrap when it works), "bwrap"
(require it; refuse to run code without it), or "off". When no sandbox is in
use, the tools that execute code are classed RISKY so the approval gate asks.
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path


class SandboxUnavailable(RuntimeError):
    pass


@functools.cache
def _bwrap_works() -> bool:
    exe = shutil.which("bwrap")
    if exe is None:
        return False
    try:
        proc = subprocess.run(
            [exe, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
             "--unshare-all", "--die-with-parent", "true"],
            capture_output=True, timeout=10,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def enabled(settings) -> bool:
    """True when code execution will actually be sandboxed."""
    mode = getattr(settings, "sandbox", "auto")
    if mode == "off":
        return False
    return _bwrap_works()


def _ro_paths_under(hidden: Path, paths) -> list[Path]:
    """Paths that live under a hidden directory and must be re-exposed read-only."""
    out = []
    for p in paths:
        p = Path(p).resolve()
        if p.exists() and (p == hidden or hidden in p.parents):
            out.append(p)
    return out


def wrap(cmd: list[str], *, writable: Path, cwd: Path | None = None,
         extra_ro: tuple[Path, ...] = ()) -> list[str]:
    """Build the bwrap command line that runs `cmd` confined to `writable`."""
    writable = Path(writable).resolve()
    args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp"]
    home = Path(os.path.expanduser("~")).resolve()
    if home != Path("/"):
        args += ["--tmpfs", str(home)]
        # The interpreter (e.g. a venv) or Janus itself may live under $HOME.
        keep = {sys.prefix, sys.base_prefix, sys.exec_prefix, *map(str, extra_ro)}
        for p in _ro_paths_under(home, keep):
            args += ["--ro-bind", str(p), str(p)]
    args += ["--bind", str(writable), str(writable),
             "--chdir", str(Path(cwd).resolve() if cwd else writable),
             "--setenv", "HOME", str(writable),
             "--unshare-all", "--die-with-parent", "--new-session", "--"]
    return args + list(cmd)


def run(cmd: list[str], *, settings, writable: Path, cwd: Path | None = None,
        timeout: float, extra_ro: tuple[Path, ...] = (),
        env: dict | None = None) -> subprocess.CompletedProcess:
    """Run `cmd`, sandboxed when enabled. Raises SandboxUnavailable if required but missing."""
    mode = getattr(settings, "sandbox", "auto")
    if enabled(settings):
        full = wrap(cmd, writable=writable, cwd=cwd, extra_ro=extra_ro)
    elif mode == "bwrap":
        raise SandboxUnavailable("sandbox='bwrap' but bubblewrap is not installed or not usable")
    else:
        full = list(cmd)
    return subprocess.run(full, cwd=str(cwd or writable), capture_output=True,
                          text=True, timeout=timeout, env=env)
