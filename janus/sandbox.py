"""OS-level sandbox for running model-written code (shell, Python, checks, tests).

Inspecting a command or a Python snippet cannot tell whether it is harmful, so
Janus confines execution instead, using the native mechanism of each OS:

  - Linux:   bubblewrap (`bwrap`)
  - macOS:   Seatbelt (`sandbox-exec`, built in)
  - Windows: no sandbox backend; code execution goes through the approval gate.

When a backend is available, code runs with:

  - the whole filesystem mounted read-only,
  - only the workspace (and a private /tmp) writable,
  - the home directory hidden behind an empty tmpfs (no ~/.ssh, tokens, etc.),
  - no network (and, under bwrap, its own PID/IPC namespaces).

The `sandbox` setting picks the mode: "auto" (use the OS backend when it works),
"on" (require one; refuse to run code without it), "bwrap" / "seatbelt" (require
that specific backend), or "off". When no sandbox is in
use, the tools that execute code are classed RISKY so the approval gate asks.
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class SandboxUnavailable(RuntimeError):
    pass


@functools.cache
def _bwrap_works() -> bool:
    exe = shutil.which("bwrap")
    if exe is None or not sys.platform.startswith("linux"):
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


_SEATBELT = "/usr/bin/sandbox-exec"


@functools.cache
def _seatbelt_works() -> bool:
    if sys.platform != "darwin" or not os.path.exists(_SEATBELT):
        return False
    try:
        proc = subprocess.run([_SEATBELT, "-p", "(version 1)(allow default)(deny network*)",
                               "/usr/bin/true"], capture_output=True, timeout=10)
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def backend(settings) -> str | None:
    """The sandbox backend that will be used ("bwrap", "seatbelt"), or None."""
    mode = getattr(settings, "sandbox", "auto")
    if mode == "off":
        return None
    if mode in ("auto", "on", "bwrap") and _bwrap_works():
        return "bwrap"
    if mode in ("auto", "on", "seatbelt") and _seatbelt_works():
        return "seatbelt"
    return None


def enabled(settings) -> bool:
    """True when code execution will actually be sandboxed."""
    return backend(settings) is not None


def install_hint() -> str:
    """How to get a sandbox on this OS, for error messages."""
    if sys.platform.startswith("linux"):
        return "install bubblewrap (`apt install bubblewrap` or your distro's equivalent)"
    if sys.platform == "darwin":
        return "sandbox-exec (Seatbelt) should be built in; check that /usr/bin/sandbox-exec works"
    return f"no sandbox backend exists for {sys.platform}"


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


def _sb_str(p) -> str:
    return '"' + str(p).replace("\\", "\\\\").replace('"', '\\"') + '"'


def seatbelt_profile(*, writable: Path, tmp: Path, extra_ro: tuple[Path, ...] = ()) -> str:
    """A Seatbelt (SBPL) profile mirroring the bwrap policy. Later rules win."""
    writable = Path(writable).resolve()
    home = Path(os.path.expanduser("~")).resolve()
    rules = ["(version 1)", "(allow default)", "(deny network*)",
             "(deny file-write*)",
             f"(allow file-write* (subpath {_sb_str(writable)}) (subpath {_sb_str(Path(tmp).resolve())})"
             ' (literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty")'
             ' (regex #"^/dev/fd/") (regex #"^/dev/ttys"))']
    if home != Path("/"):
        rules.append(f"(deny file-read* (subpath {_sb_str(home)}))")
        rules.append(f"(allow file-read-metadata (literal {_sb_str(home)}))")
        keep = {sys.prefix, sys.base_prefix, sys.exec_prefix, *map(str, extra_ro), str(writable)}
        for p in _ro_paths_under(home, keep):
            rules.append(f"(allow file-read* (subpath {_sb_str(p)}))")
    return "\n".join(rules)


def run(cmd: list[str], *, settings, writable: Path, cwd: Path | None = None,
        timeout: float, extra_ro: tuple[Path, ...] = (),
        env: dict | None = None) -> subprocess.CompletedProcess:
    """Run `cmd`, sandboxed when enabled. Raises SandboxUnavailable if required but missing."""
    mode = getattr(settings, "sandbox", "auto")
    kind = backend(settings)
    if kind is None and mode not in ("auto", "off"):
        raise SandboxUnavailable(f"sandbox={mode!r} but no usable sandbox: {install_hint()}")
    if kind == "bwrap":
        full = wrap(cmd, writable=writable, cwd=cwd, extra_ro=extra_ro)
    elif kind == "seatbelt":
        with tempfile.TemporaryDirectory(prefix="janus-sb-") as tmp:
            env = dict(os.environ if env is None else env,
                       HOME=str(Path(writable).resolve()), TMPDIR=tmp)
            profile = seatbelt_profile(writable=writable, tmp=Path(tmp), extra_ro=extra_ro)
            return subprocess.run([_SEATBELT, "-p", profile, *cmd], cwd=str(cwd or writable),
                                  capture_output=True, text=True, timeout=timeout, env=env)
    else:
        full = list(cmd)
    return subprocess.run(full, cwd=str(cwd or writable), capture_output=True,
                          text=True, timeout=timeout, env=env)
