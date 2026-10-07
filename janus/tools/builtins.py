"""Built-in tools: files, shell, python, and web access.

Risk assessment pattern: file writes and shell/python are escalated to RISKY
when they touch paths outside the workspace or contain destructive commands,
so the approval gate can prompt. Reads and in-workspace writes stay SAFE.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from janus.tools.base import Risk, Tool, ToolResult

# Shell command fragments we always treat as risky regardless of path.
_DANGER_RE = re.compile(
    r"(\brm\b|\bsudo\b|\bmkfs\b|\bdd\b|\bshutdown\b|\breboot\b|\bkill(all)?\b|"
    r"\bchmod\b|\bchown\b|\bmv\b|>\s*/|\bcurl\b|\bwget\b|\bgit\s+push\b|\bpip\s+install\b|"
    r"\bapt\b|\bbrew\b|\bnpm\b)"
)
_MAX_OUTPUT = 8000


def _within_workspace(path: Path, settings) -> bool:
    try:
        resolved = (path if path.is_absolute() else settings.workspace_abs / path).resolve()
        resolved.relative_to(settings.workspace_abs)
        return True
    except (ValueError, OSError):
        return False


def _abs(path_str: str, settings) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (settings.workspace_abs / p)


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_OUTPUT else text[:_MAX_OUTPUT] + "\n...[truncated]"


# --- read_file ---
class ReadFileArgs(BaseModel):
    path: str = Field(description="File path, relative to the workspace or absolute.")


def _read_file(a: ReadFileArgs, *, settings) -> ToolResult:
    p = _abs(a.path, settings)
    if not p.is_file():
        return ToolResult(False, f"not a file: {p}")
    try:
        return ToolResult(True, _clip(p.read_text(errors="replace")))
    except OSError as e:
        return ToolResult(False, f"read error: {e}")


# --- write_file ---
class WriteFileArgs(BaseModel):
    path: str = Field(description="File path, relative to the workspace or absolute.")
    content: str = Field(description="Full file contents to write.")


def _write_file(a: WriteFileArgs, *, settings) -> ToolResult:
    p = _abs(a.path, settings)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(a.content)
        return ToolResult(True, f"wrote {len(a.content)} chars to {p}")
    except OSError as e:
        return ToolResult(False, f"write error: {e}")


def _assess_write(args: dict, settings):
    p = Path(args.get("path", ""))
    if _within_workspace(p, settings):
        return Risk.SAFE, ""
    return Risk.RISKY, f"writes outside workspace: {p}"


# --- list_dir ---
class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="Directory path.")


def _list_dir(a: ListDirArgs, *, settings) -> ToolResult:
    p = _abs(a.path, settings)
    if not p.is_dir():
        return ToolResult(False, f"not a directory: {p}")
    entries = sorted(
        f"{'d' if c.is_dir() else 'f'} {c.name}" for c in p.iterdir()
    )
    return ToolResult(True, _clip("\n".join(entries) or "(empty)"))


# --- run_shell ---
class RunShellArgs(BaseModel):
    command: str = Field(description="Shell command to run (bash -c).")
    timeout: int = Field(default=120, description="Timeout in seconds.")


def _run_shell(a: RunShellArgs, *, settings) -> ToolResult:
    try:
        proc = subprocess.run(
            a.command, shell=True, cwd=str(settings.workspace_abs),
            capture_output=True, text=True, timeout=a.timeout,
        )
        out = f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        return ToolResult(proc.returncode == 0, _clip(out), {"returncode": proc.returncode})
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"command timed out after {a.timeout}s")
    except Exception as e:  # noqa: BLE001
        return ToolResult(False, f"shell error: {e}")


def _assess_shell(args: dict, settings):
    cmd = args.get("command", "")
    if _DANGER_RE.search(cmd):
        return Risk.RISKY, "command contains a potentially destructive operation"
    return Risk.SAFE, ""


# --- run_python ---
class RunPythonArgs(BaseModel):
    code: str = Field(description="Python source to execute in a fresh subprocess.")
    timeout: int = Field(default=120, description="Timeout in seconds.")


def _run_python(a: RunPythonArgs, *, settings) -> ToolResult:
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", a.code], cwd=str(settings.workspace_abs),
            capture_output=True, text=True, timeout=a.timeout,
        )
        out = f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        return ToolResult(proc.returncode == 0, _clip(out), {"returncode": proc.returncode})
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"python timed out after {a.timeout}s")
    except Exception as e:  # noqa: BLE001
        return ToolResult(False, f"python error: {e}")


# --- web_search ---
class WebSearchArgs(BaseModel):
    query: str = Field(description="Search query.")
    max_results: int = Field(default=5, description="How many results.")


def _web_search(a: WebSearchArgs, *, settings) -> ToolResult:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(a.query, max_results=a.max_results))
    except Exception as e:  # noqa: BLE001
        return ToolResult(False, f"search error: {e}")
    lines = [f"{r.get('title','')}\n  {r.get('href','')}\n  {r.get('body','')}" for r in results]
    return ToolResult(True, _clip("\n\n".join(lines) or "(no results)"))


# --- web_fetch ---
class WebFetchArgs(BaseModel):
    url: str = Field(description="URL to fetch and extract readable text from.")


def _web_fetch(a: WebFetchArgs, *, settings) -> ToolResult:
    try:
        import httpx
        import trafilatura
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            html = client.get(a.url).text
        text = trafilatura.extract(html) or ""
    except Exception as e:  # noqa: BLE001
        return ToolResult(False, f"fetch error: {e}")
    return ToolResult(True, _clip(text or "(no extractable text)"))


def builtin_tools() -> list[Tool]:
    return [
        Tool("read_file", "Read a text file.", ReadFileArgs, _read_file, Risk.SAFE),
        Tool("write_file", "Write (overwrite) a text file.", WriteFileArgs, _write_file,
             Risk.SAFE, assess=_assess_write),
        Tool("list_dir", "List a directory's entries.", ListDirArgs, _list_dir, Risk.SAFE),
        Tool("run_shell", "Run a shell command in the workspace.", RunShellArgs, _run_shell,
             Risk.SAFE, assess=_assess_shell),
        Tool("run_python", "Run Python code in an isolated subprocess.", RunPythonArgs,
             _run_python, Risk.SAFE),
        Tool("web_search", "Search the web.", WebSearchArgs, _web_search, Risk.SAFE),
        Tool("web_fetch", "Fetch a URL and extract readable text.", WebFetchArgs, _web_fetch,
             Risk.SAFE),
    ]
