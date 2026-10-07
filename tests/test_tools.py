"""Tool behavior and path-based risk assessment."""
from janus.tools import build_registry
from janus.tools.base import Risk


def test_builtin_tools_present():
    reg = build_registry()
    names = {t.name for t in reg.all()}
    assert {"read_file", "write_file", "run_shell", "run_python"} <= names


def test_write_inside_workspace_is_safe(settings):
    reg = build_registry()
    tool = reg.get("write_file")
    risk, _ = tool.risk_for({"path": "notes.txt", "content": "hi"}, settings)
    assert risk is Risk.SAFE


def test_write_outside_workspace_is_risky(settings):
    reg = build_registry()
    tool = reg.get("write_file")
    risk, reason = tool.risk_for({"path": "/etc/evil.conf", "content": "x"}, settings)
    assert risk is Risk.RISKY
    assert "outside workspace" in reason


def test_shell_danger_is_risky(settings):
    reg = build_registry()
    tool = reg.get("run_shell")
    risk, _ = tool.risk_for({"command": "rm -rf /"}, settings)
    assert risk is Risk.RISKY
    safe_risk, _ = tool.risk_for({"command": "echo hello"}, settings)
    assert safe_risk is Risk.SAFE


def test_read_and_write_roundtrip(settings):
    reg = build_registry()
    w = reg.get("write_file").run({"path": "a.txt", "content": "hello"}, settings=settings)
    assert w.ok
    r = reg.get("read_file").run({"path": "a.txt"}, settings=settings)
    assert r.ok and "hello" in r.output
