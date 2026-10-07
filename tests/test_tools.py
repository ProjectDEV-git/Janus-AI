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


def test_read_outside_workspace_is_risky(settings):
    reg = build_registry()
    for name, args in [("read_file", {"path": "/root/.ssh/id_rsa"}),
                       ("list_dir", {"path": "/"}),
                       ("read_file", {"path": "../outside.txt"})]:
        risk, reason = reg.get(name).risk_for(args, settings)
        assert risk is Risk.RISKY, (name, args)
        assert "outside workspace" in reason
    assert reg.get("read_file").risk_for({"path": "a.txt"}, settings)[0] is Risk.SAFE
    assert reg.get("list_dir").risk_for({}, settings)[0] is Risk.SAFE


def test_code_execution_is_risky_without_sandbox(settings):
    # No command or snippet is "safe" when it runs unconfined: inspection can't tell.
    reg = build_registry()
    assert reg.get("run_shell").risk_for({"command": "echo hello"}, settings)[0] is Risk.RISKY
    assert reg.get("run_python").risk_for(
        {"code": "import os; os.remove('/x')"}, settings)[0] is Risk.RISKY


def test_code_execution_is_safe_inside_sandbox(settings, monkeypatch):
    from janus import sandbox
    monkeypatch.setattr(sandbox, "enabled", lambda s: True)
    reg = build_registry()
    assert reg.get("run_shell").risk_for({"command": "echo hi"}, settings)[0] is Risk.SAFE
    assert reg.get("run_python").risk_for({"code": "print(1)"}, settings)[0] is Risk.SAFE


def test_run_python_can_import_workspace_files(settings):
    reg = build_registry()
    reg.get("write_file").run({"path": "m.py", "content": "def f():\n    return 7\n"},
                              settings=settings)
    r = reg.get("run_python").run({"code": "from m import f; print(f())"}, settings=settings)
    assert r.ok and "7" in r.output, r.output


def test_read_and_write_roundtrip(settings):
    reg = build_registry()
    w = reg.get("write_file").run({"path": "a.txt", "content": "hello"}, settings=settings)
    assert w.ok
    r = reg.get("read_file").run({"path": "a.txt"}, settings=settings)
    assert r.ok and "hello" in r.output
