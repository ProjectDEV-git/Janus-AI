"""The OS sandbox that makes running model-written code SAFE. Needs bubblewrap."""
import sys
from pathlib import Path

import pytest

from janus import sandbox
from janus.tools import build_registry

pytestmark = pytest.mark.skipif(not sandbox._bwrap_works(),
                                reason="bubblewrap not installed or not usable here")


@pytest.fixture
def sb(settings):
    settings.sandbox = "bwrap"
    return settings


def _py(settings, code):
    return build_registry().get("run_python").run({"code": code}, settings=settings)


def test_workspace_is_writable(sb):
    r = _py(sb, "open('out.txt', 'w').write('hi'); print('OK')")
    assert r.ok, r.output
    assert (sb.workspace_abs / "out.txt").read_text() == "hi"


def test_outside_workspace_is_read_only(sb, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    r = _py(sb, f"import os; os.remove({str(victim)!r})")
    assert not r.ok
    assert victim.read_text() == "keep me"


def test_home_is_hidden(sb):
    home = Path("~").expanduser()
    secret = home / ".janus_sandbox_probe"
    secret.write_text("secret")
    try:
        r = _py(sb, f"print(open({str(secret)!r}).read())")
        assert not r.ok and "secret" not in r.output.split("--- stderr ---")[0]
    finally:
        secret.unlink()


def test_no_network(sb):
    r = _py(sb, "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)")
    assert not r.ok


def test_shell_runs_in_workspace(sb):
    r = build_registry().get("run_shell").run({"command": "pwd"}, settings=sb)
    assert r.ok and str(sb.workspace_abs) in r.output


def test_required_sandbox_refuses_when_missing(sb, monkeypatch):
    monkeypatch.setattr(sandbox, "_bwrap_works", lambda: False)
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.run([sys.executable, "-c", "pass"], settings=sb,
                    writable=sb.workspace_abs, timeout=10)
