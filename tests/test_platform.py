"""Cross-OS sandbox selection and shell choice (no real sandbox needed)."""
from pathlib import Path

from janus import sandbox
from janus.tools import builtins


def _settings(mode):
    return type("S", (), {"sandbox": mode})()


def test_backend_prefers_bwrap_then_seatbelt(monkeypatch):
    monkeypatch.setattr(sandbox, "_bwrap_works", lambda: False)
    monkeypatch.setattr(sandbox, "_seatbelt_works", lambda: True)
    assert sandbox.backend(_settings("auto")) == "seatbelt"
    assert sandbox.backend(_settings("bwrap")) is None
    assert sandbox.backend(_settings("off")) is None


def test_required_sandbox_raises_without_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "_bwrap_works", lambda: False)
    monkeypatch.setattr(sandbox, "_seatbelt_works", lambda: False)
    try:
        sandbox.run(["true"], settings=_settings("on"), writable=tmp_path, timeout=5)
    except sandbox.SandboxUnavailable:
        pass
    else:
        raise AssertionError("expected SandboxUnavailable")


def test_seatbelt_profile_confines_writes_and_network(tmp_path):
    prof = sandbox.seatbelt_profile(writable=tmp_path, tmp=tmp_path / "t")
    assert "(deny network*)" in prof and "(deny file-write*)" in prof
    assert str(tmp_path.resolve()) in prof
    home = str(Path("~").expanduser().resolve())
    assert f'(deny file-read* (subpath "{home}"))' in prof


def test_windows_shell_falls_back_to_powershell(monkeypatch):
    monkeypatch.setattr(builtins.os, "name", "nt")
    monkeypatch.setattr(builtins.shutil, "which", lambda x: x if x == "powershell" else None)
    assert builtins._shell_argv("dir")[0] == "powershell"
    monkeypatch.setattr(builtins.shutil, "which", lambda x: None)
    assert builtins._shell_argv("dir") == ["cmd", "/c", "dir"]
