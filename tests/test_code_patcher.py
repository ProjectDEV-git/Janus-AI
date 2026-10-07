"""Reversibility of self-edits: the git transaction keeps or discards cleanly."""
import subprocess
from pathlib import Path

import pytest

from janus.selfimprove.gittx import PatchTransaction, current_branch, has_commit, patch_paths


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t.t")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "checkout", "-q", "-b", "work")
    (tmp_path / "mod.py").write_text("VALUE = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


_DIFF = """--- a/mod.py
+++ b/mod.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""


def test_keep_merges_change(repo):
    tx = PatchTransaction(root=repo)
    ok, detail = tx.begin_and_apply(_DIFF, paths=["mod.py"])
    assert ok, detail
    tx.keep()
    assert current_branch(repo) == "work"
    assert "VALUE = 2" in (repo / "mod.py").read_text()
    # tx branch is gone
    branches = _git(repo, "branch").stdout
    assert "selfedit" not in branches


def test_revert_discards_change(repo):
    tx = PatchTransaction(root=repo)
    ok, detail = tx.begin_and_apply(_DIFF, paths=["mod.py"])
    assert ok, detail
    tx.revert()
    assert current_branch(repo) == "work"
    assert (repo / "mod.py").read_text() == "VALUE = 1\n"
    branches = _git(repo, "branch").stdout
    assert "selfedit" not in branches


def test_bad_patch_aborts_cleanly(repo):
    tx = PatchTransaction(root=repo)
    ok, detail = tx.begin_and_apply("not a real diff\n", paths=["mod.py"])
    assert not ok
    assert current_branch(repo) == "work"
    branches = _git(repo, "branch").stdout
    assert "selfedit" not in branches


def test_dirty_tree_is_refused(repo):
    (repo / "mod.py").write_text("VALUE = 99\n")  # uncommitted change
    tx = PatchTransaction(root=repo)
    ok, detail = tx.begin_and_apply(_DIFF, paths=["mod.py"])
    assert not ok
    assert "dirty" in detail


def test_no_commit_repo_is_refused(tmp_path):
    _git(tmp_path, "init", "-q")
    assert not has_commit(tmp_path)
    tx = PatchTransaction(root=tmp_path)
    ok, detail = tx.begin_and_apply(_DIFF, paths=["mod.py"])
    assert not ok


def test_staged_change_counts_as_dirty(repo):
    (repo / "mod.py").write_text("VALUE = 99\n")
    _git(repo, "add", "mod.py")
    ok, detail = PatchTransaction(root=repo).begin_and_apply(_DIFF, paths=["mod.py"])
    assert not ok and "dirty" in detail


def test_untracked_files_are_not_committed(repo):
    (repo / "scratch.db").write_text("runtime state")
    tx = PatchTransaction(root=repo)
    ok, detail = tx.begin_and_apply(_DIFF, paths=["mod.py"])
    assert ok, detail
    tx.keep()
    assert "scratch.db" not in _git(repo, "ls-files").stdout
    assert (repo / "scratch.db").exists()


def test_patch_paths_lists_every_touched_file(repo):
    multi = _DIFF + """--- a/tests/test_x.py
+++ b/tests/test_x.py
@@ -0,0 +1 @@
+assert True
"""
    assert patch_paths(repo, _DIFF) == {"mod.py"}
    assert patch_paths(repo, multi) == {"mod.py", "tests/test_x.py"}
    assert patch_paths(repo, "garbage") == set()


class _FakeCtxLLM:
    def __init__(self, diff):
        self.diff = diff

    def chat_json(self, messages, **_):
        return {"rationale": "r", "diff": self.diff}


def test_patcher_rejects_diff_outside_target(tmp_path, settings, monkeypatch):
    from janus.selfimprove import code_patcher
    from janus.selfimprove.base import ImproveContext

    root = tmp_path / "repo"
    (root / "janus").mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    (root / "janus" / "agent.py").write_text("X = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    monkeypatch.setattr(type(settings), "root", property(lambda self: root))

    sneaky = """--- a/janus/agent.py
+++ b/janus/agent.py
@@ -1 +1 @@
-X = 1
+X = 2
--- a/tests/test_safety.py
+++ b/tests/test_safety.py
@@ -0,0 +1 @@
+pass
"""
    seen = []
    gate = type("G", (), {"decide": lambda self, *a: seen.append(a)})()
    ctx = ImproveContext(settings=settings, llm=_FakeCtxLLM(sneaky), memory=None)
    assert code_patcher.CodePatcherStrategy(gate=gate).propose(ctx) is None
    assert not seen  # rejected before even asking for approval
    assert (root / "janus" / "agent.py").read_text() == "X = 1\n"
