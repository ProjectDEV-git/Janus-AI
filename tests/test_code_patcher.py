"""Reversibility of self-edits: the git transaction keeps or discards cleanly."""
import subprocess
from pathlib import Path

import pytest

from janus.selfimprove.gittx import PatchTransaction, current_branch, has_commit


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
    ok, detail = tx.begin_and_apply(_DIFF)
    assert ok, detail
    tx.keep()
    assert current_branch(repo) == "work"
    assert "VALUE = 2" in (repo / "mod.py").read_text()
    # tx branch is gone
    branches = _git(repo, "branch").stdout
    assert "selfedit" not in branches


def test_revert_discards_change(repo):
    tx = PatchTransaction(root=repo)
    ok, detail = tx.begin_and_apply(_DIFF)
    assert ok, detail
    tx.revert()
    assert current_branch(repo) == "work"
    assert (repo / "mod.py").read_text() == "VALUE = 1\n"
    branches = _git(repo, "branch").stdout
    assert "selfedit" not in branches


def test_bad_patch_aborts_cleanly(repo):
    tx = PatchTransaction(root=repo)
    ok, detail = tx.begin_and_apply("not a real diff\n")
    assert not ok
    assert current_branch(repo) == "work"
    branches = _git(repo, "branch").stdout
    assert "selfedit" not in branches


def test_dirty_tree_is_refused(repo):
    (repo / "mod.py").write_text("VALUE = 99\n")  # uncommitted change
    tx = PatchTransaction(root=repo)
    ok, detail = tx.begin_and_apply(_DIFF)
    assert not ok
    assert "dirty" in detail


def test_no_commit_repo_is_refused(tmp_path):
    _git(tmp_path, "init", "-q")
    assert not has_commit(tmp_path)
    tx = PatchTransaction(root=tmp_path)
    ok, detail = tx.begin_and_apply(_DIFF)
    assert not ok
