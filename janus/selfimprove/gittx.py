"""Git transaction helper for reversible self-edits.

Wraps a patch application in a throwaway branch so a self-edit can be either
merged into the working branch (keep) or discarded entirely (revert), with no
trace left on reject. Requires the repo to have at least one commit.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class GitError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(root),
                          capture_output=True, text=True)


def has_commit(root: Path) -> bool:
    return _git(root, "rev-parse", "--verify", "HEAD").returncode == 0


def patch_paths(root: Path, diff_text: str) -> set[str]:
    """Files a diff would touch (as git sees them), or an empty set if it doesn't parse."""
    r = subprocess.run(["git", "apply", "--numstat", "-"], cwd=str(root), input=diff_text,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return set()
    paths = set()
    for line in r.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            paths.add(parts[2])
    return paths


def current_branch(root: Path) -> str:
    r = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    return r.stdout.strip()


@dataclass
class PatchTransaction:
    root: Path
    orig_branch: str = ""
    tx_branch: str = ""
    committed: bool = False

    def begin_and_apply(self, diff_text: str, paths: list[str]) -> tuple[bool, str]:
        """Create a tx branch, apply the diff, commit only `paths`. Returns (ok, detail)."""
        if not has_commit(self.root):
            return False, "repo has no commits; cannot run a self-edit transaction"
        self.orig_branch = current_branch(self.root)
        # Staged or unstaged changes to tracked files would be swept into (or lost by) the tx.
        if _git(self.root, "diff", "--quiet", "HEAD").returncode != 0:
            return False, "working tree is dirty; commit or stash before self-editing"
        self.tx_branch = f"janus/selfedit/{int(time.time())}"
        if _git(self.root, "checkout", "-b", self.tx_branch).returncode != 0:
            return False, "could not create transaction branch"

        patch_file = self.root / ".janus_patch.diff"
        patch_file.write_text(diff_text)
        applied = _git(self.root, "apply", "--whitespace=nowarn", str(patch_file))
        patch_file.unlink(missing_ok=True)
        if applied.returncode != 0:
            self._abort()
            return False, f"patch did not apply: {applied.stderr[:300]}"

        # Commit only the patched files, never untracked runtime state (workspace, db, ...).
        _git(self.root, "add", "--", *paths)
        committed = _git(self.root, "commit", "-m", "janus: self-edit candidate")
        if committed.returncode != 0:
            self._abort()
            return False, f"commit failed: {committed.stderr[:200]}"
        self.committed = True
        return True, "applied"

    def keep(self) -> None:
        """Accept: merge the tx branch back into the original branch."""
        if not self.committed:
            return
        _git(self.root, "checkout", self.orig_branch)
        _git(self.root, "merge", "--ff-only", self.tx_branch)
        _git(self.root, "branch", "-D", self.tx_branch)

    def revert(self) -> None:
        """Reject: return to the original branch and delete the tx branch."""
        self._abort()

    def _abort(self) -> None:
        if self.orig_branch:
            _git(self.root, "checkout", "--force", self.orig_branch)
        if self.tx_branch:
            _git(self.root, "branch", "-D", self.tx_branch)
