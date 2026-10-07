"""Scorecard comparison is the fitness gate for every self-change."""
import janus.benchmark as bench
from janus.benchmark import Scorecard, TaskScore, load_tasks
from tests.conftest import FakeLLM


def _card(scores):
    return Scorecard([TaskScore(*s) for s in scores])


def test_tasks_load():
    tasks = load_tasks()
    assert len(tasks) >= 5
    assert all("goal" in t and "check" in t for t in tasks)


def test_cheaper_same_success_is_better():
    base = _card([("a", True, 100, 10, "finished")])
    cheaper = _card([("a", True, 50, 8, "finished")])
    better, _ = cheaper.is_better_than(base)
    assert better


def test_lower_success_is_rejected():
    base = _card([("a", True, 100, 10, "finished"), ("b", True, 100, 10, "finished")])
    worse = _card([("a", False, 1, 1, "budget"), ("b", True, 1, 1, "finished")])
    better, _ = worse.is_better_than(base)
    assert not better


def test_same_cost_is_not_improvement():
    base = _card([("a", True, 100, 10, "finished")])
    same = _card([("a", True, 100, 10, "finished")])
    better, _ = same.is_better_than(base)
    assert not better


def test_no_baseline_accepts():
    better, _ = _card([("a", True, 1, 1, "finished")]).is_better_than(None)
    assert better


def test_tiny_saving_is_within_noise():
    base = _card([("a", True, 1000, 10, "finished")])
    slightly = _card([("a", True, 980, 10, "finished")])  # 2% < 5% margin
    assert not slightly.is_better_than(base)[0]
    assert slightly.is_better_than(base, min_gain=0.01)[0]


def test_faster_alone_is_not_improvement():
    base = _card([("a", True, 100, 10, "finished")])
    faster = _card([("a", True, 100, 2, "finished")])
    assert not faster.is_better_than(base)[0]


def test_repeats_compare_per_task_cost():
    base = _card([("a", True, 100, 1, "finished")])
    repeated = _card([("a", True, 90, 1, "finished"), ("a", True, 90, 1, "finished")])
    assert repeated.is_better_than(base)[0]  # 90/task vs 100/task, despite 180 total


def test_check_imports_the_workspace_file(settings):
    # Regression: with `python -I` the cwd is not importable and every check failed.
    (settings.workspace_abs / "reverse.py").write_text("def reverse(s):\n    return s[::-1]\n")
    check = next(t["check"] for t in load_tasks() if t["id"] == "reverse_string")
    assert bench._check(check, settings)


def test_each_task_gets_a_fresh_workspace(settings, monkeypatch):
    # A file left behind by an earlier run must not pass the check.
    (settings.workspace_abs / "reverse.py").write_text("def reverse(s):\n    return s[::-1]\n")
    monkeypatch.setattr(bench, "LLM", lambda s: FakeLLM([
        {"thought": "claim done", "finish": {"done": True, "evidence": "none"}}]))
    task = next(t for t in load_tasks() if t["id"] == "reverse_string")
    score = bench._run_task(task, settings)
    assert score.status == "finished"
    assert not score.success
    assert (settings.workspace_abs / "reverse.py").exists()  # the user's workspace is untouched


def test_preflight_requires_sandbox_or_trust(settings, monkeypatch):
    from janus import sandbox
    monkeypatch.setattr(sandbox, "enabled", lambda s: False)
    assert not bench.preflight(settings, trust=False)[0]
    assert bench.preflight(settings, trust=True)[0]
    monkeypatch.setattr(sandbox, "enabled", lambda s: True)
    assert bench.preflight(settings, trust=False)[0]
