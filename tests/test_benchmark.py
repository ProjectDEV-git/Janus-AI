"""Scorecard comparison is the fitness gate for every self-change."""
from janus.benchmark import Scorecard, TaskScore, load_tasks


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
