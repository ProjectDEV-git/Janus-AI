"""Agent loop: finish, blocked actions, and the kill switch / budgets."""
import json

from janus.agent import Agent, StopController
from janus.approval import ApprovalGate
from janus.memory import Memory
from janus.tools import build_registry
from tests.conftest import FakeLLM


def _agent(settings, script, prompter=None):
    llm = FakeLLM(script)
    reg = build_registry()
    mem = Memory(settings.db_path)
    gate = ApprovalGate(settings, prompter=prompter)
    stop = StopController(stopfile=settings.stopfile)
    return Agent(settings, llm, reg, mem, gate, stop), mem, stop


def test_agent_finishes_with_evidence(settings):
    agent, mem, _ = _agent(settings, [
        {"thought": "done already", "finish": {"done": True, "evidence": "trivially true"}},
    ])
    out = agent.run("do nothing")
    assert out.status == "finished"
    assert "trivially true" in out.evidence
    mem.close()


def test_iteration_budget_stops_loop(settings):
    settings.max_iterations = 2
    # Never finishes: always a safe action.
    agent, mem, _ = _agent(settings, [])
    out = agent.run("loop forever")
    assert out.status == "budget"
    assert out.iterations == 2
    mem.close()


def test_stopfile_halts_loop(settings):
    settings.stopfile.write_text("stop")
    agent, mem, _ = _agent(settings, [])
    out = agent.run("should be stopped immediately")
    assert out.status == "stopped"
    assert out.iterations == 0
    mem.close()


def test_blocked_tool_does_not_crash_and_is_recorded(settings):
    settings.trust = False  # risky write will be blocked (no prompter)
    agent, mem, _ = _agent(settings, [
        {"thought": "write outside", "action": {"tool": "write_file",
                                                "args": {"path": "/etc/x", "content": "y"}}},
        {"thought": "give up", "finish": {"done": True, "evidence": "stopped trying"}},
    ])
    out = agent.run("try a risky write then finish")
    assert out.status == "finished"
    events = mem.events_for(out.run_id)
    assert any(e["kind"] == "decision" and "blocked" in e["payload"] for e in events)
    mem.close()


def test_unknown_tool_is_handled(settings):
    agent, mem, _ = _agent(settings, [
        {"thought": "use missing", "action": {"tool": "nope", "args": {}}},
        {"thought": "done", "finish": {"done": True, "evidence": "recovered"}},
    ])
    out = agent.run("use a nonexistent tool")
    assert out.status == "finished"
    mem.close()


def test_model_sees_its_own_previous_actions(settings):
    agent, mem, _ = _agent(settings, [
        {"thought": "look around", "action": {"tool": "list_dir", "args": {"path": "."}}},
        {"thought": "done", "finish": {"done": True, "evidence": "listed"}},
    ])
    agent.run("list the workspace")
    second_call = agent.llm.seen[1]
    assistant = [m for m in second_call if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert json.loads(assistant[0]["content"])["action"]["tool"] == "list_dir"
    # The observation follows the action it belongs to.
    assert second_call[-1]["role"] == "user" and "OBSERVATION" in second_call[-1]["content"]
    mem.close()
