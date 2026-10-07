"""The approval gate is the primary safety control; these pin its behavior."""
from janus.approval import ApprovalGate, Decision
from janus.tools import build_registry


def _tool(name):
    return build_registry().get(name)


def test_safe_action_auto_approved(settings):
    settings.auto_approve_safe = True
    gate = ApprovalGate(settings, prompter=None)
    res = gate.decide(_tool("read_file"), {"path": "x.txt"})
    assert res.decision is Decision.APPROVED
    assert res.policy == "auto_approve_safe"


def test_risky_action_blocked_without_prompter(settings):
    settings.trust = False
    gate = ApprovalGate(settings, prompter=None)
    res = gate.decide(_tool("write_file"), {"path": "/etc/passwd", "content": "x"})
    assert res.decision is Decision.BLOCKED
    assert res.policy == "blocked_noninteractive"


def test_risky_action_approved_in_trust_mode(settings):
    settings.trust = True
    gate = ApprovalGate(settings, prompter=None)
    res = gate.decide(_tool("run_shell"), {"command": "sudo rm -rf /tmp/x"})
    assert res.decision is Decision.APPROVED
    assert res.policy == "trust_mode"


def test_risky_action_respects_prompter(settings):
    settings.trust = False
    approvals = []
    gate_yes = ApprovalGate(settings, prompter=lambda n, a, r: approvals.append(n) or True)
    res = gate_yes.decide(_tool("write_file"), {"path": "/root/x", "content": "y"})
    assert res.decision is Decision.APPROVED and res.policy == "user_approved"

    gate_no = ApprovalGate(settings, prompter=lambda n, a, r: False)
    res2 = gate_no.decide(_tool("write_file"), {"path": "/root/x", "content": "y"})
    assert res2.decision is Decision.BLOCKED and res2.policy == "user_denied"
