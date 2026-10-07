"""Self-training pipeline: the parts that run without a GPU are fully tested."""
import json

from janus.memory import Memory
from janus.train.dataset import build_dataset
from janus.train.gguf import write_modelfile
from janus.train.pipeline import run_training_cycle
from janus.train.registry import ModelRegistry


def _seed_finished_run(mem: Memory, goal: str = "do a thing") -> int:
    run_id = mem.start_run(goal)
    mem.log_event(run_id, "thought", {"iter": 1, "text": "write the file"})
    mem.log_event(run_id, "action", {"iter": 1, "tool": "write_file",
                                     "args": {"path": "x.py", "content": "print(1)"}})
    mem.log_event(run_id, "observation", {"iter": 1, "ok": True, "output": "wrote 8 chars"})
    mem.log_event(run_id, "thought", {"iter": 2, "text": "done"})
    mem.log_event(run_id, "decision", {"iter": 2, "finish": True, "evidence": "file written"})
    mem.finish_run(run_id, status="finished", iterations=2, tokens=20, result="file written")
    return run_id


def test_dataset_build_from_finished_run(settings):
    mem = Memory(settings.db_path)
    _seed_finished_run(mem)
    stats = build_dataset(mem, settings)
    assert stats.examples == 1
    assert stats.assistant_turns == 2  # one action + one finish
    lines = stats.path.read_text().strip().splitlines()
    ex = json.loads(lines[0])
    roles = [m["role"] for m in ex["messages"]]
    assert roles[0] == "system" and roles[1] == "user"
    assert "assistant" in roles
    # The assistant turn reproduces the action JSON protocol.
    assert any('"action"' in m["content"] for m in ex["messages"] if m["role"] == "assistant")
    mem.close()


def test_dataset_excludes_unfinished(settings):
    mem = Memory(settings.db_path)
    rid = mem.start_run("failed one")
    mem.log_event(rid, "thought", {"iter": 1, "text": "try"})
    mem.finish_run(rid, status="budget", iterations=1, tokens=5, result="gave up")
    stats = build_dataset(mem, settings)
    assert stats.examples == 0
    mem.close()


def test_registry_lineage(settings):
    reg = ModelRegistry(settings.db_path)
    assert reg.next_version() == 1
    assert reg.current_tag(default="base:model") == "base:model"
    reg.register(version=1, tag="janus:v1", parent_tag="base:model",
                 base_model="google/gemma-3n-E2B", dataset_hash="abc", adapter_dir="/a")
    assert reg.next_version() == 2
    reg.adopt("janus:v1")
    assert reg.current_tag(default="base:model") == "janus:v1"
    rows = reg.list()
    assert len(rows) == 1 and rows[0].adopted
    reg.close()


def test_write_modelfile_is_pure(settings, tmp_path):
    gguf = tmp_path / "janus-v1.Q4_K_M.gguf"
    gguf.write_bytes(b"\x00")
    mf = write_modelfile(gguf, "janus:v1", settings, parent_tag="base:model")
    text = mf.read_text()
    assert f"FROM {gguf.name}" in text
    assert "lineage:" in text and "janus:v1" in text


def test_pipeline_skips_below_min_examples(settings):
    settings.train_min_examples = 5
    mem = Memory(settings.db_path)
    _seed_finished_run(mem)  # only 1 example
    outcome = run_training_cycle(settings, mem, adopt=False, console=None)
    assert outcome.status == "skipped"
    mem.close()


def test_pipeline_dataset_only_without_gpu(settings):
    settings.train_min_examples = 1
    mem = Memory(settings.db_path)
    _seed_finished_run(mem)
    outcome = run_training_cycle(settings, mem, adopt=False, console=None)
    # No GPU/deps in CI -> stops after building the dataset.
    assert outcome.status == "dataset_only"
    assert "dataset ready" in outcome.detail
    mem.close()
