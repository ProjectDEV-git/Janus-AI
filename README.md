# Janus AI

A **local, autonomous, self-improving agent**. Give it a goal; it plans, acts,
checks its own progress against evidence, and keeps going until the goal is
*verified done* or you stop it. Janus runs entirely on a local model through
[Ollama](https://ollama.com) — no cloud API required.

Janus can improve itself across four layers: its prompts and memory, tools it
writes for itself, patches to its own source code, and (optionally) LoRA
fine-tuning. Every self-change is a **propose → test → keep-or-revert**
transaction gated on a benchmark, so a change is kept only if it measurably
helps, and reverted automatically otherwise.

## Why it's safe to let it run

Janus acts on your real filesystem and can run shell/Python, so the brakes live
in Janus, not in the model:

- **Kill switch** — `janus stop`, Ctrl+C, or the `.janus_stop` file halt it
  cleanly at the next step.
- **Budgets** — hard caps on iterations, wall-clock time, and tokens per task.
- **Approval gate** — destructive or out-of-workspace actions (deleting/overwriting
  outside the workspace, `rm`/`sudo`-class shell, edits to Janus's own code)
  pause for your confirmation. Read-only and in-workspace actions run freely.
- **Audit log** — every thought, tool call, and result is recorded to SQLite and
  a replayable JSONL transcript.

`--trust` widens what runs without a prompt; the kill switch and audit log stay
on regardless. Use it only when you're watching.

## Install

```bash
# 1. Install Ollama and pull the model
ollama pull hf.co/HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive:IQ3_M

# 2. Install Janus
pip install -e ".[dev]"

# 3. Check connectivity + config
janus config
```

## Usage

```bash
janus task "write and test a function that reverses a string"  # run a goal
janus improve      # run a self-improvement cycle against the benchmark
janus bench        # measure the current benchmark scorecard
janus stop         # halt a running Janus
janus log          # inspect the audit trail
janus replay <id>  # replay a past task transcript
```

## Configuration

Edit `janus.toml` (model, workspace root, budgets, approval policy) or override
any field with `JANUS_*` environment variables (e.g. `JANUS_MODEL=...`).

## Layout

```
janus/            package: config, llm, memory, agent, approval, tools, selfimprove, benchmark, cli
  tools/dynamic/  tools Janus writes for itself
  selfimprove/    the four self-improvement strategies
tests/            pytest suite (approval gate, budgets, revert-on-failure)
benchmark/tasks/  the efficiency benchmark suite
```

## License

MIT.
