# Janus AI

A **local, autonomous, self-improving agent**. Give it a goal; it plans, acts,
checks its own progress against evidence, and keeps going until the goal is
*verified done* or you stop it. Janus runs entirely on a local model through
[Ollama](https://ollama.com) — no cloud API required.

Janus can improve itself across four layers: its prompts and memory, tools it
writes for itself, patches to its own source code, and training **its own model**
(a LoRA lineage descended from the base Gemma weights). Every self-change is a **propose → test → keep-or-revert**
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

## Becoming its own model (self-training)

Janus doesn't just tune prompts — it can train **its own model**, a versioned
lineage descended from the base Gemma weights that retrains on what actually
worked:

```
base Gemma safetensors ─┐
                        ├─ LoRA fine-tune on Janus's own winning transcripts
winning run transcripts ┘
        │
        ▼  merge → convert → quantize (GGUF) → `ollama create janus:vN`
        ▼  A/B benchmark janus:vN vs current ──win──► adopt   ──lose──► keep on record
```

```bash
janus train            # mine winning runs, train a LoRA, build janus:vN, A/B it
janus train --adopt    # ...and switch to it if it beats the current model
janus model list       # see the lineage (version, parent, status, adopted, success)
janus model use janus:v2   # roll forward/back to any version
janus improve --train --adopt   # self-improve AND self-train in one cycle
```

**Important facts:**

- **You cannot train the `IQ3_M` GGUF** — that is a ~3-bit *inference* copy with
  no gradients. Training starts from the full-precision base weights. Set
  `base_model_id` in `janus.toml` to the matching base (default
  `google/gemma-3n-E2B`); the GGUF stays as the thing Ollama runs.
- **Training needs a GPU** and the extra deps: `pip install -e ".[finetune]"`.
  Without them, `janus train` still mines the dataset and tells you exactly what
  it would run — it degrades gracefully instead of failing.
- **GGUF export needs [llama.cpp](https://github.com/ggerganov/llama.cpp)** built
  locally (for `convert_hf_to_gguf.py` + the quantizer). Point `llama_cpp_dir`
  at it, or leave blank to autodetect `~/llama.cpp`, `/opt/llama.cpp`, `./llama.cpp`.
- **A new version is adopted only if it wins the A/B benchmark** — same
  propose → test → keep-or-revert discipline as every other self-change, and
  adoption is a gated/opt-in action. Old versions stay in the lineage so you can
  always roll back with `janus model use`.

### No GPU? Train on Google Colab (free T4)

You don't need a local GPU. Janus builds the training dataset locally, you train
on a free Colab T4, and import the finished model back:

```bash
# 1. On your machine: mine your winning runs into a dataset
janus dataset                  # writes datasets/sft.jsonl

# 2. In Colab: open notebooks/janus_train_colab.ipynb, set Runtime -> T4 GPU,
#    upload sft.jsonl, run all cells. It LoRA-trains and exports a GGUF zip.

# 3. Back on your machine: import the downloaded model into the lineage
unzip janus_model.zip -d janus_model
janus model import janus_model/*.gguf --adopt
janus model list
```

`janus model import` writes the Ollama Modelfile, runs `ollama create`, records
the version in the lineage, and (with `--adopt`) activates it. Skip `--adopt` to
A/B it first (`janus bench` per tag, then `janus model use janus:vN`). The Colab
notebook uses [Unsloth](https://github.com/unslothai/unsloth), which fits Gemma
LoRA on a free T4 and exports GGUF directly — no local llama.cpp needed.

## Configuration

Edit `janus.toml` (model, workspace root, budgets, approval policy) or override
any field with `JANUS_*` environment variables (e.g. `JANUS_MODEL=...`).

## Layout

```
janus/            package: config, llm, memory, agent, approval, tools, selfimprove, benchmark, cli
  tools/dynamic/  tools Janus writes for itself
  selfimprove/    the four self-improvement strategies
  train/          self-training: dataset mining, LoRA trainer, GGUF export, lineage
notebooks/        janus_train_colab.ipynb — train on a free Colab GPU
tests/            pytest suite (approval gate, budgets, revert-on-failure)
benchmark/tasks/  the efficiency benchmark suite
```

## License

MIT.
