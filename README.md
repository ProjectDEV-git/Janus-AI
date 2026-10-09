# Janus AI

An **autonomous, self-improving agent** that works with any model. Give it a goal;
it plans, acts, checks its own progress against evidence, and keeps going until
the goal is *verified done* or you stop it. By default Janus runs entirely on a
local model through [Ollama](https://ollama.com) — no cloud API required — but it
can equally drive any OpenAI-compatible API or Anthropic's Claude
([Choosing a model](#choosing-a-model)).

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
- **Sandbox** — shell commands, Python, and benchmark checks run inside the OS
  sandbox: [bubblewrap](https://github.com/containers/bubblewrap) on Linux,
  Seatbelt (`sandbox-exec`, built in) on macOS. The filesystem is read-only
  except the workspace, your home directory is hidden, and there is no network. Set `sandbox` in `janus.toml`
  (`auto` / `on` / `bwrap` / `seatbelt` / `off`).
- **Approval gate** — anything that can reach beyond the workspace pauses for your
  confirmation: reading, listing or writing files outside the workspace, running
  shell/Python *without* the sandbox, and every edit to Janus itself (a new tool
  or a source patch, with the code shown to you). Inside the workspace, and with
  the sandbox on, actions run freely.
- **Self-edits are fenced** — a source patch may only touch the one file it was
  shown (never `tests/`, which judge it), and its tests run in the sandbox.
- **Audit log** — every thought, tool call, and result is recorded to SQLite and
  a replayable JSONL transcript.

`--trust` widens what runs without a prompt; the kill switch and audit log stay
on regardless. Use it only when you're watching.

Janus runs on Linux, macOS and Windows. Windows has no sandbox backend
(`run_shell` uses bash if found, e.g. Git Bash, else PowerShell). Without a
sandbox (Windows, or Linux without bubblewrap), code execution is gated, so
`janus bench`, `janus improve` and the `janus train` A/B step refuse to run
model-written code unless you pass `--trust`. On Linux, install bubblewrap
(`apt install bubblewrap` or your distro's equivalent) to avoid that.

## Install

```bash
# 1. Install Ollama and pull the model (skip if you use a hosted model, see below)
ollama pull hf.co/HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive:IQ3_M

# 2. Install Janus
pip install -e ".[dev]"

# 3. Check connectivity + config
janus config
```

## Choosing a model

Janus speaks three protocols, which between them cover practically every model.
Set `provider` and `model` in `janus.toml` (or `JANUS_PROVIDER` / `JANUS_MODEL`),
and put API keys in the environment, not the file:

| Where the model runs | `provider` | `model` example | `api_base` | Key |
|---|---|---|---|---|
| Ollama (local, default) | `ollama` | any pulled tag, e.g. `qwen3:8b`, `llama3.1` | — (`ollama_host`) | — |
| OpenAI | `openai` | `gpt-4.1-mini` | — | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | any Claude model id | — | `ANTHROPIC_API_KEY` |
| OpenRouter | `openai` | `meta-llama/llama-3.3-70b-instruct` | `https://openrouter.ai/api/v1` | `JANUS_API_KEY` |
| Groq | `openai` | `llama-3.3-70b-versatile` | `https://api.groq.com/openai/v1` | `JANUS_API_KEY` |
| Google Gemini | `openai` | `gemini-2.5-flash` | `https://generativelanguage.googleapis.com/v1beta/openai` | `JANUS_API_KEY` |
| Mistral | `openai` | `mistral-large-latest` | `https://api.mistral.ai/v1` | `JANUS_API_KEY` |
| DeepSeek | `openai` | `deepseek-chat` | `https://api.deepseek.com/v1` | `JANUS_API_KEY` |
| LM Studio / vLLM / llama.cpp server | `openai` | whatever the server loaded | e.g. `http://localhost:1234/v1` | usually none |

`JANUS_API_KEY` works for every provider; `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
are used when it isn't set. Run `janus config` to check the connection.

Models differ in the options they accept. When a server rejects an optional
parameter (JSON mode, `temperature`, `seed`, `max_tokens`), Janus drops it and
retries, and remembers that for the rest of the run. It also copes with replies
wrapped in prose, code fences, or `<think>` blocks from reasoning models.
For Ollama, `num_ctx` (default 8192) sets the context window; Ollama's own default
is small enough to cut off the agent's history.

Self-training (below) produces local models served by Ollama, so `janus train`
needs `provider = "ollama"`. With a hosted provider, `janus dataset` still exports
your winning runs as training data.

## Usage

```bash
janus task "write and test a function that reverses a string"  # run a goal
janus improve      # run a self-improvement cycle against the benchmark
janus bench        # measure the current benchmark scorecard (each task in a fresh workspace)
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
