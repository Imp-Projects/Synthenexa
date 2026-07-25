# 🚀 Synthenexa — Quick Start Guide

Synthenexa is an **iterative LLM prompt-optimization system** that automatically discovers universal, model-agnostic instructions by cross-testing them across multiple LLMs:

```
Seed Prompt → Mutated Prompts → Cross-Model Code Gen → Evaluation (EvalPlus) → Refined Universal Prompt
```

---

## How the Services Work Together

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐
│  LLM Service │    │  Eval Service│    │  Storage / Tracker   │
│  (adapters)  │───▶│  (EvalPlus)  │───▶│  (file or sqlite)    │
└──────────────┘    └──────────────┘    └──────────────────────┘
        │                                          │
        ▼                                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    run_optimization.py                       │
│    Pool Mutations → Cross-Test → Score → Select Best Prompt  │
└──────────────────────────────────────────────────────────────┘
```

| Component | Description |
|---|---|
| `synthenexa_pipeline/scripts/run_optimization.py` | Main entry point — orchestrates the full evolution loop |
| `synthenexa_pipeline/services/llm/` | Provider-agnostic LLM adapters (OpenAI, Gemini, Ollama, Groq, etc.) |
| `synthenexa_pipeline/services/eval_service.py` | Runs generated code against test cases via [EvalPlus](https://github.com/evalplus/evalplus) |
| `synthenexa_pipeline/services/storage.py` | Writes experiment artifacts to the filesystem or SQLite |
| `synthenexa_pipeline/core/` | Shared models, config, reward functions, and metrics |
| `synthenexa_pipeline/pipeline/` | Prompt evolution strategies (greedy, epsilon-greedy) |
| **Ollama** | Optional — run open-weight models locally (e.g. `qwen3-coder-next:cloud`) |

---

## Prerequisites

**Dev Container (recommended):** Docker Desktop (v24+) and the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) VS Code extension.

**Local setup:** Python 3.11, and optionally [Ollama](https://ollama.ai/).

---

## Setup

### Option A — Dev Container (Recommended)

1. Open the project in VS Code.
2. When prompted, click **"Reopen in Container"**, or run `Dev Containers: Reopen in Container` from the Command Palette (`Ctrl+Shift+P`).
3. The container builds Python 3.11 and runs `pip install -r requirements.txt` automatically (~2–3 min on first run).

The container pre-configures `PYTHONPATH` and the Ollama bridge (`host.docker.internal:22000`) — no manual environment setup needed.

### Option B — Local Setup

```bash
# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set PYTHONPATH (add to .bashrc / .zshrc to persist)
export PYTHONPATH="$(pwd)/synthenexa_pipeline"
```

> Experiment artifacts are written to `synthenexa_pipeline/experiments/artifacts/` by default.

---

## Running the Pipeline

### Smoke test — verify everything works

```bash
# Gemini
python synthenexa_pipeline/scripts/run_optimization.py \
  --model gemini:gemini-2.0-flash --generations 1 --mutations 1 --experiment smoke_gemini

# OpenAI
python synthenexa_pipeline/scripts/run_optimization.py \
  --model openai:gpt-4o-mini --generations 1 --mutations 1 --experiment smoke_openai

# Ollama (local)
python synthenexa_pipeline/scripts/run_optimization.py \
  --model nemotron-3-super:cloud --generations 1 --mutations 1 --experiment smoke_ollama
```

### Standard run

```bash
python synthenexa_pipeline/scripts/run_optimization.py \
  --model gemini:gemini-2.0-flash --generations 3 --mutations 5 --experiment my_first_run
```

### Epsilon-greedy exploration

```bash
python synthenexa_pipeline/scripts/run_optimization.py \
  --model openai:gpt-4o-mini --generations 2 --mutations 3 \
  --policy epsilon_greedy --epsilon 0.2 --experiment exploratory_run
```

### Ensemble cross-testing (Multi-model discovery)

```bash
python synthenexa_pipeline/scripts/run_optimization.py \
  --models openai:gpt-4o,gemini:gemini-2.0-flash,nemotron-3-super:cloud \
  --ensemble --ensemble-strategy mean \
  --generations 5 --mutations 10 --experiment multi_model_discovery
```

### Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model <name>` | `nemotron-3-super:cloud` | Single LLM (`provider:model-name`) |
| `--models <list>` | — | Comma-separated list for multi-model runs |
| `--available-models` | — | Auto-discover all models configured in `.env` |
| `--generations <int>` | `3` | Number of evolution cycles |
| `--mutations <int>` | `5` | Prompt variations per cycle |
| `--policy <name>` | `greedy` | `greedy` or `epsilon_greedy` |
| `--epsilon <float>` | `0.1` | Exploration rate (epsilon-greedy only) |
| `--dataset <path>` | default MBPP | Path to a `.jsonl` benchmark file |
| `--experiment <name>` | auto-generated | Label for this run's artifacts |

### Where to find results

```
synthenexa_pipeline/experiments/artifacts/<run-id>/
├── best_prompt.json          ← Highest-scoring prompt found
├── prompts_gen_X.jsonl       ← Candidate prompts for generation X
├── evaluations_gen_X.jsonl   ← Pass/fail results per candidate
└── snapshot_gen_X.json       ← Full pipeline state at generation X
```

### Viewing results in the browser

After running an experiment, start the built-in results viewer:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` to browse experiment outputs interactively.

---

## Useful Commands

### Tests

```bash
# Fast unit tests
pytest -q tests/services/test_llm_factory.py \
          tests/services/test_llm_service.py \
          tests/services/test_ollama_adapter.py \
          tests/pipeline/test_pipeline_model_switching.py

# Full test suite
pytest -q
```

### Code quality

```bash
ruff check .    # lint
ruff format .   # format
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'services'`**
`PYTHONPATH` is not set. Run `export PYTHONPATH="$(pwd)/synthenexa_pipeline"` (or `PYTHONPATH=synthenexa_pipeline pytest -q` for tests).

**Ollama times out**
Ensure Ollama is running (`ollama serve`) and that `OLLAMA_BASE_URL` in your `.env` matches the port in use (default `22000`). Pull the model first: `ollama pull qwen3-coder-next:cloud`.

**`AuthenticationError` for a cloud provider**
The API key in `.env` is wrong or expired — verify it on the provider's dashboard and ensure there are no trailing spaces.

**Dev Container fails to build**
Run `docker info` to confirm Docker is running, then use **Dev Containers: Rebuild Container** from the Command Palette. Docker needs at least 4 GB RAM and 10 GB disk.

---

## Tips

- **Start small** — always run `--generations 1 --mutations 1` first to verify your setup before a full experiment.
- **Free local runs** — `ollama pull qwen3-coder-next:cloud` + `--model qwen3-coder-next:cloud` requires no API keys.
- See [`commands.md`](commands.md) for a full provider-specific command cheat-sheet and [`synthenexa_pipeline/RUN_PIPELINE.md`](synthenexa_pipeline/RUN_PIPELINE.md) for deeper pipeline documentation.
