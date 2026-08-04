# Synthenexa — Architecture

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Architecture & Design Decisions](#3-architecture--design-decisions)
4. [Development Container](#4-development-container)
5. [Getting Started](#5-getting-started)
6. [Path to Production](#6-path-to-production)

---

## 1. Project Overview

Synthenexa is a **multi-model prompt-evolution pipeline** with RL-inspired selection. It discovers universal, model-agnostic prompts by treating optimization as a cross-architecture search problem:

- A **seed prompt** is mutated by an ensemble of LLMs into a diverse pool of candidates.
- Every candidate is cross-tested on every model in the ensemble to generate Python code against a benchmark dataset.
- The generated code is executed and scored (pass@1).
- Scores are aggregated across models (e.g., mean, min, max) to calculate a unified reward.
- A **policy** (greedy or epsilon-greedy) selects the prompt with the best universal score as the seed for the next generation.
- The loop repeats for *N* generations, converging on prompts that perform consistently well across different LLMs.

This is sometimes described as a DAG (Directed Acyclic Graph) pipeline **within each generation** — data flows strictly from mutation → execution → evaluation → scoring → selection. Across generations the best prompt is fed back as the next seed, making the outer loop cyclic/iterative, which is the RL-style aspect.

---

## 2. Directory Structure

```
Synthenexa/
├── .devcontainer/                  # Dev Container definition
│   ├── devcontainer.json           # VS Code Dev Container config
│   ├── docker-compose.yml          # App service
│   └── Dockerfile                  # Python 3.11 base image
│
├── synthenexa_pipeline/            # All application code lives here
│   ├── core/                       # Domain primitives (no I/O)
│   │   ├── config.py               # Central config — loads .env, Pydantic models
│   │   ├── context.py              # PipelineContext — shared state carrier
│   │   ├── models.py               # Core data structures (PromptCandidate, etc.)
│   │   ├── schema.py               # Lightweight schema types (EvalResult, etc.)
│   │   ├── reward.py               # Reward computation (pass@1 + penalty)
│   │   └── metrics.py              # Derived metrics helpers
│   │
│   ├── pipeline/                   # Evolution loop logic
│   │   └── prompt_evolution/
│   │       ├── workflow.py         # Outer N-generation loop controller
│   │       ├── stages/             # One class per DAG stage
│   │       │   ├── mutation.py     # LLM generates new prompt candidates
│   │       │   ├── execution.py    # LLM generates code from each candidate
│   │       │   ├── evaluation.py   # EvalPlus runs code against test suite
│   │       │   ├── scoring.py      # Converts eval results to scalar rewards
│   │       │   ├── selection.py    # Policy picks the next seed prompt
│   │       │   ├── ensemble_execution.py  # Cross-model execution & evaluation
│   │       │   └── ensemble_scoring.py    # Aggregates per-model scores
│   │       └── policies/           # Pluggable selection strategies
│   │           ├── greedy.py       # Always pick highest-scored candidate
│   │           ├── epsilon_greedy.py # Occasionally explore a random candidate
│   │           └── registry.py     # Maps policy name strings to classes
│   │
│   ├── services/                   # I/O and external integrations
│   │   ├── llm/                    # LLM abstraction layer
│   │   │   ├── base.py             # BaseLLMAdapter interface
│   │   │   ├── factory.py          # LLMFactory — resolves provider from model name
│   │   │   ├── registry.py         # ModelRegistry + ModelSpec — tiered model list
│   │   │   ├── openai_adapter.py   # OpenAI Chat Completions
│   │   │   ├── gemini_adapter.py   # Google Gemini API
│   │   │   ├── ollama_adapter.py   # Local Ollama runtime
│   │   │   ├── cloud_adapter.py    # Ollama cloud-tagged models (:cloud suffix)
│   │   │   └── health.py           # Provider reachability checks
│   │   ├── llm_service.py          # High-level LLM façade used by pipeline stages
│   │   ├── eval_service.py         # EvalPlus wrapper (code execution + scoring)
│   │   ├── dataset_service.py      # Loads .jsonl benchmark datasets
│   │   ├── database.py             # SQLAlchemy engine + session for SQLite
│   │   ├── storage.py              # Persists artifacts (file or SQLite backend)
│   │   └── tracker.py              # Lightweight per-generation reward logger
│   │
│   ├── api.py                      # FastAPI results viewer (serves on :8000)
│   ├── templates/
│   │   └── index.html              # Web UI for browsing experiment outputs
│   │
│   ├── scripts/
│   │   └── run_optimization.py     # CLI entry point — parses args, wires everything up
│   │
│   ├── Dataset/                    # Benchmark datasets (.jsonl)
│   └── experiments/artifacts/      # Per-run output (gitignored)
│
├── tests/                          # pytest test suite
├── .env                            # Local secrets (gitignored)
├── requirements.txt
├── pyproject.toml                  # Ruff linter/formatter config
├── docker-compose.yml
├── QUICKSTART.md
└── ARCHITECTURE.md                 # This file
```

---

## 3. Architecture & Design Decisions

### 3.1 Layered separation: `core` → `pipeline` → `services` → `scripts`

The code is split into four layers with strict dependency direction:

```
scripts/        (CLI, wiring)
    ↓ uses
pipeline/       (evolution logic, DAG stages, policies)
    ↓ uses
services/       (LLM, eval, storage — all I/O)
    ↓ uses
core/           (config, data models, reward — no I/O)
```

`core` has no dependencies on any other internal layer. This means reward functions, config structs, and data models can be tested in isolation without mocking any I/O.

### 3.2 `PipelineContext` as a state carrier

Rather than passing dozens of arguments between pipeline stages, each stage receives and returns a single `PipelineContext` object. This object accumulates shared state (current prompt, candidates, scores, reward history, etc.) across the generation loop. Stages are composable and can be reordered or replaced without changing their signatures.

### 3.3 Provider-neutral LLM layer

The `services/llm/` package decouples the pipeline from any specific LLM provider:

```
model string   →   LLMFactory   →   BaseLLMAdapter
"openai:gpt-4o-mini"   →   OpenAIAdapter
"gemini:gemini-2.0-flash"  →   GeminiAdapter
"nemotron-3-super:cloud"   →   CloudAdapter
"cogito:3b"              →   OllamaAdapter
```

The routing key is encoded in the model name prefix (`openai:`, `gemini:`, or bare name → Ollama). Adding a new provider requires only a new adapter class and a factory branch — no changes to the pipeline stages.

`ModelRegistry` maintains a tiered catalogue of all supported models (Tier 1 OpenAI → Tier 4 local Ollama), enabling `--available-models` to auto-discover configured providers from `.env` at runtime.

### 3.4 The Prompt Evolution Pipeline (DAG Stages)

The pipeline executes a strict Directed Acyclic Graph (DAG) of stages during each generation. Because the primary goal is discovering **universal, model-agnostic prompts**, the default architectural flow is the multi-model ensemble mode.

**Phase 1 — Global Mutation (Pooling Ideas)**
`EnsembleExecutionStage (Part 1)`: Each model in the ensemble generates *N* prompt variants independently from the current seed. All variants are gathered into a single **Master Prompt Pool**. For example, 3 models generating 5 mutations each yields 15 unique candidate prompts.

**Phase 2 — Cross-Execution & Evaluation (The Matrix)**
`EnsembleExecutionStage (Part 2)` & `EvaluationStage`: Every prompt in the master pool is tested on **every** model in the ensemble. Each model uses all 15 prompts to generate code for every benchmark task, which is then run through EvalPlus. This creates a dense score matrix (e.g., 15 prompts × 3 models).

**Phase 3 — Scoring & Selection**
`EnsembleScoringStage`: Aggregates each prompt's per-model scores into a single combined reward using a configurable strategy (`mean`, `min`, `median`, `max`, or `vote`). The default `mean` strategy ensures the selected prompt must perform **consistently well across all models**, not just over-index on one. `median` is useful when one model is noisy, `vote` is a ranking-based consensus strategy that counts how many models ranked the prompt first, and `min` remains the strictest robustness check.
`SelectionStage`: Applies the active policy (e.g., greedy) to pick the highest-scoring universal prompt as the next seed.

This cross-testing approach prevents prompt overfitting to a single model and mathematically guarantees that the winning prompt has been battle-tested across multiple LLM architectures.

### 3.5 Single-Model Execution (Subset)

While Synthenexa is designed for multi-model discovery, it can also run in a single-model mode for rapid testing or budget-constrained experiments. In this mode, the matrix collapses to a 1D vector:
- `MutationStage`: A single LLM generates all candidates.
- `ExecutionStage`: The same LLM generates code from those candidates.
- `ScoringStage`: Evaluates and scores without needing an aggregation strategy.

### 3.6 Pluggable selection policies

The `pipeline/policies/` package follows a simple strategy pattern. Both `GreedyPolicy` and `EpsilonGreedyPolicy` inherit from `Policy` and implement a single `select(candidates, scores)` method. `PolicyRegistry` maps CLI strings to policy classes, making it trivial to add new strategies (e.g. UCB, softmax sampling) without modifying the workflow controller.

### 3.7 Results Web UI

A lightweight FastAPI application (`synthenexa_pipeline/api.py`) serves a browser-based dashboard for visual analysis of experiment results. It reads the canonical experiment artifact store, distills the latest run into a small summary, and renders a curated set of charts that show score progression, winning prompts, and prompt-pool trends. After running the pipeline, start the web server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The dashboard reads from `experiments/artifacts/` and renders results via `templates/index.html` at `http://localhost:8000`. This keeps the UI grounded in the real experiment outputs while still presenting them as a concise, presentation-friendly analysis layer rather than a raw artifact browser.

---

## 4. Development Container

### Why a Dev Container?

Synthenexa depends on Python 3.11, several system libraries (`build-essential` for compiled dependencies), and a specific `PYTHONPATH` configuration. Without a container, a new developer would need to manually replicate this environment, which is error-prone and time-consuming.

The `.devcontainer/` directory provides a fully automated, reproducible setup using the [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) specification.

### What it provides

| Concern | How it's handled |
|---|---|
| Python 3.11 | Pre-installed in `mcr.microsoft.com/devcontainers/python:3.11` |
| System deps | `build-essential`, `curl` installed in the Dockerfile |
| pip deps | `pip install -r requirements.txt` runs automatically via `postCreateCommand` |
| `PYTHONPATH` | Set to `/workspaces/Synthenexa/synthenexa_pipeline` via `remoteEnv` in `devcontainer.json` |
| Ollama bridge | `host.docker.internal:22000` routes to an Ollama process on the developer's host machine |
| Editor config | Ruff formatter (format-on-save), Pylance, debugpy all pre-installed |
| Consistency | Every developer gets the same environment regardless of host OS |

### Ensemble strategy requirements

The ensemble stage needs at least 2 models because each prompt is cross-evaluated across the model set. Strategy-specific guidance:

- `mean`: Best general-purpose default. Use when you want a smooth signal from all model scores.
- `min`: Best for strict robustness checks. Use when a prompt must not fail badly on any model.
- `median`: Best when one model may be an outlier. Works best when there are at least 3 model scores per prompt.
- `max`: Best for exploratory runs or specialist prompts. It can hide weaknesses because it only reflects the strongest model.
- `vote`: Best when you want a ranking-based consensus signal. Each model votes for its highest-scoring candidate prompt, and the score is the fraction of model votes received. It is most meaningful when you have several prompt variations so models can make a real choice, and it works best with 3 or more models.

As a practical lower bound, the ensemble is usable with 2 models and 1 or more mutations, but `vote` becomes much more informative once the run includes multiple mutations per model and a broader prompt pool.

### Container architecture

```
Host machine
├── VS Code (Dev Containers extension)
├── Ollama (running on host, port 22000)
│
└── Docker
    └── app container  (Python 3.11, all deps, workspace mounted)
        ├── SQLite DB      (experiments/synthenexa.db — zero-config)
        └── FastAPI UI     (localhost:8000 — results viewer)
```

The workspace directory is bind-mounted into the container, so code edits in VS Code are reflected immediately without rebuilding.

---

## 5. Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) v24+
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd Synthenexa
```

### Step 2 — Create your `.env` file

The project reads API keys and runtime settings from `.env` in the project root. This file is gitignored and must be created manually. At minimum, set the key for whichever LLM provider you plan to use:

```dotenv
# Cloud providers (set at least one)
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=...

# Local Ollama (if running locally)
OLLAMA_BASE_URL=http://host.docker.internal:22000
OLLAMA_MODEL=nemotron-3-super:cloud

```

### Step 3 — Open in Dev Container

1. Open the project folder in VS Code.
2. When prompted, click **"Reopen in Container"** (or run `Dev Containers: Reopen in Container` from the Command Palette).
3. The container builds (~2–3 min on first run) and installs all dependencies automatically.

### Step 4 — Run a smoke test

From the integrated terminal inside the container:

```bash
python synthenexa_pipeline/scripts/run_optimization.py \
  --model gemini:gemini-2.0-flash \
  --generations 1 --mutations 1 \
  --experiment smoke_test
```

Results are written to `synthenexa_pipeline/experiments/artifacts/<run-id>/`.

### Step 5 — Run the test suite

```bash
# Fast unit tests
pytest -q tests/services/test_llm_factory.py \
          tests/services/test_llm_service.py \
          tests/services/test_ollama_adapter.py \
          tests/pipeline/test_pipeline_model_switching.py

# Full suite
pytest -q
```

> **Without Docker:** If you prefer a local setup, create a Python 3.11 venv, run `pip install -r requirements.txt`, and set `PYTHONPATH=$(pwd)/synthenexa_pipeline`. See [QUICKSTART.md](QUICKSTART.md) for details.

---

## 6. Path to Production

The current architecture is optimised for research and experimentation. Moving to a production-grade system would involve several significant changes.

### Asynchronous execution

The pipeline currently runs stages synchronously in a single process. At scale, mutation and code execution are the bottlenecks — each requires one or more LLM calls. A production version should execute candidates concurrently using an async task queue (e.g. Celery, ARQ, or Python's `asyncio` with bounded concurrency). `MAX_WORKERS` in `core/config.py` is already a placeholder for this.

### Observability

Production runs would benefit from:
- **Structured logging** — replace `print()` with a structured logger (e.g. `structlog`) emitting JSON to a log aggregator
- **Distributed tracing** — instrument LLM call latency and stage timing (OpenTelemetry)
- **Metrics** — export reward history, pass rates, and model latency to a metrics store (Prometheus/Grafana)

### API surface

The project currently offers two interfaces: the CLI (`run_optimization.py`) for running experiments, and a lightweight FastAPI server (`api.py`) for browsing results. A production API layer would extend this to allow triggering runs programmatically, integrating with CI/CD, and expanding the dashboard into a full experiment comparison tool.

### Security & secrets management

`.env` files work for development but are unsuitable for production. API keys should be stored in a secrets manager (AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager) and injected at runtime, not loaded from disk.

### Model cost management

Running multi-model experiments against cloud APIs can become expensive quickly. A production system should track token usage per run, enforce per-experiment budget caps, and prefer local Ollama models for mutation steps (which are lower-stakes) while reserving cloud models for final evaluation.

### Containerised deployment

The Dev Container definition is a good foundation but is development-only (`sleep infinity` command, workspace bind-mount). For production, the pipeline would need a proper entrypoint Dockerfile — a single-stage or multi-stage build that copies only the application code, installs dependencies without dev tools, and runs `run_optimization.py` as the container command.
