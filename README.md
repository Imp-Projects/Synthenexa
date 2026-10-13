# Synthenexa

Synthenexa is a prototype that tackles a practical problem in working with large language models: **prompts that work well on one model often fail on another**, and writing effective, universal instructions by hand is slow, inconsistent, and hard to optimise systematically.

Instead of relying on manual prompt engineering or over-fitting to a single provider, Synthenexa automates the discovery of robust, model-agnostic prompts. It starts with a base prompt, asks multiple LLMs to independently generate improved variations, cross-tests how well every variation performs across all models on a real coding task, and uses the combined results to produce an even better prompt in the next round. The process repeats until it converges on instructions that perform consistently well regardless of the underlying LLM architecture.

The core objective is to **automatically refine user prompts for better results across LLMs**. Single-model runs are supported for debugging and smoke testing, but the primary workflow is cross-model ensemble evaluation.

---

## How the prompt refinement works

Each refinement cycle follows five steps:

```
Base Prompt
    │
    ▼
① Mutate — Every LLM in the ensemble independently generates N variations of the current best prompt
    │
    ▼
② Execute — Every candidate prompt is pooled and cross-tested across ALL models to generate Python code
    │
    ▼
③ Evaluate — The generated code is run against a test suite (pass / fail)
    │
    ▼
④ Score — Each candidate is assigned a unified reward by aggregating its performance across all models
    │
    ▼
⑤ Select — The prompt with the best universal score becomes the seed for the next cycle
    │
    └──► Repeat for N generations → Best Universal Prompt Found
```

This loop runs for a configurable number of *generations*. With each generation, the system discovers prompt instructions that more reliably guide different LLMs toward correct, working code, preventing the prompt from overfitting to any single model.

---

## How the architecture supports this approach

The system is built around a simple principle: **universal prompt quality is measured objectively by the downstream output it produces across diverse architectures**, not by human judgment.

| Concern | How Synthenexa addresses it |
|---|---|
| **Model-agnostic discovery** | Pools candidate prompts from multiple LLMs, ensuring a diverse range of "ideas" on how to instruct a model |
| **Ensemble cross-testing** | Every candidate is evaluated on every model simultaneously, mathematically guaranteeing the winning prompt is battle-tested across different architectures |
| **Objective scoring** | Uses automated test execution (EvalPlus) to score candidates on real coding tasks — no subjective evaluation |
| **Intelligent selection** | A configurable *policy* (greedy or epsilon-greedy) decides which prompt to carry forward, balancing exploitation of the best known prompt with occasional exploration of lower-ranked ones |
| **Provider flexibility** | Works across OpenAI, Google Gemini, and local open-weight models seamlessly from a single CLI command |
| **Reproducibility** | Every run is saved to disk (prompts, scores, evaluations, snapshots) so results can be compared and reproduced |

---

## Key features of the prototype

- **Multi-provider support** — Run experiments with OpenAI, Google Gemini, or local open-weight models via Ollama seamlessly
- **Ensemble cross-testing** — Discover universal prompts that generalise across LLM architectures by evaluating candidates on multiple models simultaneously
- **Configurable ensemble strategies** — Aggregate multi-model scores using different strategies (mean, min, max) to select for consistency, highest floor, or highest peak performance
- **Pluggable selection policies** — Switch between `greedy` (always pick the best) and `epsilon_greedy` (occasionally explore) without changing pipeline code
- **Multi-model comparison** — Run the same experiment across several models simultaneously and compare which provider produces the best prompts
- **Results Web UI** — A built-in FastAPI dashboard (`uvicorn api:app --host 0.0.0.0 --port 8000`) for instantly browsing experiment outputs in the browser
- **Zero-config SQLite storage** — Experiment data is persisted to a local SQLite database alongside file artifacts — no external database setup required

---

## Quick links

- [QUICKSTART.md](QUICKSTART.md) — Set up the project and run your first experiment
- [ARCHITECTURE.md](ARCHITECTURE.md) — How the system is structured and why
- [commands.md](commands.md) — Command cheat-sheet for all supported providers

## Evidence of Prompt Refinement

The project demonstrates improvement through a matched, held-out comparison:

1. The original `prompt_0` is evaluated on every selected model.
2. Mutations and selection use only train/validation tasks.
3. The frozen refined prompt and `prompt_0` are both evaluated on the same held-out test tasks.
4. SQLite and per-run artifact manifests preserve models, seeds, task IDs, scores, and prompt lineage.
5. Analysis reports absolute and relative improvement per model and in aggregate.

Run the reproducible protocol in [commands.md](commands.md), then inspect the
generated held-out comparison and analysis plots. Runs with missing baseline or
test provenance are marked incomplete rather than being assigned an estimated score.
