# Synthenexa

> **Prototype Objective:** Design a system that automatically refines user prompts for better results across LLMs.

---

## What is this project?

Synthenexa is a prototype that tackles a practical problem in working with large language models: **the quality of the output depends heavily on how the prompt is written**, and writing effective prompts by hand is slow, inconsistent, and hard to optimise systematically.

Instead of relying on manual prompt engineering, Synthenexa automates the refinement process. It starts with a base prompt, generates improved variations, tests how well each variation performs on a real task, and uses the results to produce an even better prompt in the next round. The process repeats until it converges on the most effective prompt it can find.

---

## How the prompt refinement works

Each refinement cycle follows five steps:

```
Base Prompt
    │
    ▼
① Mutate — An LLM generates N variations of the current best prompt
    │
    ▼
② Execute — Each candidate prompt is used to generate Python code solutions
    │
    ▼
③ Evaluate — The generated code is run against a test suite (pass / fail)
    │
    ▼
④ Score — Each candidate is assigned a reward based on how many tests it passed
    │
    ▼
⑤ Select — The best-scoring prompt becomes the seed for the next cycle
    │
    └──► Repeat for N generations → Best Prompt Found
```

This loop runs for a configurable number of *generations*. With each generation, the system discovers prompt instructions that more reliably guide the LLM toward correct, working code.

---

## How the architecture supports the objective

The system is built around a simple principle: **prompt quality is measured objectively by the downstream output it produces**, not by human judgment.

| Concern | How Synthenexa addresses it |
|---|---|
| **Exploration** | Generates multiple prompt candidates per cycle (configurable via `--mutations`) rather than committing to a single rewrite |
| **Objective scoring** | Uses automated test execution (EvalPlus) to score candidates on real coding tasks — no subjective evaluation |
| **Intelligent selection** | A configurable *policy* (greedy or epsilon-greedy) decides which prompt to carry forward, balancing exploitation of the best known prompt with occasional exploration of lower-ranked ones |
| **Provider flexibility** | Works across OpenAI, Google Gemini, and local Ollama models — the same pipeline runs against any LLM without code changes |
| **Reproducibility** | Every run is saved to disk (prompts, scores, evaluations, snapshots) so results can be compared and reproduced |

---

## Key features of the prototype

- **Multi-provider support** — Run experiments with OpenAI (GPT-4o), Google Gemini, or local open-weight models via Ollama from a single CLI command
- **Pluggable selection policies** — Switch between `greedy` (always pick the best) and `epsilon_greedy` (occasionally explore) without changing pipeline code
- **Configurable evolution** — Control the number of generations and mutations per run to trade off speed against thoroughness
- **Multi-model comparison** — Run the same experiment across several models simultaneously and compare which provider produces the best prompts
- **Structured experiment output** — Each run produces a complete artifact directory: the best prompt found, per-generation candidate lists, evaluation results, and full state snapshots

---

## Quick links

- [QUICKSTART.md](QUICKSTART.md) — Set up the project and run your first experiment
- [ARCHITECTURE.md](ARCHITECTURE.md) — How the system is structured and why
- [commands.md](commands.md) — Command cheat-sheet for all supported providers
