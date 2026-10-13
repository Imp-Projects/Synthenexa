# Running the Synthenexa Pipeline

This guide will help you run the prompt optimization pipeline. The core workflow is cross-model prompt refinement: multiple LLMs generate prompt variants, the variants are tested across models, and the system keeps the prompt that generalises best.

## 1. Prerequisites

Before running the pipeline, ensure you have:
1. **Ollama running** in the background with the required models (e.g., `nemotron-3-super:cloud`).
2. **Activated your virtual environment** from the root of the project:
   ```bash
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

## 2. Running the Pipeline

Navigate to the `synthenexa_pipeline` directory if you aren't already there:
```bash
cd synthenexa_pipeline
```

Run the pipeline using the optimization script. The primary workflow is ensemble mode:
```bash
python scripts/run_optimization.py --models qwen3-coder-next:cloud,nemotron-3-super:cloud --ensemble --ensemble-strategy mean --generations 3 --mutations 5
```
*This runs 3 evolution generations, creating 5 mutated prompt variations per generation and cross-evaluates them across the selected models.*

## 3. Useful Command-Line Options

You can customize the pipeline run using several flags:

- `--generations <int>`: How many cycles of evolution to run (Default: 3).
- `--mutations <int>`: How many new prompt variations to generate per cycle (Default: 5).
- `--models <list>`: Comma-separated list of models to use for the primary ensemble workflow.
- `--ensemble`: Enables cross-model prompt refinement, which is the core project objective.
- `--ensemble-strategy <name>`: How scores are combined across models. Use `mean`, `min`, `median`, `max`, `vote`, or `robust` (70% mean + 30% weakest-model score).
- `--model <name>`: Single-model development smoke test mode. Useful for debugging one provider, but not the main workflow.
- `--policy <name>`: How the pipeline selects the next prompt. Use `greedy` to always pick the highest score, or `epsilon_greedy` to occasionally explore random prompts (Default: `greedy`).

### Ensemble strategy guidance

All ensemble strategies require at least 2 models because the ensemble stage performs cross-evaluation.

- `mean`: Best default for smooth optimization. Works well as soon as you have 2 or more models and at least 2 prompt variants in the pool, but it is most informative when you generate several mutations per model so one noisy score does not dominate the average.
- `min`: Conservative choice for robustness. Works with 2 or more models, but it is most meaningful when each model has had a real chance to disagree, so use multiple mutations per model rather than a single candidate.
- `median`: Robust to one noisy or extreme model. Works with 2 or more models, but is most useful when you have at least 3 model scores for each prompt so the middle value is not just an average of two scores.
- `max`: Specialist or exploratory choice. Works with 2 or more models, but can over-favor prompts that only fit one model well, so it is best used when you explicitly want to discover model-specific wins.
- `vote`: Ranking-based consensus choice. Each model votes for the candidate prompt it scored highest, and the output is the share of model votes received by that candidate. It requires multiple candidate variations or mutations so the models have something meaningful to rank, and it works best with 3 or more models.

Practical rule of thumb: if you are only running 1 mutation per model, the ensemble still works, but `vote` becomes much less informative because there are too few prompt variations to compare.

**Example:** Run a quick single-model smoke test for development:
```bash
python scripts/run_optimization.py --generations 2 --mutations 3 --model qwen3-coder-next:cloud
```

## 4. Where to Find Your Results

As the pipeline runs, it will print its progress to the terminal, including a visual **Reward History** table at the very end.

All detailed logs and artifacts are automatically saved in the `experiments/artifacts/` directory, organized by your Run ID. Inside this folder you will find:
- **`best_prompt.json`**: The highest-scoring prompt discovered during the run.
- **`prompts_gen_X.jsonl`**: The candidate prompts generated during generation X.
- **`evaluations_gen_X.jsonl`**: The pass/fail results for every candidate.
- **`snapshot_gen_X.json`**: The complete state of the pipeline at that generation.

The Web UI is a presentation layer on top of those artifacts: it reads the latest run summary, then renders a small set of charts that highlight score progression, winning prompts, and prompt-pool trends. It is intentionally focused on visual proof of improvement rather than raw artifact browsing.
