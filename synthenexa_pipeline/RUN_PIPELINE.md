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
- `--ensemble-strategy <name>`: How scores are combined across models. Use `mean`, `min`, or `max`.
- `--model <name>`: Single-model development smoke test mode. Useful for debugging one provider, but not the main workflow.
- `--policy <name>`: How the pipeline selects the next prompt. Use `greedy` to always pick the highest score, or `epsilon_greedy` to occasionally explore random prompts (Default: `greedy`).

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
