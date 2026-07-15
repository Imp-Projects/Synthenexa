# Running the Synthenexa Pipeline

This guide will help you run the reinforcement-learning prompt optimization pipeline. The pipeline uses an LLM to repeatedly generate, test, and score new system prompts to find the best instructions for writing Python code.

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

Run the pipeline using the optimization script. Here is the standard command:
```bash
python scripts/run_optimization.py --generations 3 --mutations 5
```
*This runs 3 evolution generations, creating 5 mutated prompt variations per generation.*

## 3. Useful Command-Line Options

You can customize the pipeline run using several flags:

- `--generations <int>`: How many cycles of evolution to run (Default: 3).
- `--mutations <int>`: How many new prompt variations to generate per cycle (Default: 5).
- `--model <name>`: The Ollama model to use for mutations and code generation (Default: `nemotron-3-super:cloud`).
- `--policy <name>`: How the pipeline selects the best prompt. Use `greedy` to always pick the highest score, or `epsilon_greedy` to occasionally explore random prompts (Default: `greedy`).

**Example:** Run a quick test with a smaller model:
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
