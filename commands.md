# Setup & Testing

## Activate virtual environment
```bash
source .venv/bin/activate
```

## Unit tests (Fast)
```bash
pytest -q tests/services/test_llm_factory.py tests/services/test_llm_service.py tests/services/test_ollama_adapter.py tests/pipeline/test_pipeline_model_switching.py
```

## Full test suite
```bash
pytest -q
```

---

# Pipeline Execution Commands

## Primary Workflow: Cross-Model Ensemble Evaluation

Synthenexa is designed to refine prompts by testing them across multiple LLMs. The commands below reflect the main workflow of the project: generate prompt variants, cross-evaluate them across models, and select prompts that generalise well.

### Interactive model selection
```bash
python synthenexa_pipeline/scripts/run_optimization.py --interactive --ensemble --generations 1 --mutations 5 --experiment interactive_ensemble_smoke
```

### Explicit multi-model ensemble run
```bash
python synthenexa_pipeline/scripts/run_optimization.py --models qwen3-coder-next:cloud,nemotron-3-super:cloud --ensemble --ensemble-strategy mean --generations 3 --mutations 5 --experiment explicit_ensemble_run
```

### Ensemble run with OpenAI + Gemini + local models
```bash
python synthenexa_pipeline/scripts/run_optimization.py --available-models --ensemble --ensemble-strategy mean --generations 3 --mutations 5 --experiment available_models_ensemble
```

### Ensemble run with a different dataset
```bash
python synthenexa_pipeline/scripts/run_optimization.py --interactive --ensemble --dataset Dataset/benchmark_mbpp_micro.jsonl --generations 2 --mutations 3 --experiment dataset_ensemble_smoke
```

> [!NOTE]
> The commands above constitute the primary workflow for prompt refinement using an ensemble of LLMs.

---

## Development Smoke Tests

Single-model runs are useful for validating the pipeline, debugging one provider, or checking a small change quickly. They are not the project’s core workflow, but they are helpful during development.

### Single-model smoke test
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model qwen3-coder-next:cloud --generations 1 --mutations 1 --experiment single_model_smoke
```

### OpenAI provider smoke test
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model openai:gpt-4o-mini --generations 1 --mutations 1 --experiment openai_smoke
```

### Gemini provider smoke test
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model gemini:gemini-2.0-flash --generations 1 --mutations 1 --experiment gemini_smoke
```

### Common flag combinations
- `--generations <N>` – Number of generations per model.
- `--mutations <N>` – Number of prompt mutations per generation.
- `--ensemble-strategy mean|min|median|max|vote` – How ensemble outputs are aggregated.
- `--dataset <path>` – Specify a custom dataset.
- `--experiment <name>` – Tag the run for tracking.
