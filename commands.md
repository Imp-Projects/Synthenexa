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

## 1. Interactive & Ensemble Execution Commands

### Interactive model selection
```bash
python synthenexa_pipeline/scripts/run_optimization.py --interactive --generations 1 --mutations 1 --experiment interactive_smoke
```

### Interactive multi-model ensemble run
```bash
python synthenexa_pipeline/scripts/run_optimization.py --interactive --ensemble --generations 2 --mutations 2 --experiment interactive_ensemble_smoke
```

### Explicit multi-model ensemble run (qwen and nemotron)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --models qwen3-coder-next:cloud,nemotron-3-super:cloud --ensemble --ensemble-strategy mean --generations 3 --mutations 5 --experiment explicit_ensemble_run
```

### Ensemble run with a different dataset
```bash
python synthenexa_pipeline/scripts/run_optimization.py --interactive --ensemble --dataset Dataset/benchmark_mbpp_micro.jsonl --generations 2 --mutations 3 --experiment dataset_ensemble_smoke
```

> [!NOTE]
> The commands above constitute the primary workflow for prompt refinement using an ensemble of LLMs.

---

## Reference: Example Model Commands (Optional)

These examples illustrate how individual models can be included within an ensemble. They are provided for reference only and are **not** part of the main workflow.

### OpenAI model in ensemble
```bash
python synthenexa_pipeline/scripts/run_optimization.py --models openai:gpt-4o-mini,qwen3-coder-next:cloud --ensemble --ensemble-strategy mean --generations 2 --mutations 2 --experiment openai_ensemble_example
```

### Gemini model in ensemble
```bash
python synthenexa_pipeline/scripts/run_optimization.py --models gemini:gemini-2.0-flash,qwen3-coder-next:cloud --ensemble --ensemble-strategy mean --generations 2 --mutations 2 --experiment gemini_ensemble_example
```

### Common flag combinations (reference)
- `--generations <N>` – Number of generations per model.
- `--mutations <N>` – Number of prompt mutations per generation.
- `--ensemble-strategy mean|median|vote` – How ensemble outputs are aggregated.
- `--dataset <path>` – Specify a custom dataset.
- `--experiment <name>` – Tag the run for tracking.
