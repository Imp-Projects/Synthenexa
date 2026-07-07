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

## 1. Nemotron Model Test Commands (Default)

### Single-model smoke test
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model nemotron-3-super:cloud --generations 1 --mutations 1 --experiment smoke_nemotron
```

### Single-model exploratory run (epsilon-greedy)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model nemotron-3-super:cloud --generations 2 --mutations 3 --policy epsilon_greedy --epsilon 0.2 --experiment epsilon_smoke_nemotron
```

### Run with a different dataset
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model nemotron-3-super:cloud --dataset Dataset/benchmark_mbpp_micro.jsonl --generations 1 --mutations 1 --experiment dataset_smoke_nemotron
```

### Larger realistic run (Single model)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model nemotron-3-super:cloud --generations 3 --mutations 5 --experiment realistic_run_nemotron
```

---

## 2. OpenAI Model Test Commands

### Single-model smoke test
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model openai:gpt-4o-mini --generations 1 --mutations 1 --experiment smoke_openai
```

### Single-model exploratory run (epsilon-greedy)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model openai:gpt-4o-mini --generations 2 --mutations 3 --policy epsilon_greedy --epsilon 0.2 --experiment epsilon_smoke_openai
```

### Run with a different dataset
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model openai:gpt-4o-mini --dataset Dataset/benchmark_mbpp_micro.jsonl --generations 1 --mutations 1 --experiment dataset_smoke_openai
```

### Larger realistic run (Single model)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model openai:gpt-4o-mini --generations 3 --mutations 5 --experiment realistic_run_openai
```

---

## 3. Gemini Model Test Commands

### Single-model smoke test
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model gemini:gemini-2.0-flash --generations 1 --mutations 1 --experiment smoke_gemini
```

### Single-model exploratory run (epsilon-greedy)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model gemini:gemini-2.0-flash --generations 2 --mutations 3 --policy epsilon_greedy --epsilon 0.2 --experiment epsilon_smoke_gemini
```

### Run with a different dataset
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model gemini:gemini-2.0-flash --dataset Dataset/benchmark_mbpp_micro.jsonl --generations 1 --mutations 1 --experiment dataset_smoke_gemini
```

### Larger realistic run (Single model)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model gemini:gemini-2.0-flash --generations 3 --mutations 5 --experiment realistic_run_gemini
```

---

## Multi-Model Execution Commands

### Multi-model comparison run (across all available providers)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --available-models --generations 1 --mutations 1 --experiment multi_model_smoke
```

### Large-scale realistic run (OpenAI, Gemini, and Nemotron)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --models openai:gpt-4o,gemini:gemini-2.0-flash,nemotron-3-super:cloud --generations 5 --mutations 10 --experiment large_scale_multi_model
```
