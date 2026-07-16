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

## 1. Default Model (qwen3-coder-next:cloud) Test Commands

### Single-model smoke test
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model qwen3-coder-next:cloud --generations 1 --mutations 1 --experiment smoke_qwen
```

### Single-model exploratory run (epsilon-greedy)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model qwen3-coder-next:cloud --generations 2 --mutations 3 --policy epsilon_greedy --epsilon 0.2 --experiment epsilon_smoke_qwen
```

### Run with a different dataset
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model qwen3-coder-next:cloud --dataset Dataset/benchmark_mbpp_micro.jsonl --generations 1 --mutations 1 --experiment dataset_smoke_qwen
```

### Larger realistic run (Single model)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --model qwen3-coder-next:cloud --generations 3 --mutations 5 --experiment realistic_run_qwen
```

---

## 2. Interactive & Ensemble Execution Commands

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

---

## 3. OpenAI Model Test Commands

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

## 4. Gemini Model Test Commands

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

## 5. Multi-Model Execution Commands (No Ensemble)

### Multi-model comparison run (across all available providers)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --available-models --generations 1 --mutations 1 --experiment multi_model_smoke
```

### Large-scale realistic run (OpenAI, Gemini, and Qwen)
```bash
python synthenexa_pipeline/scripts/run_optimization.py --models openai:gpt-4o,gemini:gemini-2.0-flash,qwen3-coder-next:cloud --generations 5 --mutations 10 --experiment large_scale_multi_model
```
