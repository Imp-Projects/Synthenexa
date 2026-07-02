# Activate virtual environment
source .venv/bin/activate

# Unit tests (Fast)
pytest -q tests/services/test_llm_factory.py tests/services/test_llm_service.py tests/services/test_ollama_adapter.py tests/pipeline/test_pipeline_model_switching.py

# Full test suite
pytest -q

# Single-model smoke test
python synthenexa_pipeline/scripts/run_optimization.py --model openai:gpt-4o-mini --generations 1 --mutations 1 --experiment smoke_single

# Single-model exploratory run (epsilon-greedy)
python synthenexa_pipeline/scripts/run_optimization.py --model openai:gpt-4o-mini --generations 2 --mutations 3 --policy epsilon_greedy --epsilon 0.2 --experiment epsilon_smoke

# Multi-model comparison run (across all available providers)
python synthenexa_pipeline/scripts/run_optimization.py --available-models --generations 1 --mutations 1 --experiment multi_model_smoke

# Run with a different dataset
python synthenexa_pipeline/scripts/run_optimization.py --model openai:gpt-4o-mini --dataset Dataset/benchmark_mbpp_micro.jsonl --generations 1 --mutations 1 --experiment dataset_smoke

# Larger realistic run (Single model)
python synthenexa_pipeline/scripts/run_optimization.py --model openai:gpt-4o-mini --generations 3 --mutations 5 --experiment realistic_run

# Large-scale realistic run (OpenAI, Gemini, and Nemotron)
python synthenexa_pipeline/scripts/run_optimization.py --models openai:gpt-4o,gemini:gemini-2.0-flash,nemotron-3-super:cloud --generations 5 --mutations 10 --experiment large_scale_multi_model
