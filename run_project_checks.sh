#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo "Virtual environment not found at .venv. Activate it first or create one." >&2
  exit 1
fi

source .venv/bin/activate

echo "==> Running unit tests"
pytest -q tests/services/test_llm_factory.py; pytest -q tests/services/test_llm_service.py; pytest -q tests/services/test_ollama_adapter.py; pytest -q tests/pipeline/test_pipeline_model_switching.py

echo "\n==> Running full test suite"
pytest -q

echo "\n==> Single-model smoke test"
python synthenexa_pipeline/scripts/run_optimization.py --model nemotron-3-super:cloud --generations 1 --mutations 1 --experiment smoke_single

echo "\n==> Single-model exploratory run"
python synthenexa_pipeline/scripts/run_optimization.py --model nemotron-3-super:cloud --generations 2 --mutations 3 --policy epsilon_greedy --epsilon 0.2 --experiment epsilon_smoke

echo "\n==> Multi-model comparison run"
python synthenexa_pipeline/scripts/run_optimization.py --models qwen2.5:3b,cogito:3b,nemotron-3-super:cloud --generations 1 --mutations 1 --experiment multi_model_smoke

echo "\n==> Run across all locally available Ollama models"
python synthenexa_pipeline/scripts/run_optimization.py --available-models --generations 1 --mutations 1 --experiment available_models_smoke

echo "\n==> Run with a different dataset"
python synthenexa_pipeline/scripts/run_optimization.py --model nemotron-3-super:cloud --dataset Dataset/benchmark_mbpp_micro.jsonl --generations 1 --mutations 1 --experiment dataset_smoke

echo "\n==> Larger realistic run"
python synthenexa_pipeline/scripts/run_optimization.py --model nemotron-3-super:cloud --generations 3 --mutations 5 --experiment realistic_run

echo "\nAll project checks completed."
