import os
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import Mock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPELINE = os.path.join(ROOT, "synthenexa_pipeline")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

from core.context import PipelineContext  # noqa: E402
from core.schema import PromptCandidate  # noqa: E402
from pipeline.prompt_evolution.stages.mutation import MutationStage  # noqa: E402
from scripts.run_optimization import (  # noqa: E402
    _print_comparison_table,
    _resolve_model_list,
    _safe_name,
    _save_comparison_summary,
)
from services.llm_service import LLMService  # noqa: E402


class PipelineModelSwitchingTests(unittest.TestCase):
    @patch("services.llm_service.LLMFactory")
    def test_mutation_stage_accepts_services_for_different_models(self, factory_cls):
        for model_name in ["qwen2.5:3b", "cogito:3b"]:
            with self.subTest(model=model_name):
                adapter = Mock()
                adapter.generate.return_value = f"Improved prompt from {model_name}"
                adapter.base_url = "http://ollama.test"
                factory_cls.return_value.create.return_value = adapter

                service = LLMService(model=model_name, base_url="http://ollama.test")
                stage = MutationStage(llm_service=service, num_mutations=1)
                context = PipelineContext(
                    model_name=model_name,
                    run_id="test",
                    base_prompt=PromptCandidate(
                        id="prompt_0",
                        text="Write a Python function to solve the task.",
                    ),
                    tasks=[],
                )

                updated = stage.run(context)

                self.assertEqual(service.model, model_name)
                self.assertEqual(len(updated.candidate_prompts), 1)
                self.assertIn(model_name, updated.candidate_prompts[0].text)

    @patch("scripts.run_optimization._available_ollama_models")
    def test_available_model_resolution_excludes_large_and_unavailable_models(self, available_models):
        available_models.return_value = [
            "gemma4:latest",
            "qwen2.5-coder:32b",
            "cogito:3b",
            "nemotron-3-super:cloud",
            "qwen2.5:3b",
        ]
        args = Namespace(
            available_models=True,
            models="",
            model="qwen2.5:3b",
            exclude_models="qwen2.5-coder:32b,nemotron-3-super:cloud",
            ollama_base_url="http://ollama.test",
        )

        self.assertEqual(
            _resolve_model_list(args),
            ["gemma4:latest", "cogito:3b", "qwen2.5:3b"],
        )

    def test_model_list_resolution_preserves_single_model_default(self):
        args = Namespace(
            available_models=False,
            models="",
            model="qwen2.5:3b",
            exclude_models="qwen2.5-coder:32b,nemotron-3-super:cloud",
            ollama_base_url="http://ollama.test",
        )

        self.assertEqual(_resolve_model_list(args), ["qwen2.5:3b"])

    def test_explicit_model_list_excludes_duplicates_and_blocked_models(self):
        args = Namespace(
            available_models=False,
            models="qwen2.5:3b,nemotron-3-super:cloud,qwen2.5:3b,cogito:3b",
            model="nemotron-3-super:cloud",
            exclude_models="nemotron-3-super:cloud",
            ollama_base_url="http://ollama.test",
        )

        self.assertEqual(_resolve_model_list(args), ["qwen2.5:3b", "cogito:3b"])

    def test_safe_name_replaces_model_name_separators(self):
        self.assertEqual(_safe_name("qwen2.5:3b"), "qwen2.5_3b")

    def test_comparison_summary_ranks_models_by_score(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_pipeline = sys.modules["scripts.run_optimization"]._PIPELINE
            sys.modules["scripts.run_optimization"]._PIPELINE = temp_dir
            try:
                path = _save_comparison_summary(
                    "comparison",
                    [
                        {
                            "model": "cogito:3b",
                            "experiment": "comparison/cogito_3b",
                            "best_score": 0.2,
                            "best_prompt": "Prompt B",
                            "best_prompt_id": "b",
                            "reward_history": {1: 0.2},
                        },
                        {
                            "model": "qwen2.5:3b",
                            "experiment": "comparison/qwen2.5_3b",
                            "best_score": 0.8,
                            "best_prompt": "Prompt A",
                            "best_prompt_id": "a",
                            "reward_history": {1: 0.8},
                        },
                    ],
                )
                with open(path, "r", encoding="utf-8") as f:
                    summary = f.read()
            finally:
                sys.modules["scripts.run_optimization"]._PIPELINE = original_pipeline

        self.assertIn('"model": "qwen2.5:3b"', summary)
        self.assertLess(summary.index("qwen2.5:3b"), summary.index("cogito:3b"))

    def test_comparison_table_prints_without_reward_history_state(self):
        _print_comparison_table(
            [
                {
                    "model": "qwen2.5:3b",
                    "experiment": "comparison/qwen2.5_3b",
                    "best_score": 0.8,
                    "best_prompt": "Prompt A",
                    "best_prompt_id": "a",
                    "reward_history": {1: 0.8},
                }
            ]
        )


if __name__ == "__main__":
    unittest.main()
