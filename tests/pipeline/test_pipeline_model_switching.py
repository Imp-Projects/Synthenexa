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
from core.schema import EvalResult, PromptCandidate  # noqa: E402
from pipeline.prompt_evolution.stages.mutation import MutationStage, prompt_hash  # noqa: E402
from pipeline.prompt_evolution.stages.execution import ExecutionStage  # noqa: E402
from services.llm_cache import LLMResponseCache  # noqa: E402
from scripts.run_optimization import (  # noqa: E402
    _print_comparison_table,
    _resolve_model_list,
    _run_repeated_experiments,
    _safe_name,
    _save_comparison_summary,
    _validate_ollama_experiment,
)
from services.llm_service import LLMService  # noqa: E402


class PipelineModelSwitchingTests(unittest.TestCase):
    def test_execution_cache_avoids_repeated_model_calls(self):
        service = Mock()
        service.model = "qwen2.5:3b"
        service.generate.return_value = "def solve(x): return x"
        stage = ExecutionStage(service)
        context = PipelineContext(
            model_name="qwen2.5:3b", run_id="test", tasks=[
                {"task_id": "t1", "prompt": "identity", "entry_point": "solve", "test": "assert solve(1)==1"}
            ],
            candidate_prompts=[PromptCandidate(id="p1", text="Prompt")],
        )

        stage.run(context)
        context.candidate_prompts = [PromptCandidate(id="p2", text="Prompt")]
        stage.run(context)

        self.assertEqual(service.generate.call_count, 1)
        self.assertEqual(context.call_stats["generation_calls"], 1)
        self.assertEqual(context.call_stats["generation_cache_hits"], 1)

    def test_persistent_cache_reuses_response_across_stage_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = LLMResponseCache(temp_dir)
            first_service = Mock(model="qwen2.5:3b")
            first_service.generate.return_value = "def solve(x): return x"
            context = PipelineContext(
                model_name="qwen2.5:3b", run_id="one", tasks=[
                    {"task_id": "t1", "prompt": "identity", "entry_point": "solve", "test": "assert solve(1)==1"}
                ],
                candidate_prompts=[PromptCandidate(id="p1", text="Prompt")],
            )
            ExecutionStage(first_service, cache=cache).run(context)

            second_service = Mock(model="qwen2.5:3b")
            second_context = PipelineContext(
                model_name="qwen2.5:3b", run_id="two", tasks=context.tasks,
                candidate_prompts=[PromptCandidate(id="different_id", text="Prompt")],
            )
            ExecutionStage(second_service, cache=cache).run(second_context)

            first_service.generate.assert_called_once()
            second_service.generate.assert_not_called()
            self.assertEqual(second_context.call_stats["persistent_cache_hits"], 1)

    def test_mutation_roles_and_duplicate_suppression(self):
        adapter = Mock()
        adapter.generate.side_effect = ["Same prompt", "  same   prompt  ", "A different prompt"]
        adapter.base_url = "http://ollama.test"
        with patch("services.llm_service.LLMFactory") as factory_cls:
            factory_cls.return_value.create.return_value = adapter
            service = LLMService(model="qwen2.5:3b", base_url="http://ollama.test")
            stage = MutationStage(llm_service=service, num_mutations=3)
            context = PipelineContext(
                model_name="qwen2.5:3b",
                run_id="test",
                generation=1,
                base_prompt=PromptCandidate(id="prompt_0", text="Base prompt"),
                current_prompt=PromptCandidate(id="prompt_0", text="Base prompt"),
                tasks=[],
            )
            updated = stage.run(context)

        self.assertEqual(len(updated.candidate_prompts), 2)
        self.assertEqual(updated.call_stats["mutation_calls"], 3)
        self.assertEqual(updated.candidate_prompts[0].mutation_role, "clarity and instruction structure")
        self.assertEqual(updated.candidate_prompts[1].mutation_role, "output-format reliability and constraint discipline")
        self.assertEqual(updated.candidate_prompts[0].prompt_hash, prompt_hash("Same prompt"))

    def test_later_generation_mutation_receives_parent_feedback(self):
        adapter = Mock()
        adapter.generate.return_value = "Improved prompt"
        adapter.base_url = "http://ollama.test"
        with patch("services.llm_service.LLMFactory") as factory_cls:
            factory_cls.return_value.create.return_value = adapter
            service = LLMService(model="qwen2.5:3b", base_url="http://ollama.test")
            stage = MutationStage(llm_service=service, num_mutations=1)
            parent = PromptCandidate(id="parent", text="Parent prompt")
            context = PipelineContext(
                model_name="qwen2.5:3b", run_id="test", generation=2,
                current_prompt=parent, base_prompt=parent, tasks=[],
                scores={"parent": 0.5},
                eval_results={"parent": EvalResult(prompt_id="parent", passed=1, failed=1, total=2, errors=["edge"])},
            )
            stage.run(context)

        request = adapter.generate.call_args.kwargs["prompt"]
        self.assertIn("Parent score: 0.5000", request)
        self.assertIn("failure(s)", request)

    def test_ollama_guardrails_accept_configured_models_and_limits(self):
        args = Namespace(generations=3, mutations=5)
        _validate_ollama_experiment(
            args,
            ["nemotron-3-ultra:cloud", "gpt-oss:120b-cloud", "neeoggl/glm-5.2:latest"],
        )

    def test_ollama_guardrails_reject_non_ollama_models(self):
        args = Namespace(generations=2, mutations=3)
        with self.assertRaisesRegex(ValueError, "Ollama-only"):
            _validate_ollama_experiment(args, ["openai:gpt-4o"])

    def test_ollama_guardrails_reject_expensive_run_sizes(self):
        with self.assertRaises(ValueError):
            _validate_ollama_experiment(Namespace(generations=4, mutations=3), ["cogito:3b"])
        with self.assertRaises(ValueError):
            _validate_ollama_experiment(Namespace(generations=2, mutations=6), ["cogito:3b"])

    @patch("scripts.run_optimization._run_single_experiment")
    def test_repetitions_use_distinct_names_and_incremented_seeds(self, run_single):
        run_single.side_effect = lambda args, model, run_id, experiment: {
            "model": model,
            "experiment": experiment,
            "best_score": 0.5,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            module = sys.modules["scripts.run_optimization"]
            original_pipeline = module._PIPELINE
            setattr(module, "_PIPELINE", temp_dir)
            try:
                args = Namespace(
                    experiment="batch",
                    seed=10,
                    repetitions=2,
                    ensemble=False,
                )
                results = _run_repeated_experiments(args, ["model-a"])
            finally:
                setattr(module, "_PIPELINE", original_pipeline)

        self.assertEqual([call.args[0].seed for call in run_single.call_args_list], [10, 11])
        self.assertEqual([call.args[3] for call in run_single.call_args_list], ["batch_rep1", "batch_rep2"])
        self.assertEqual([result["experiment"] for result in results], ["batch_rep1", "batch_rep2"])

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
            no_openai=True,
            no_gemini=True,
            interactive=False,
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
