"""
tests/pipeline/test_ensemble_pipeline.py

Unit tests for the ensemble prompt optimization pipeline.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPELINE = os.path.join(ROOT, "synthenexa_pipeline")
if PIPELINE not in sys.path:
    sys.path.insert(0, PIPELINE)

from core.context import PipelineContext  # noqa: E402
from core.schema import EvalResult, ExecutionResult, PromptCandidate  # noqa: E402
from pipeline.prompt_evolution.stages.ensemble_scoring import EnsembleScoringStage  # noqa: E402
from pipeline.prompt_evolution.stages.ensemble_execution import EnsembleExecutionStage  # noqa: E402


def _make_context(**kwargs) -> PipelineContext:
    defaults = dict(
        model_name="ensemble",
        run_id="test",
        base_prompt=PromptCandidate(id="prompt_0", text="Write a function."),
        tasks=[{"task_id": "t1", "entry_point": "solve", "test": "assert solve(1)==1"}],
    )
    defaults.update(kwargs)
    return PipelineContext(**defaults)


class EnsembleScoringStageTests(unittest.TestCase):
    def _context_with_ensemble_scores(self, scores: dict) -> PipelineContext:
        ctx = _make_context()
        ctx.ensemble_scores = scores
        ctx.candidate_prompts = [
            PromptCandidate(id=pid, text=f"Prompt {pid}")
            for pid in scores
        ]
        return ctx

    def test_mean_aggregation(self):
        ctx = self._context_with_ensemble_scores({
            "prompt_a": {"ModelA": 0.8, "ModelB": 0.4},
            "prompt_b": {"ModelA": 0.6, "ModelB": 0.6},
        })
        stage = EnsembleScoringStage(strategy="mean")
        ctx = stage.run(ctx)

        self.assertAlmostEqual(ctx.scores["prompt_a"], 0.6)
        self.assertAlmostEqual(ctx.scores["prompt_b"], 0.6)

    def test_min_aggregation(self):
        ctx = self._context_with_ensemble_scores({
            "prompt_a": {"ModelA": 0.9, "ModelB": 0.1},
            "prompt_b": {"ModelA": 0.7, "ModelB": 0.7},
        })
        stage = EnsembleScoringStage(strategy="min")
        ctx = stage.run(ctx)

        self.assertAlmostEqual(ctx.scores["prompt_a"], 0.1)
        self.assertAlmostEqual(ctx.scores["prompt_b"], 0.7)

    def test_max_aggregation(self):
        ctx = self._context_with_ensemble_scores({
            "prompt_a": {"ModelA": 0.3, "ModelB": 0.9},
        })
        stage = EnsembleScoringStage(strategy="max")
        ctx = stage.run(ctx)

        self.assertAlmostEqual(ctx.scores["prompt_a"], 0.9)

    def test_median_aggregation(self):
        ctx = self._context_with_ensemble_scores({
            "prompt_a": {"ModelA": 0.1, "ModelB": 0.9, "ModelC": 0.7},
            "prompt_b": {"ModelA": 0.2, "ModelB": 0.4, "ModelC": 0.8, "ModelD": 0.6},
        })
        stage = EnsembleScoringStage(strategy="median")
        ctx = stage.run(ctx)

        self.assertAlmostEqual(ctx.scores["prompt_a"], 0.7)
        self.assertAlmostEqual(ctx.scores["prompt_b"], 0.5)

    def test_vote_aggregation(self):
        ctx = self._context_with_ensemble_scores({
            "prompt_a": {"ModelA": 0.9, "ModelB": 0.1, "ModelC": 0.4, "ModelD": 0.2},
            "prompt_b": {"ModelA": 0.8, "ModelB": 0.95, "ModelC": 0.7, "ModelD": 0.3},
            "prompt_c": {"ModelA": 0.2, "ModelB": 0.4, "ModelC": 0.6, "ModelD": 0.9},
        })
        stage = EnsembleScoringStage(strategy="vote")
        ctx = stage.run(ctx)

        self.assertAlmostEqual(ctx.scores["prompt_a"], 0.25)
        self.assertAlmostEqual(ctx.scores["prompt_b"], 0.5)
        self.assertAlmostEqual(ctx.scores["prompt_c"], 0.25)

    def test_best_prompt_updated_by_ensemble_scoring(self):
        ctx = self._context_with_ensemble_scores({
            "prompt_low":  {"ModelA": 0.2, "ModelB": 0.2},
            "prompt_high": {"ModelA": 0.9, "ModelB": 0.9},
        })
        stage = EnsembleScoringStage(strategy="mean")
        ctx = stage.run(ctx)

        self.assertIsNotNone(ctx.best_prompt)
        self.assertEqual(ctx.best_prompt.id, "prompt_high")
        self.assertAlmostEqual(ctx.best_score, 0.9)

    def test_empty_model_scores_yields_zero(self):
        ctx = self._context_with_ensemble_scores({
            "prompt_a": {},
        })
        stage = EnsembleScoringStage(strategy="mean")
        ctx = stage.run(ctx)

        self.assertEqual(ctx.scores["prompt_a"], 0.0)

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            EnsembleScoringStage(strategy="bogus")  # type: ignore

    def test_selection_does_not_overwrite_global_best(self):
        from pipeline.prompt_evolution.stages.selection import SelectionStage
        from pipeline.prompt_evolution.policies.greedy import GreedySelectionPolicy

        ctx = self._context_with_ensemble_scores({
            "prompt_a": {"ModelA": 0.4, "ModelB": 0.4},
            "prompt_b": {"ModelA": 0.9, "ModelB": 0.9},
        })
        ctx.scores = {"prompt_a": 0.4, "prompt_b": 0.9}
        ctx.best_prompt = PromptCandidate(id="global_best", text="Best so far")
        ctx.best_score = 0.95

        stage = SelectionStage(policy=GreedySelectionPolicy())
        ctx = stage.run(ctx)

        self.assertEqual(ctx.best_prompt.id, "global_best")
        self.assertAlmostEqual(ctx.best_score, 0.95)
        self.assertEqual(ctx.current_prompt.id, "prompt_b")


class EnsembleExecutionStageTests(unittest.TestCase):
    def _make_mut_stage(self, model_name: str) -> MagicMock:
        """Return a MutationStage mock that adds one candidate to context."""
        def side_effect(ctx: PipelineContext) -> PipelineContext:
            safe = model_name.replace(":", "_").replace("-", "_")
            ctx.candidate_prompts = [
                PromptCandidate(
                    id=f"prompt_0_gen1_m0",
                    text=f"Prompt from {model_name}",
                )
            ]
            return ctx

        mock = MagicMock()
        mock.run.side_effect = side_effect
        return mock

    def _make_exec_stage(self) -> MagicMock:
        """Return an ExecutionStage mock that fills generated_codes."""
        def side_effect(ctx: PipelineContext) -> PipelineContext:
            for c in ctx.candidate_prompts:
                ctx.generated_codes[c.id] = [
                    ExecutionResult(prompt_id=c.id, task_id="t1", output="def solve(x): return x")
                ]
            return ctx

        mock = MagicMock()
        mock.run.side_effect = side_effect
        return mock

    def _make_eval_stage(self, score: float) -> MagicMock:
        """Return an EvaluationStage mock that writes a fixed EvalResult."""
        def side_effect(ctx: PipelineContext) -> PipelineContext:
            total = int(round(score * 10))
            for c in ctx.candidate_prompts:
                ctx.eval_results[c.id] = EvalResult(
                    prompt_id=c.id,
                    passed=total,
                    failed=10 - total,
                    total=10,
                )
            return ctx

        mock = MagicMock()
        mock.run.side_effect = side_effect
        return mock

    def test_ensemble_scores_contains_all_models(self):
        ctx = _make_context()
        ctx.current_prompt = ctx.base_prompt
        ctx.generation = 1

        pipelines = [
            ("ModelA", self._make_mut_stage("ModelA"), self._make_exec_stage(), self._make_eval_stage(0.8)),
            ("ModelB", self._make_mut_stage("ModelB"), self._make_exec_stage(), self._make_eval_stage(0.4)),
        ]
        stage = EnsembleExecutionStage(model_pipelines=pipelines)
        ctx = stage.run(ctx)

        for pid, model_scores in ctx.ensemble_scores.items():
            self.assertIn("ModelA", model_scores)
            self.assertIn("ModelB", model_scores)

        self.assertIn("ModelA", ctx.eval_results_by_model)
        self.assertIn("ModelB", ctx.eval_results_by_model)
        for model_name, results in ctx.eval_results_by_model.items():
            for eval_result in results.values():
                self.assertEqual(eval_result.evaluator_model, model_name)

    def test_candidate_count_equals_num_models_times_mutations(self):
        ctx = _make_context()
        ctx.current_prompt = ctx.base_prompt
        ctx.generation = 1

        pipelines = [
            ("ModelA", self._make_mut_stage("ModelA"), self._make_exec_stage(), self._make_eval_stage(0.5)),
            ("ModelB", self._make_mut_stage("ModelB"), self._make_exec_stage(), self._make_eval_stage(0.5)),
        ]
        stage = EnsembleExecutionStage(model_pipelines=pipelines)
        ctx = stage.run(ctx)

        # Each model produced 1 mutation → 2 total unique candidates
        self.assertEqual(len(ctx.candidate_prompts), 2)

    def test_empty_pipeline_raises(self):
        with self.assertRaises(ValueError):
            EnsembleExecutionStage(model_pipelines=[])


if __name__ == "__main__":
    unittest.main()
