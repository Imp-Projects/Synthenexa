"""
pipeline/prompt_evolution/stages/evaluation.py — Run unit tests on generated code.

For each prompt candidate:
  - Collects code outputs from context.generated_codes
  - Pairs them with the corresponding tasks (which carry entry_point + test harness)
  - Calls EvalService.evaluate() which runs the MBPP check(candidate) harness
  - Stores a rich EvalResult (with syntax_errors count) in context.eval_results
"""
from core.context import PipelineContext
from core.schema import EvalResult
from pipeline.prompt_evolution.stages.base import Stage
from services.eval_service import EvalService


class EvaluationStage(Stage):
    """Runs MBPP/HumanEval test harnesses on all generated code outputs."""

    def __init__(self, eval_service: EvalService):
        self.eval_service = eval_service

    def run(self, context: PipelineContext) -> PipelineContext:
        print(f"[EvaluationStage] Evaluating {len(context.generated_codes)} prompt candidates...")
        context.eval_results = {}

        # Build a task lookup: task_id → task dict (for ordered matching)
        task_by_id = {t["task_id"]: t for t in context.tasks}

        for prompt_id, executions in context.generated_codes.items():
            # Reconstruct ordered (code, task) pairs in the same order as execution
            codes: list[str] = []
            tasks: list[dict] = []
            for ex in executions:
                codes.append(ex.output)
                tasks.append(task_by_id.get(ex.task_id, {}))

            summary = self.eval_service.evaluate(codes, tasks)

            result = EvalResult(
                prompt_id=prompt_id,
                passed=summary["passed"],
                failed=summary["failed"],
                total=summary["total"],
            )
            context.eval_results[prompt_id] = result

            # Log error details for visibility
            score_pct = (summary["passed"] / summary["total"] * 100) if summary["total"] else 0
            print(
                f"  ✓ {prompt_id}: "
                f"{summary['passed']}/{summary['total']} passed ({score_pct:.0f}%) "
                f"| syntax_errors={summary['syntax_errors']}"
            )
            if summary.get("errors"):
                for err in summary["errors"][:3]:  # show first 3 errors max
                    print(f"    ↳ {err[:120]}")

        return context
