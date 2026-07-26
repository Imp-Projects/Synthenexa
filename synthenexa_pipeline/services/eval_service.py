"""Execute generated code against MBPP/HumanEval test harnesses.

The public API stays the same, but execution now happens inside a sandboxed
child process instead of the main pipeline process.
"""

from __future__ import annotations

from typing import Any, Dict, List

from services.eval_sandbox import SandboxEvaluator


class EvalService:
    """
    Evaluates a batch of code outputs for one prompt candidate
    against the full task list.
    """

    def __init__(self, timeout_seconds: float = 10.0):
        self._sandbox = SandboxEvaluator(timeout_seconds=timeout_seconds)

    def evaluate(
        self,
        generated_codes: List[str],
        tasks: List[dict],
    ) -> Dict[str, Any]:
        """
        Run all generated codes against their corresponding tasks.

        Args:
            generated_codes: List of code strings (one per task).
            tasks:           List of task dicts with `entry_point` and `test` fields.

        Returns:
            {
                "passed":        int,
                "failed":        int,
                "total":         int,
                "syntax_errors": int,
                "errors":        list[str],
            }
        """
        passed = 0
        failed = 0
        syntax_errors = 0
        errors: List[str] = []
        timings: List[float] = []

        for code, task in zip(generated_codes, tasks):
            result = self._sandbox.run_code_on_task(code, task)
            elapsed = result.get("elapsed_ms", 0.0)
            timings.append(elapsed)

            if result["passed"]:
                passed += 1
            else:
                failed += 1
                if result["error_category"] == "syntax_error":
                    syntax_errors += 1
                if result["error_message"]:
                    errors.append(
                        f"[{task.get('task_id', '?')}] "
                        f"{result['error_category']}: {result['error_message']} "
                        f"({elapsed:.1f}ms)"
                    )

        total_ms = sum(timings)
        min_ms = min(timings, default=0.0)
        max_ms = max(timings, default=0.0)
        avg_ms = total_ms / len(timings) if timings else 0.0

        return {
            "passed": passed,
            "failed": failed,
            "total": len(tasks),
            "syntax_errors": syntax_errors,
            "errors": errors,
            "total_time_ms": total_ms,
            "min_time_ms": min_ms,
            "max_time_ms": max_ms,
            "avg_time_ms": avg_ms,
        }
