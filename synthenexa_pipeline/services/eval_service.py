"""
services/eval_service.py — Execute generated code against MBPP/HumanEval test harnesses.

Key design:
  - Each MBPP task ships its own `check(candidate)` function in the `test` field.
  - We exec() the generated code, then call check(entry_point_fn) to run assertions.
  - Syntax errors are tracked separately from logic/assertion errors.
  - All execution is sandboxed in an isolated globals dict (no real sandbox — see warning).

WARNING: exec() of arbitrary LLM-generated code is inherently unsafe.
         In production, wrap this in a Docker container or subprocess with resource limits.
"""
import traceback
from typing import Any, Dict, List


# ------------------------------------------------------------------ #
#  Standalone helper (used by stages/evaluation.py directly)         #
# ------------------------------------------------------------------ #

def run_code_on_task(code: str, task: dict) -> Dict[str, Any]:
    """
    Run a single generated code string against one MBPP/HumanEval task.

    Returns:
        {
            "passed":        bool,
            "error_category": "ok" | "syntax_error" | "runtime_error" | "assertion_error",
            "error_message":  str,
        }
    """
    entry_point = task.get("entry_point", "")
    test_harness = task.get("test", "")

    # Strip markdown fences (belt-and-suspenders, LLMService also strips)
    clean_code = _strip_fences(code)

    exec_globals: Dict[str, Any] = {}

    # --- Step 1: exec the generated solution ---
    try:
        exec(compile(clean_code, "<generated>", "exec"), exec_globals)
    except SyntaxError as exc:
        return {
            "passed": False,
            "error_category": "syntax_error",
            "error_message": str(exc),
        }
    except Exception as exc:
        return {
            "passed": False,
            "error_category": "runtime_error",
            "error_message": str(exc),
        }

    # --- Step 2: inject the test harness and call check(fn) ---
    try:
        exec(compile(test_harness, "<test_harness>", "exec"), exec_globals)

        # The MBPP harness defines check(candidate); call it with the solution fn.
        check_fn = exec_globals.get("check")
        candidate_fn = exec_globals.get(entry_point) if entry_point else None

        if check_fn and candidate_fn:
            check_fn(candidate_fn)
        elif check_fn:
            # Fallback: call check() with no args (some harnesses are self-contained)
            check_fn()
        # else: no check function — treat as passed (best-effort)

        return {"passed": True, "error_category": "ok", "error_message": ""}

    except AssertionError as exc:
        return {
            "passed": False,
            "error_category": "assertion_error",
            "error_message": str(exc),
        }
    except Exception as exc:
        return {
            "passed": False,
            "error_category": "runtime_error",
            "error_message": traceback.format_exc(limit=3),
        }


# ------------------------------------------------------------------ #
#  EvalService class (used by EvaluationStage)                       #
# ------------------------------------------------------------------ #

class EvalService:
    """
    Evaluates a batch of code outputs for one prompt candidate
    against the full task list.
    """

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
        passed        = 0
        failed        = 0
        syntax_errors = 0
        errors: List[str] = []

        for code, task in zip(generated_codes, tasks):
            result = run_code_on_task(code, task)

            if result["passed"]:
                passed += 1
            else:
                failed += 1
                if result["error_category"] == "syntax_error":
                    syntax_errors += 1
                if result["error_message"]:
                    errors.append(
                        f"[{task.get('task_id', '?')}] "
                        f"{result['error_category']}: {result['error_message']}"
                    )

        return {
            "passed":        passed,
            "failed":        failed,
            "total":         len(tasks),
            "syntax_errors": syntax_errors,
            "errors":        errors,
        }


# ------------------------------------------------------------------ #
#  Internal helpers                                                   #
# ------------------------------------------------------------------ #

def _strip_fences(code: str) -> str:
    """Remove markdown code fences from LLM output."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    return code
