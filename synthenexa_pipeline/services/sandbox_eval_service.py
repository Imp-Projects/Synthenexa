"""
services/sandbox_eval_service.py — Remote sandbox evaluation using E2B.

Mirrors the EvalService interface exactly so it can be swapped in transparently
via the --eval-backend flag without changing any pipeline stage code.

How it works:
  - Each code/task pair is evaluated inside an isolated E2B microVM sandbox.
  - The generated solution and the MBPP test harness are combined into a single
    script and executed via sandbox.run_code().
  - A configurable timeout (default: 10s) is enforced by E2B at the VM level,
    preventing infinite loops from hanging the pipeline.
  - Evaluations run concurrently using ThreadPoolExecutor for speed.

Requirements:
  - E2B_API_KEY must be set in the environment (or .env file).
  - pip install e2b-code-interpreter
"""
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from dotenv import load_dotenv

# Resolve .env relative to the repo root (two levels up from services/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
load_dotenv(os.path.join(_ROOT, ".env"), override=False)

try:
    from e2b_code_interpreter import Sandbox
    _E2B_AVAILABLE = True
except ImportError:
    _E2B_AVAILABLE = False


# ------------------------------------------------------------------ #
#  Per-task sandbox runner                                            #
# ------------------------------------------------------------------ #

def run_code_on_task_sandbox(code: str, task: dict, timeout: int = 10) -> Dict[str, Any]:
    """
    Execute a single generated code string against one MBPP task inside an E2B sandbox.

    Args:
        code:    Generated Python solution (may include markdown fences).
        task:    Task dict with 'entry_point' and 'test' fields.
        timeout: Maximum seconds before E2B forcefully terminates execution.

    Returns:
        {
            "passed":         bool,
            "error_category": "ok" | "syntax_error" | "runtime_error" | "assertion_error",
            "error_message":  str,
            "elapsed_ms":     float,
        }
    """
    if not _E2B_AVAILABLE:
        return {
            "passed": False,
            "error_category": "runtime_error",
            "error_message": "e2b-code-interpreter is not installed. Run: pip install e2b-code-interpreter",
            "elapsed_ms": 0.0,
        }

    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        return {
            "passed": False,
            "error_category": "runtime_error",
            "error_message": "E2B_API_KEY is not set. Add it to your .env file.",
            "elapsed_ms": 0.0,
        }

    entry_point = task.get("entry_point", "")
    test_harness = task.get("test", "")
    clean_code = _strip_fences(code)

    # Build a self-contained script: solution + harness + runner
    if entry_point:
        runner = f"\ncheck({entry_point})\nprint('__PASS__')\n"
    else:
        runner = "\ncheck()\nprint('__PASS__')\n"

    script = f"{clean_code}\n\n{test_harness}\n{runner}"

    _t_start = time.perf_counter()
    try:
        with Sandbox(api_key=api_key, timeout=timeout) as sbx:
            result = sbx.run_code(script)
            elapsed_ms = (time.perf_counter() - _t_start) * 1000

            stdout = result.logs.stdout or []
            stderr = result.logs.stderr or []
            stdout_text = "\n".join(stdout).strip()
            stderr_text = "\n".join(stderr).strip()

            # Check for our sentinel pass marker
            if "__PASS__" in stdout_text:
                return {
                    "passed": True,
                    "error_category": "ok",
                    "error_message": "",
                    "elapsed_ms": elapsed_ms,
                }

            # Classify failure from stderr
            error_category = "runtime_error"
            if "SyntaxError" in stderr_text:
                error_category = "syntax_error"
            elif "AssertionError" in stderr_text:
                error_category = "assertion_error"

            return {
                "passed": False,
                "error_category": error_category,
                "error_message": stderr_text[:500],
                "elapsed_ms": elapsed_ms,
            }

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - _t_start) * 1000
        return {
            "passed": False,
            "error_category": "runtime_error",
            "error_message": traceback.format_exc(limit=3),
            "elapsed_ms": elapsed_ms,
        }


# ------------------------------------------------------------------ #
#  SandboxEvalService — drop-in replacement for EvalService           #
# ------------------------------------------------------------------ #

class SandboxEvalService:
    """
    Evaluates a batch of generated code outputs using remote E2B sandboxes.

    Implements the same evaluate(codes, tasks) interface as EvalService so
    it can be swapped in without any pipeline stage changes.
    """

    def __init__(self, max_workers: int = 8, timeout: int = 10):
        """
        Args:
            max_workers: Maximum number of concurrent E2B sandbox sessions.
            timeout:     Per-sandbox execution timeout in seconds.
        """
        self.max_workers = max_workers
        self.timeout = timeout

    def evaluate(
        self,
        generated_codes: List[str],
        tasks: List[dict],
    ) -> Dict[str, Any]:
        """
        Run all generated codes against their corresponding tasks in parallel sandboxes.

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
                "total_time_ms": float,
                "min_time_ms":   float,
                "max_time_ms":   float,
                "avg_time_ms":   float,
            }
        """
        pairs = list(zip(generated_codes, tasks))
        results_by_index: Dict[int, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(run_code_on_task_sandbox, code, task, self.timeout): idx
                for idx, (code, task) in enumerate(pairs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_by_index[idx] = future.result()
                except Exception as exc:
                    results_by_index[idx] = {
                        "passed": False,
                        "error_category": "runtime_error",
                        "error_message": str(exc),
                        "elapsed_ms": 0.0,
                    }

        # Aggregate in original order
        passed        = 0
        failed        = 0
        syntax_errors = 0
        errors: List[str] = []
        timings: List[float] = []

        for idx, (_, task) in enumerate(pairs):
            result  = results_by_index[idx]
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
        min_ms   = min(timings, default=0.0)
        max_ms   = max(timings, default=0.0)
        avg_ms   = total_ms / len(timings) if timings else 0.0

        return {
            "passed":        passed,
            "failed":        failed,
            "total":         len(tasks),
            "syntax_errors": syntax_errors,
            "errors":        errors,
            "total_time_ms": total_ms,
            "min_time_ms":   min_ms,
            "max_time_ms":   max_ms,
            "avg_time_ms":   avg_ms,
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
