"""Sandboxed task execution helpers for generated code evaluation.

The sandbox is process-based:
  - code runs in a fresh child process
  - a restricted import function limits module access
  - a timeout prevents runaway execution

This is intentionally lightweight so it can run in the current project
without changing the pipeline API, while still avoiding direct exec() in the
main orchestration process.
"""

from __future__ import annotations

import builtins
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable


SAFE_MODULES = {
    "math",
    "re",
    "itertools",
    "functools",
    "collections",
    "heapq",
    "bisect",
    "operator",
    "statistics",
    "string",
}


@dataclass
class SandboxResult:
    passed: bool
    error_category: str
    error_message: str
    elapsed_ms: float


def _strip_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    return code


def _restricted_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root not in SAFE_MODULES:
        raise ImportError(f"Import of '{name}' is blocked by the sandbox")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _sandbox_builtins() -> dict[str, Any]:
    allowed = [
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "pow",
        "print",
        "range",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    ]
    safe = {name: getattr(builtins, name) for name in allowed}
    safe["__import__"] = _restricted_import
    return safe


def _worker(code: str, test_harness: str, entry_point: str, queue: mp.Queue) -> None:
    t_start = time.perf_counter()
    exec_globals: Dict[str, Any] = {"__builtins__": _sandbox_builtins()}

    try:
        exec(compile(code, "<generated>", "exec"), exec_globals)
    except SyntaxError as exc:
        queue.put(
            SandboxResult(
                passed=False,
                error_category="syntax_error",
                error_message=str(exc),
                elapsed_ms=(time.perf_counter() - t_start) * 1000,
            ).__dict__
        )
        return
    except Exception as exc:
        queue.put(
            SandboxResult(
                passed=False,
                error_category="runtime_error",
                error_message=str(exc),
                elapsed_ms=(time.perf_counter() - t_start) * 1000,
            ).__dict__
        )
        return

    try:
        exec(compile(test_harness, "<test_harness>", "exec"), exec_globals)
        check_fn = exec_globals.get("check")
        candidate_fn = exec_globals.get(entry_point) if entry_point else None

        if check_fn and candidate_fn:
            check_fn(candidate_fn)
        elif check_fn:
            check_fn()

        queue.put(
            SandboxResult(
                passed=True,
                error_category="ok",
                error_message="",
                elapsed_ms=(time.perf_counter() - t_start) * 1000,
            ).__dict__
        )
    except AssertionError as exc:
        queue.put(
            SandboxResult(
                passed=False,
                error_category="assertion_error",
                error_message=str(exc),
                elapsed_ms=(time.perf_counter() - t_start) * 1000,
            ).__dict__
        )
    except Exception:
        queue.put(
            SandboxResult(
                passed=False,
                error_category="runtime_error",
                error_message=traceback.format_exc(limit=3),
                elapsed_ms=(time.perf_counter() - t_start) * 1000,
            ).__dict__
        )


class SandboxEvaluator:
    """Runs generated code in a child process with a restricted runtime."""

    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds

    def run_code_on_task(self, code: str, task: dict) -> Dict[str, Any]:
        entry_point = task.get("entry_point", "")
        test_harness = task.get("test", "")
        clean_code = _strip_fences(code)

        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_worker,
            args=(clean_code, test_harness, entry_point, queue),
            daemon=True,
        )
        proc.start()
        proc.join(self.timeout_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            return {
                "passed": False,
                "error_category": "timeout",
                "error_message": f"Execution exceeded {self.timeout_seconds:.1f}s",
                "elapsed_ms": self.timeout_seconds * 1000,
            }

        if queue.empty():
            return {
                "passed": False,
                "error_category": "runtime_error",
                "error_message": "Sandbox worker exited without returning a result",
                "elapsed_ms": 0.0,
            }

        return queue.get()
