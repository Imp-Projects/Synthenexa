import os
import re
import glob
import json
import argparse
import subprocess
import sys
import tempfile
import gzip
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("rank_prompts")

def _configure_logging(verbose: bool = False, debug: bool = False) -> None:
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(format="%(levelname)s | %(message)s", level=level)


# ---------------------------------------------------------------------------
# Path helpers (kept compatible with the original utils.paths interface)
# ---------------------------------------------------------------------------
try:
    from utils.paths import resolve_pipeline_path, pipeline_root
except ImportError:  # allow standalone usage without the utils package
    def resolve_pipeline_path(p):
        return Path(p)

    def pipeline_root():
        return Path(__file__).parent


# ---------------------------------------------------------------------------
# File-opening helper
# ---------------------------------------------------------------------------
def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


# ---------------------------------------------------------------------------
# Fix 12: Task-ID validation
# ---------------------------------------------------------------------------
def _validate_task_ids(sample_task_ids: set, testset_path: str) -> None:
    testset_ids: set = set()
    try:
        with open(testset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    testset_ids.add(json.loads(line).get("task_id"))
                except json.JSONDecodeError:
                    pass
    except OSError as exc:
        logger.warning("Could not read testset for task-ID validation: %s", exc)
        return

    missing_in_testset = sample_task_ids - testset_ids
    missing_in_sample = testset_ids - sample_task_ids

    if missing_in_testset:
        logger.warning(
            "Task IDs in sample but NOT in testset (%d): %s",
            len(missing_in_testset),
            sorted(missing_in_testset)[:10],
        )
    if missing_in_sample:
        logger.debug(
            "Task IDs in testset but NOT in sample (%d): %s",
            len(missing_in_sample),
            sorted(missing_in_sample)[:10],
        )
    if not missing_in_testset:
        logger.info("Task-ID validation passed: all sample IDs found in testset.")


# ---------------------------------------------------------------------------
# Subset problem-file builder
# ---------------------------------------------------------------------------
def _build_subset_problem_file(sample_file_path: str, testset_path: str) -> str:
    sample_task_ids: set = set()
    with open(sample_file_path, "r", encoding="utf-8") as sf:
        for line in sf:
            line = line.strip()
            if not line:
                continue
            try:
                sample_task_ids.add(json.loads(line).get("task_id"))
            except json.JSONDecodeError:
                continue

    # Fix 12: validate task IDs against the testset
    _validate_task_ids(sample_task_ids, testset_path)

    fd, subset_path = tempfile.mkstemp(prefix="subset_problem_", suffix=".jsonl")
    os.close(fd)
    matched = 0
    with open(testset_path, "r", encoding="utf-8") as tf, open(subset_path, "w", encoding="utf-8") as out:
        for line in tf:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("task_id") in sample_task_ids:
                out.write(json.dumps(row) + "\n")
                matched += 1

    logger.debug("Built subset problem file with %d matching tasks: %s", matched, subset_path)
    if matched == 0:
        logger.warning(
            "Subset problem file is EMPTY — no task IDs matched between sample and testset. "
            "Evaluation will likely produce zero scores."
        )
    return subset_path


# ---------------------------------------------------------------------------
# Fix 1: Flexible result-file discovery via glob
# ---------------------------------------------------------------------------
def _resolve_result_file(sample_file_path: str) -> str | None:
    """
    Search for a result file associated with *sample_file_path* using glob
    so that non-standard suffixes and paths are found automatically.
    """
    base = sample_file_path
    # Strip .jsonl (and optionally .gz) to get the stem
    for suffix in (".jsonl.gz", ".jsonl"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    patterns = [
        sample_file_path + "_results.jsonl",
        sample_file_path + "_results.jsonl.gz",
        base + "_results.jsonl",
        base + "_results.jsonl.gz",
        base + "*_results.jsonl",
        base + "*_results.jsonl.gz",
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            chosen = sorted(matches)[0]
            logger.debug("Resolved result file: %s → %s", sample_file_path, chosen)
            return chosen

    logger.debug("No result file found for: %s", sample_file_path)
    return None


# ---------------------------------------------------------------------------
# Fix 8 & 9: Subprocess with captured output; eval-script path validation
# ---------------------------------------------------------------------------
def _generate_results(sample_file_path: str, testset_path: str) -> str | None:
    if not testset_path or not os.path.isfile(testset_path):
        logger.warning("Testset not found, cannot auto-generate results: %s", testset_path)
        return None

    eval_script = pipeline_root() / "human_eval" / "evaluate_functional_correctness.py"

    # Fix 9: validate eval script exists before calling subprocess
    if not eval_script.is_file():
        logger.error("Eval script not found: %s — cannot auto-generate results.", eval_script)
        return None

    venv_python = pipeline_root().parent / "venv" / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    subset_problem_file = _build_subset_problem_file(sample_file_path, testset_path)
    cmd = [python_exe, "-E", str(eval_script), sample_file_path, f"--problem_file={subset_problem_file}"]

    logger.info(
        "Generating result file for %s using %s ...",
        os.path.basename(sample_file_path),
        python_exe,
    )
    logger.debug("Command: %s", " ".join(cmd))

    try:
        # Fix 8: capture stdout/stderr for debugging
        proc = subprocess.run(
            cmd,
            check=True,
            cwd=str(pipeline_root()),
            capture_output=True,
            text=True,
        )
        if proc.stdout:
            logger.debug("Eval stdout:\n%s", proc.stdout)
        if proc.stderr:
            logger.debug("Eval stderr:\n%s", proc.stderr)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Failed to generate results for %s (exit code %d):\nSTDOUT:\n%s\nSTDERR:\n%s",
            os.path.basename(sample_file_path),
            exc.returncode,
            exc.stdout or "(none)",
            exc.stderr or "(none)",
        )
        return None
    finally:
        try:
            os.remove(subset_problem_file)
        except OSError:
            pass

    return _resolve_result_file(sample_file_path)


# ---------------------------------------------------------------------------
# Fix 4: Flexible prompt-ID extraction
# ---------------------------------------------------------------------------
def _extract_prompt_id(filename: str) -> int | None:
    """
    Extract an integer prompt ID from a filename.
    Supports: prompt_12.jsonl, eval_results_007.jsonl, run42_out.jsonl, …
    """
    numbers = re.findall(r"\d+", filename)
    if not numbers:
        return None
    try:
        return int(numbers[-1])  # use the last numeric group (most commonly the ID)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Fix 3 & 7: Multi-key schema + empty/malformed file detection
# ---------------------------------------------------------------------------
def _is_passed(result_data: dict) -> bool:
    """
    Return True if the result record indicates a passing evaluation.
    Supports schemas with keys: passed, pass, success, status.
    """
    if result_data.get("passed") is True:
        return True
    if result_data.get("pass") is True:
        return True
    if result_data.get("success") is True:
        return True
    if str(result_data.get("status", "")).lower() == "passed":
        return True
    return False


def _read_result_stats(result_file_path: str) -> tuple[int, int, float, list, list]:
    total_count = 0
    passed_count = 0
    passed_task_ids: list = []
    failed_details: list = []
    invalid_lines = 0

    with _open_text(result_file_path) as result_file:
        for raw_line in result_file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                result_data = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                logger.debug(
                    "Skipping invalid JSON in %s: %s",
                    result_file_path,
                    raw_line.strip()[:120],
                )
                continue

            total_count += 1
            if _is_passed(result_data):
                passed_count += 1
                task_id = result_data.get("task_id")
                if task_id is not None:
                    passed_task_ids.append(task_id)
            else:
                task_id = result_data.get("task_id", "unknown")
                error_msg = result_data.get("result", "failed")
                failed_details.append(f"Task {task_id}: {error_msg}")

    # Fix 7: detect empty / entirely-malformed files
    if total_count == 0:
        if invalid_lines > 0:
            logger.warning(
                "Result file contains %d line(s) but ALL are invalid JSON: %s",
                invalid_lines,
                result_file_path,
            )
        else:
            logger.warning("Result file is empty: %s", result_file_path)
    elif invalid_lines > 0:
        logger.warning(
            "%d invalid JSON line(s) skipped in %s",
            invalid_lines,
            result_file_path,
        )

    score = passed_count / total_count if total_count > 0 else 0.0
    return total_count, passed_count, score, passed_task_ids, failed_details


# ---------------------------------------------------------------------------
# Error extraction for feedback loop
# ---------------------------------------------------------------------------
def _extract_errors_for_prompt(
    eval_results_folder: str,
    prompt_id: int,
    max_errors: int = 10,
) -> list[str]:
    """
    Collect failed result strings from the result file belonging to *prompt_id*.
    These are injected into evaluation_feedback.jsonl so 1_prompt_mutate.py
    knows exactly what to fix in the next generation.
    """
    errors: list[str] = []

    # Find the sample file for this prompt_id
    for filename in sorted(os.listdir(eval_results_folder)):
        if not filename.endswith(".jsonl") or filename.endswith("_results.jsonl"):
            continue
        if _extract_prompt_id(filename) != prompt_id:
            continue

        sample_path = os.path.join(eval_results_folder, filename)
        result_path = _resolve_result_file(sample_path)
        if not result_path:
            break

        with _open_text(result_path) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not _is_passed(record):
                    result_str = record.get("result", "")
                    task_id = record.get("task_id", "unknown")
                    if result_str and result_str.lower() not in ("failed", ""):
                        errors.append(f"[{task_id}] {result_str}")
                    if len(errors) >= max_errors:
                        break
        break

    return errors


# ---------------------------------------------------------------------------
# Fix 11: Evaluation status tracking
# ---------------------------------------------------------------------------
EVAL_STATUS_SUCCESS = "success"
EVAL_STATUS_MISSING = "missing_results"
EVAL_STATUS_INVALID = "invalid_schema"
EVAL_STATUS_FAILED = "failed_eval"


# ---------------------------------------------------------------------------
# Core ranking logic
# ---------------------------------------------------------------------------
def rank_prompts(
    eval_results_folder: str,
    testset_path: str | None = None,
    strict: bool = False,
) -> list[dict]:
    """
    Rank prompt samples by pass@1 score and normalized inverse-frequency
    weighted score.

    Parameters
    ----------
    eval_results_folder : directory containing JSONL sample files
    testset_path        : optional path to the problem testset for auto-eval
    strict              : raise an error instead of a warning on missing results
    """
    prompt_stats: dict[int, dict] = {}
    task_correct_counts: dict = {}

    for filename in sorted(os.listdir(eval_results_folder)):
        if not filename.endswith(".jsonl") or filename.endswith("_results.jsonl"):
            continue

        prompt_id = _extract_prompt_id(filename)
        if prompt_id is None:
            logger.error("Could not extract prompt_id from filename: %s — skipping.", filename)
            continue

        jsonl_file_path = os.path.join(eval_results_folder, filename)
        result_file_path = _resolve_result_file(jsonl_file_path)

        # Auto-generate if testset is available
        if not result_file_path:
            result_file_path = _generate_results(jsonl_file_path, testset_path)

        # Fix 2: explicit warning / strict failure on missing results
        if not result_file_path:
            msg = (
                f"Missing result file for {filename} "
                f"(tried glob patterns around {jsonl_file_path})"
            )
            if strict:
                raise FileNotFoundError(msg)
            logger.warning("%s — assigning score=0.0 with status=%s", msg, EVAL_STATUS_MISSING)
            prompt_stats[prompt_id] = {
                "prompt_id": prompt_id,
                "score": 0.0,
                "weighted_score": 0.0,
                "passed_count": 0,
                "total_count": 0,
                "eval_status": EVAL_STATUS_MISSING,
            }
            continue

        total_count, passed_count, score, passed_task_ids, failed_details = _read_result_stats(result_file_path)

        # Fix 11: determine evaluation status
        if total_count == 0:
            eval_status = EVAL_STATUS_INVALID
        else:
            eval_status = EVAL_STATUS_SUCCESS

        for task_id in passed_task_ids:
            task_correct_counts[task_id] = task_correct_counts.get(task_id, 0) + 1

        prompt_stats[prompt_id] = {
            "prompt_id": prompt_id,
            "score": score,
            "weighted_score": 0.0,
            "passed_count": passed_count,
            "total_count": total_count,
            "eval_status": eval_status,
            "_passed_task_ids": passed_task_ids,
            "errors": failed_details,
        }
        logger.info(
            "%-45s score=%.2f%%  passed=%d/%d  status=%s",
            filename,
            score * 100,
            passed_count,
            total_count,
            eval_status,
        )

    ranked_scores = sorted(
        prompt_stats.values(),
        key=lambda row: (-row["score"], row["prompt_id"]),
    )

    for rank_index, row in enumerate(ranked_scores, start=1):
        row["rank"] = rank_index

    return ranked_scores


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------
def write_ranked_prompts(
    prompts_path: str,
    ranked_scores: list[dict],
    output_path: str,
    eval_results_folder: str | None = None,
) -> tuple[int, list[dict]]:
    prompts_by_id: dict = {}
    all_prompt_ids: list = []

    with open(prompts_path, "r", encoding="utf-8") as prompts_file:
        for raw_line in prompts_file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                prompt_data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON in prompts file: %s", raw_line.strip()[:120])
                continue
            prompt_id = prompt_data.get("prompt_id")
            if prompt_id is None:
                continue
            prompts_by_id[prompt_id] = prompt_data
            all_prompt_ids.append(prompt_id)

    scores_by_id = {row["prompt_id"]: row for row in ranked_scores}

    for prompt_id in all_prompt_ids:
        if prompt_id not in scores_by_id:
            scores_by_id[prompt_id] = {
                "prompt_id": prompt_id,
                "score": 0.0,
                "weighted_score": 0.0,
                "passed_count": 0,
                "total_count": 0,
                "eval_status": EVAL_STATUS_MISSING,
                "rank": None,
            }

    final_ranked = sorted(
        scores_by_id.values(),
        key=lambda row: (-row["score"], row["prompt_id"]),
    )
    for rank_index, row in enumerate(final_ranked, start=1):
        row["rank"] = rank_index

    saved = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for row in final_ranked:
            prompt_id = row["prompt_id"]
            prompt_data = prompts_by_id.get(prompt_id)
            if not prompt_data:
                continue
            prompt_data = dict(prompt_data)
            prompt_data["score"] = row["score"]
            prompt_data["passed_count"] = row["passed_count"]
            prompt_data["total_count"] = row["total_count"]
            prompt_data["eval_status"] = row.get("eval_status", EVAL_STATUS_MISSING)
            prompt_data["rank"] = row["rank"]
            # Only the top-ranked prompt carries error traces — 1_prompt_mutate.py
            # reads only the first line of this file, so errors must be on rank 1.
            if row["rank"] == 1:
                prompt_data["errors"] = row.get("errors", [])
                if prompt_data["errors"]:
                    logger.info(
                        "Injected %d error traces into feedback for top prompt_id=%d",
                        len(prompt_data["errors"]), prompt_id,
                    )
            else:
                prompt_data["errors"] = []
            out.write(json.dumps(prompt_data) + "\n")
            saved += 1

    return saved, final_ranked


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(
    eval_results_folder: str,
    prompts_path: str,
    output_path: str,
    testset_path: str | None = None,
    strict: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> None:
    _configure_logging(verbose=verbose, debug=debug)

    eval_results_folder = str(resolve_pipeline_path(eval_results_folder))
    prompts_path = str(resolve_pipeline_path(prompts_path))
    output_path = str(resolve_pipeline_path(output_path))

    if testset_path:
        testset_path = str(resolve_pipeline_path(testset_path))
    else:
        default_testset = str(Path(eval_results_folder).parent / "train_set.jsonl")
        testset_path = default_testset if os.path.isfile(default_testset) else None

    if not os.path.isdir(eval_results_folder):
        raise FileNotFoundError(f"Evaluation folder not found: {eval_results_folder}")
    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if testset_path:
        logger.info("Using test set for auto-evaluation: %s", testset_path)
    else:
        logger.info("No test set found; using only pre-existing *_results.jsonl files.")

    ranked_scores = rank_prompts(
        eval_results_folder,
        testset_path=testset_path,
        strict=strict,
    )

    if not ranked_scores:
        logger.error(
            "No prompt scores found. Ensure evaluation samples exist and result files are available."
        )
        return

    saved, final_ranked = write_ranked_prompts(
        prompts_path, ranked_scores, output_path, eval_results_folder=eval_results_folder
    )

    # Summary
    success_count = sum(1 for r in final_ranked if r.get("eval_status") == EVAL_STATUS_SUCCESS)
    missing_count = sum(1 for r in final_ranked if r.get("eval_status") == EVAL_STATUS_MISSING)
    invalid_count = sum(1 for r in final_ranked if r.get("eval_status") == EVAL_STATUS_INVALID)

    top = final_ranked[0] if final_ranked else None
    if top:
        print(
            f"\nRanked prompts saved → {output_path}\n"
            f"  Top: prompt_id={top['prompt_id']}  score={top['score']*100:.2f}%  "
            f"status={top.get('eval_status')}\n"
            f"  Entries written : {saved}\n"
            f"  Eval summary    : {success_count} success | "
            f"{missing_count} missing_results | {invalid_count} invalid_schema"
        )
    else:
        print(f"Ranked prompts saved to {output_path}. Entries written: {saved}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate pass@1 and rank prompts by weighted score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--evaluate_path",
        required=True,
        help="Folder containing evaluated JSONL files and their _results.jsonl outputs.",
    )
    parser.add_argument(
        "--origin_prompt",
        default="Dataset/mutated_prmts.jsonl",
        help="Path to the prompt JSONL file used to select the best prompts.",
    )
    parser.add_argument(
        "--best_prompt",
        required=True,
        help="Path to save the best prompt JSONL file.",
    )
    parser.add_argument(
        "--testset_path",
        default=None,
        help="Optional path to train/test set JSONL for auto-generating missing result files.",
    )
    # Fix 2 & 10: new flags
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort (raise FileNotFoundError) when a result file is missing instead of warning.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print INFO-level log messages.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print DEBUG-level log messages (very verbose).",
    )

    args = parser.parse_args()
    main(
        args.evaluate_path,
        args.origin_prompt,
        args.best_prompt,
        testset_path=args.testset_path,
        strict=args.strict,
        verbose=args.verbose,
        debug=args.debug,
    )