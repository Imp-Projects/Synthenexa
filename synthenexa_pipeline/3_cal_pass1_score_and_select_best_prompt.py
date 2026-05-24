import os
import re
import json
import argparse
import subprocess
import sys
import tempfile
import gzip
from pathlib import Path

from utils.paths import resolve_pipeline_path, pipeline_root


def _open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _build_subset_problem_file(sample_file_path, testset_path):
    sample_task_ids = set()
    with open(sample_file_path, "r", encoding="utf-8") as sf:
        for line in sf:
            if not line.strip():
                continue
            try:
                sample_task_ids.add(json.loads(line).get("task_id"))
            except json.JSONDecodeError:
                continue

    fd, subset_path = tempfile.mkstemp(prefix="subset_problem_", suffix=".jsonl")
    os.close(fd)
    with open(testset_path, "r", encoding="utf-8") as tf, open(subset_path, "w", encoding="utf-8") as out:
        for line in tf:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("task_id") in sample_task_ids:
                out.write(json.dumps(row) + "\n")
    return subset_path


def _result_candidates(sample_file_path):
    base, ext = os.path.splitext(sample_file_path)
    return [
        sample_file_path + "_results.jsonl",
        sample_file_path + "_results.jsonl.gz",
        base + "_results.jsonl",
        base + "_results.jsonl.gz",
    ]


def _resolve_result_file(sample_file_path):
    for candidate in _result_candidates(sample_file_path):
        if os.path.exists(candidate):
            return candidate
    return None


def _generate_results(sample_file_path, testset_path):
    if not testset_path or not os.path.isfile(testset_path):
        return None

    eval_script = pipeline_root() / "human_eval" / "evaluate_functional_correctness.py"
    venv_python = pipeline_root().parent / "venv" / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    subset_problem_file = _build_subset_problem_file(sample_file_path, testset_path)
    cmd = [python_exe, "-E", str(eval_script), sample_file_path, f"--problem_file={subset_problem_file}"]

    try:
        print(f"Generating result file for {os.path.basename(sample_file_path)} using {python_exe}...")
        subprocess.run(cmd, check=True, cwd=str(pipeline_root()))
    except subprocess.CalledProcessError as exc:
        print(f"Failed to generate results for {os.path.basename(sample_file_path)}: {exc}")
        return None
    finally:
        try:
            os.remove(subset_problem_file)
        except OSError:
            pass

    return _resolve_result_file(sample_file_path)


def _extract_prompt_id(filename):
    match = re.search(r'_(\d+)\.jsonl$', filename)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _read_result_stats(result_file_path):
    total_count = 0
    passed_count = 0
    passed_task_ids = []

    with _open_text(result_file_path) as result_file:
        for raw_line in result_file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                result_data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid JSON line in {result_file_path}: {raw_line.strip()}")
                continue

            total_count += 1
            if result_data.get("passed") is True:
                passed_count += 1
                task_id = result_data.get("task_id")
                if task_id is not None:
                    passed_task_ids.append(task_id)

    score = passed_count / total_count if total_count > 0 else 0.0
    return total_count, passed_count, score, passed_task_ids


def rank_prompts(eval_results_folder, testset_path=None):
    prompt_stats = {}
    task_correct_counts = {}

    for filename in sorted(os.listdir(eval_results_folder)):
        if not filename.endswith(".jsonl") or filename.endswith("_results.jsonl"):
            continue

        prompt_id = _extract_prompt_id(filename)
        if prompt_id is None:
            print(f"Error extracting prompt_id from filename: {filename}")
            continue

        jsonl_file_path = os.path.join(eval_results_folder, filename)
        result_file_path = _resolve_result_file(jsonl_file_path)
        if not result_file_path:
            result_file_path = _generate_results(jsonl_file_path, testset_path)

        if not result_file_path:
            print(f"Skipping {filename}: missing result file (looked for *_results.jsonl / *_results.jsonl.gz)")
            prompt_stats[prompt_id] = {
                "prompt_id": prompt_id,
                "score": 0.0,
                "weighted_score": 0.0,
                "passed_count": 0,
                "total_count": 0,
            }
            continue

        total_count, passed_count, score, passed_task_ids = _read_result_stats(result_file_path)
        for task_id in passed_task_ids:
            task_correct_counts[task_id] = task_correct_counts.get(task_id, 0) + 1

        prompt_stats[prompt_id] = {
            "prompt_id": prompt_id,
            "score": score,
            "weighted_score": 0.0,  # filled after global task weighting is computed
            "passed_count": passed_count,
            "total_count": total_count,
            "_passed_task_ids": passed_task_ids,
        }
        print(f"File: {filename}, score: {score} (from {os.path.basename(result_file_path)})")

    total_passes = sum(task_correct_counts.values())
    task_weights = {}
    if total_passes > 0:
        task_weights = {task_id: total_passes / count for task_id, count in task_correct_counts.items()}

    for stats in prompt_stats.values():
        weighted_score = 0.0
        for task_id in stats.get("_passed_task_ids", []):
            weighted_score += task_weights.get(task_id, 0.0)
        stats["weighted_score"] = weighted_score
        stats.pop("_passed_task_ids", None)

    ranked_scores = sorted(
        prompt_stats.values(),
        key=lambda row: (-row["weighted_score"], -row["score"], row["prompt_id"]),
    )

    for rank_index, row in enumerate(ranked_scores, start=1):
        row["rank"] = rank_index

    return ranked_scores


def write_ranked_prompts(prompts_path, ranked_scores, output_path):
    prompts_by_id = {}
    all_prompt_ids = []

    with open(prompts_path, "r", encoding="utf-8") as prompts_file:
        for raw_line in prompts_file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                prompt_data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid JSON line in {prompts_path}: {raw_line.strip()}")
                continue
            prompt_id = prompt_data.get("prompt_id")
            if prompt_id is None:
                continue
            prompts_by_id[prompt_id] = prompt_data
            all_prompt_ids.append(prompt_id)

    scores_by_id = {row["prompt_id"]: row for row in ranked_scores}

    # Ensure every prompt from origin file has a score entry, even if missing evaluation output.
    for prompt_id in all_prompt_ids:
        if prompt_id not in scores_by_id:
            scores_by_id[prompt_id] = {
                "prompt_id": prompt_id,
                "score": 0.0,
                "weighted_score": 0.0,
                "passed_count": 0,
                "total_count": 0,
                "rank": None,
            }

    final_ranked = sorted(
        scores_by_id.values(),
        key=lambda row: (-row["weighted_score"], -row["score"], row["prompt_id"]),
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
            prompt_data["weighted_score"] = row["weighted_score"]
            prompt_data["passed_count"] = row["passed_count"]
            prompt_data["total_count"] = row["total_count"]
            prompt_data["rank"] = row["rank"]
            out.write(json.dumps(prompt_data) + "\n")
            saved += 1

    return saved, final_ranked


def main(eval_results_folder, prompts_path, output_path, testset_path=None):
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
        print(f"Using test set for auto-evaluation: {testset_path}")
    else:
        print("No test set auto-detected; this script will only use existing *_results.jsonl files.")

    ranked_scores = rank_prompts(eval_results_folder, testset_path=testset_path)

    if not ranked_scores:
        print("No prompt scores found. Ensure evaluation samples exist and that result files can be generated.")
        return

    saved, final_ranked = write_ranked_prompts(prompts_path, ranked_scores, output_path)
    top = final_ranked[0] if final_ranked else None
    if top:
        print(
            f"Ranked prompts saved to {output_path}. Top prompt_id={top['prompt_id']}, "
            f"score={top['score']}, weighted_score={top['weighted_score']}, entries written={saved}"
        )
    else:
        print(f"Ranked prompts saved to {output_path}. Entries written={saved}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate pass@1 and extract best prompts.')
    parser.add_argument('--evaluate_path', type=str, required=True, help='Folder containing evaluated JSONL files and their _results.jsonl outputs.')
    parser.add_argument('--origin_prompt', type=str, default='Dataset/mutated_prmts.jsonl', help='Path to the prompt JSONL file used to select the best prompts.')
    parser.add_argument('--best_prompt', type=str, required=True, help='Path to save the best prompt JSONL file.')
    parser.add_argument('--testset_path', type=str, default=None, help='Optional path to train/test set JSONL used to auto-generate missing result files.')

    args = parser.parse_args()
    main(args.evaluate_path, args.origin_prompt, args.best_prompt, args.testset_path)
