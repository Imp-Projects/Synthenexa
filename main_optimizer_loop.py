import os
import subprocess
import json
import argparse
import sys
from pathlib import Path


def run_step(script_name: str, args: list[str]) -> None:
    cmd = [sys.executable, f"synthenexa_pipeline/{script_name}"] + args
    print(f"\n--- Running {script_name} ---")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error running {script_name} (exit code {result.returncode})")
        sys.exit(1)


def _save_final_best(feedback_path: str, output_path: str) -> None:
    """Copy the top-ranked prompt from the last feedback file to a clean output."""
    if not os.path.exists(feedback_path):
        print("No feedback file found — skipping final best-prompt export.")
        return
    with open(feedback_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(first_line + "\n")
    print(f"\nFinal best prompt saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatic prompt optimization loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generations", type=int, default=3,
                        help="Number of optimization generations to run.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model identifier (e.g. ollama:qwen3-coder:30b).")
    parser.add_argument("--num_mutations", type=int, default=5,
                        help="Number of mutated prompts to generate per generation.")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of concurrent model requests during evaluation.")
    parser.add_argument("--verbose", action="store_true",
                        help="Pass --verbose to the scoring step for INFO-level logs.")
    parser.add_argument("--debug", action="store_true",
                        help="Pass --debug to the scoring step for DEBUG-level logs.")
    args = parser.parse_args()

    base_prompt_path  = "synthenexa_pipeline/Dataset/base_prompt.jsonl"
    feedback_path     = "synthenexa_pipeline/Dataset/evaluation_feedback.jsonl"
    benchmark_path    = "synthenexa_pipeline/Dataset/benchmark_mbpp_micro.jsonl"
    final_best_path   = "synthenexa_pipeline/Dataset/best_prompt_final.jsonl"

    # Ensure base prompt exists; create a minimal default if not.
    if not os.path.exists(base_prompt_path):
        os.makedirs(os.path.dirname(base_prompt_path), exist_ok=True)
        with open(base_prompt_path, "w", encoding="utf-8") as f:
            json.dump({"prompt_id": 0, "prompt": "Write a Python function to solve the given task."}, f)
            f.write("\n")
        print(f"Created default base prompt at {base_prompt_path}")

    # Clear stale feedback from a previous full run so generation 1 starts blind.
    if os.path.exists(feedback_path):
        os.remove(feedback_path)
        print(f"Cleared previous feedback: {feedback_path}")

    for gen in range(1, args.generations + 1):
        print(f"\n======================================")
        print(f"       GENERATION {gen} / {args.generations}       ")
        print(f"======================================")

        mutated_prompts_path = f"synthenexa_pipeline/Dataset/gen_{gen}_mutated_prompts.jsonl"
        eval_output_dir      = f"synthenexa_pipeline/Dataset/gen_{gen}_eval_output"

        # ------------------------------------------------------------------
        # Step 1 — Mutate prompts
        # Feedback from the previous generation (if it exists) is injected
        # here so the mutator knows which errors to fix.
        # ------------------------------------------------------------------
        mutate_args = [
            "--base_prompt_path", base_prompt_path,
            "--output_path",      mutated_prompts_path,
            "--num_mutations",    str(args.num_mutations),
        ]
        if os.path.exists(feedback_path):
            mutate_args.extend(["--feedback_path", feedback_path])
        if args.model:
            mutate_args.extend(["--model", args.model])

        run_step("1_prompt_mutate.py", mutate_args)

        # ------------------------------------------------------------------
        # Step 2 — Generate code completions for each mutated prompt
        # ------------------------------------------------------------------
        generate_args = [
            "--trainset_path",       benchmark_path,
            "--mutated_prompt_path", mutated_prompts_path,
            "--output_path",         eval_output_dir,
            "--max_workers",         str(args.max_workers),
        ]
        if args.model:
            generate_args.extend(["--model", args.model])

        run_step("2_prompt_evaluate.py", generate_args)

        # ------------------------------------------------------------------
        # Step 3 — Score completions, rank prompts, write feedback
        # The output (--best_prompt) becomes the next generation's
        # --feedback_path input in step 1, closing the refinement loop.
        # ------------------------------------------------------------------
        score_args = [
            "--evaluate_path", eval_output_dir,
            "--origin_prompt", mutated_prompts_path,
            "--best_prompt",   feedback_path,
            "--testset_path",  benchmark_path,
        ]
        if args.verbose:
            score_args.append("--verbose")
        if args.debug:
            score_args.append("--debug")

        run_step("3_cal_pass1_score_and_select_best_prompt.py", score_args)

        print(f"\nGeneration {gen} complete. Feedback saved → {feedback_path}")

    # ----------------------------------------------------------------------
    # Post-loop: export the final best prompt to a clean file
    # ----------------------------------------------------------------------
    _save_final_best(feedback_path, final_best_path)
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()