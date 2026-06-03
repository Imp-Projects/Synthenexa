import os
import subprocess
import json
import argparse
import sys
from pathlib import Path

def run_step(script_name, args):
    cmd = [sys.executable, f"synthenexa_pipeline/{script_name}"] + args
    print(f"\n--- Running {script_name} ---")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error running {script_name}")
        exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num_mutations", type=int, default=5, help="Number of mutated prompts to generate per generation")
    args = parser.parse_args()

    base_prompt_path = "synthenexa_pipeline/Dataset/base_prompt.jsonl"
    feedback_path = "synthenexa_pipeline/Dataset/evaluation_feedback.jsonl"
    benchmark_path = "synthenexa_pipeline/Dataset/benchmark_mbpp_micro.jsonl"
    
    # Ensure base prompt exists, if not create a default one
    if not os.path.exists(base_prompt_path):
        os.makedirs(os.path.dirname(base_prompt_path), exist_ok=True)
        with open(base_prompt_path, "w", encoding="utf-8") as f:
            json.dump({"prompt_id": 0, "prompt": "Write a Python function to solve the given task."}, f)
            f.write("\n")
            
    # Clear previous feedback if any
    if os.path.exists(feedback_path):
        os.remove(feedback_path)
        
    for gen in range(1, args.generations + 1):
        print(f"\n======================================")
        print(f"       GENERATION {gen} / {args.generations}       ")
        print(f"======================================")
        
        mutated_prompts_path = f"synthenexa_pipeline/Dataset/gen_{gen}_mutated_prompts.jsonl"
        eval_output_dir = f"synthenexa_pipeline/Dataset/gen_{gen}_eval_output"
        
        # 1. Mutate
        mutate_args = [
            "--base_prompt_path", base_prompt_path,
            "--output_path", mutated_prompts_path,
            "--num_mutations", str(args.num_mutations)
        ]
        if os.path.exists(feedback_path):
            mutate_args.extend(["--feedback_path", feedback_path])
        if args.model:
            mutate_args.extend(["--model", args.model])
            
        run_step("1_prompt_mutate.py", mutate_args)
        
        # 2. Evaluate (Generate Code)
        generate_args = [
            "--trainset_path", benchmark_path,
            "--mutated_prompt_path", mutated_prompts_path,
            "--output_path", eval_output_dir,
            "--max_workers", "5"
        ]
        if args.model:
            generate_args.extend(["--model", args.model])
            
        run_step("2_prompt_evaluate.py", generate_args)
        
        # 3. Score (Run tests and extract errors)
        score_args = [
            "--evaluate_path", eval_output_dir,
            "--origin_prompt", mutated_prompts_path,
            "--best_prompt", feedback_path,
            "--testset_path", benchmark_path
        ]
        run_step("3_cal_pass1_score_and_select_best_prompt.py", score_args)
        
        print(f"\nGeneration {gen} complete. Feedback saved to {feedback_path}")

    print("\nOptimization Complete!")

if __name__ == "__main__":
    main()
