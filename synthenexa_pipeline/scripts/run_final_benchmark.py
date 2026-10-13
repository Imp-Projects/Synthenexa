#!/usr/bin/env python3
"""
scripts/run_final_benchmark.py

A harness to run the Phase C final evaluation of base prompt vs refined prompt.
It will:
1. Run the prompt optimization pipeline with a specified model and 3 repetitions.
2. Read the results of each repetition.
3. Average the baseline and refined scores across the repetitions.
4. Output the final summary metrics to the console and to a JSON file.

Usage:
    python scripts/run_final_benchmark.py --models qwen3-coder-next:cloud,nemotron-3-super:cloud --repetitions 3
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PIPELINE = _HERE.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="qwen3-coder-next:cloud")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--mutations", type=int, default=2)
    parser.add_argument("--experiment", type=str, default="final_benchmark")
    
    args = parser.parse_args()

    # 1. Run the optimization pipeline
    print(f"Running final benchmark for models: {args.models} with {args.repetitions} repetitions.")
    cmd = [
        sys.executable,
        str(_HERE / "run_optimization.py"),
        "--models", args.models,
        "--generations", str(args.generations),
        "--mutations", str(args.mutations),
        "--repetitions", str(args.repetitions),
        "--experiment", args.experiment,
        "--dataset", str(_PIPELINE / "Dataset" / "benchmark_mbpp_mini_a.jsonl")
    ]
    
    if "," in args.models:
        cmd.append("--ensemble")
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # 2. Analyze the results
    # We will use the Analysis scripts to compute the base vs refined gap for each repetition
    # and then average them.
    
    analysis_script = str(_PIPELINE / "Analysis" / "generate_analysis.py")
    all_metrics = []
    
    for rep in range(1, args.repetitions + 1):
        if "," in args.models:
            exp_name = f"{args.experiment}_rep{rep}"
        else:
            exp_name = f"{args.experiment}_rep{rep}_{args.models.replace(':', '_').replace('-', '_')}"
        
        # Run analysis to generate summary.json
        print(f"Analyzing repetition {rep}...")
        subprocess.run([sys.executable, analysis_script, "--experiment", exp_name], check=True)
        
        summary_path = _PIPELINE / "Analysis" / "output" / exp_name / "summary.json"
        if not summary_path.exists():
            print(f"Warning: could not find summary at {summary_path}")
            continue
            
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        metrics = data.get("metrics", {})
        all_metrics.append(metrics)

    if not all_metrics:
        print("No metrics collected.")
        return
        
    # 3. Average the results per model
    model_stats = defaultdict(lambda: {"base": [], "refined": [], "delta": []})
    for m in all_metrics:
        comp = m.get("held_out_comparison", {})
        if comp.get("complete"):
            for m_row in comp.get("models", []):
                model_name = m_row.get("model")
                model_stats[model_name]["base"].append(m_row.get("base_score", 0))
                model_stats[model_name]["refined"].append(m_row.get("refined_score", 0))
                model_stats[model_name]["delta"].append(m_row.get("absolute_delta", 0))

    avg_metrics = {}
    for model_name, stats in model_stats.items():
        avg_metrics[model_name] = {
            "baseline_test_score": sum(stats["base"]) / len(stats["base"]) if stats["base"] else 0,
            "refined_test_score": sum(stats["refined"]) / len(stats["refined"]) if stats["refined"] else 0,
            "held_out_improvement": sum(stats["delta"]) / len(stats["delta"]) if stats["delta"] else 0
        }
    
    # 4. Output final summary
    print("\n" + "=" * 80)
    print("  Final Benchmark Summary")
    print("=" * 80)
    print(f"Models: {args.models}")
    print(f"Repetitions: {args.repetitions}")
    print("-" * 80)
    print(f"{'Model':<30} | {'Base Pass':<10} | {'Refined Pass':<12} | {'Improvement':<10}")
    print("-" * 80)
    
    for model_name, m_avg in avg_metrics.items():
        base = m_avg['baseline_test_score'] * 100
        refined = m_avg['refined_test_score'] * 100
        imp = m_avg['held_out_improvement'] * 100
        print(f"{model_name:<30} | {base:>8.2f}% | {refined:>10.2f}% | {imp:>+8.2f} pp")
        
    print("=" * 80)
    
    out_path = _PIPELINE / "experiments" / "artifacts" / args.experiment / "final_benchmark_summary.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "models": args.models,
            "repetitions": args.repetitions,
            "averages_per_model": avg_metrics
        }, f, indent=2)
    print(f"Summary written to {out_path}")

if __name__ == "__main__":
    main()
