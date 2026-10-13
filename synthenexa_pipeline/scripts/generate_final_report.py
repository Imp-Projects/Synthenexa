#!/usr/bin/env python3
"""
scripts/generate_final_report.py

Phase D: Final Simplified Evaluation Assets
Reads the averaged metrics from a final benchmark run and generates:
1. A consolidated benchmark comparison chart (Base vs Refined).
2. An improvement summary chart (Delta).
3. A compact comparison table (Markdown).
Saves all artifacts in the run directory.

Usage:
    python scripts/generate_final_report.py --experiment final_eval
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    sys.exit("[ERROR] matplotlib is required. Install it with: pip install matplotlib")

_HERE = Path(__file__).resolve().parent
_PIPELINE = _HERE.parent


def _apply_aesthetics(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor("#FAFAFA")
    ax.figure.patch.set_facecolor("#FAFAFA")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#EAEAEA")
    ax.tick_params(axis="both", colors="#555555", length=0, labelsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#333333", pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color="#666666", labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#666666", labelpad=8)


def generate_comparison_chart(data, output_path):
    models = list(data["averages_per_model"].keys())
    base_scores = [data["averages_per_model"][m]["baseline_test_score"] * 100 for m in models]
    refined_scores = [data["averages_per_model"][m]["refined_test_score"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_aesthetics(ax, "Consolidated Benchmark Comparison", ylabel="Pass Rate (%)")

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, base_scores, width, label='Base Prompt', color='#F7931E')
    ax.bar(x + width/2, refined_scores, width, label='Refined Prompt', color='#00C9A7')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False)

    for i, (base, refined) in enumerate(zip(base_scores, refined_scores)):
        ax.text(i - width/2, base + 2, f"{base:.1f}%", ha='center', fontsize=9, color="#333333")
        ax.text(i + width/2, refined + 2, f"{refined:.1f}%", ha='center', fontsize=9, color="#333333")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="#FAFAFA", bbox_inches="tight")
    plt.close(fig)


def generate_improvement_chart(data, output_path):
    models = list(data["averages_per_model"].keys())
    improvements = [data["averages_per_model"][m]["held_out_improvement"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_aesthetics(ax, "Improvement Summary (Refined - Base)", ylabel="Delta (Percentage Points)")

    colors = ['#00C9A7' if val >= 0 else '#FF6B6B' for val in improvements]
    bars = ax.bar(models, improvements, color=colors, width=0.5)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right")
    
    y_min = min(improvements + [0])
    y_max = max(improvements + [0])
    padding = max((y_max - y_min) * 0.15, 5)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.axhline(0, color='#333333', linewidth=1)

    for bar, val in zip(bars, improvements):
        sign = "+" if val > 0 else ""
        y_pos = val + 1 if val >= 0 else val - 3
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{sign}{val:.1f} pp", 
                ha='center', fontsize=10, fontweight='bold', color=bar.get_facecolor())

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="#FAFAFA", bbox_inches="tight")
    plt.close(fig)


def generate_markdown_table(data, output_path):
    models = list(data["averages_per_model"].keys())
    
    md = [
        f"# Final Evaluation Summary: {data.get('experiment', 'Experiment')}",
        f"**Repetitions Averaged:** {data.get('repetitions', 1)}\n",
        "| Model | Base Pass Rate | Refined Pass Rate | Improvement |",
        "|-------|----------------|-------------------|-------------|"
    ]
    
    for m in models:
        base = data["averages_per_model"][m]["baseline_test_score"] * 100
        refined = data["averages_per_model"][m]["refined_test_score"] * 100
        imp = data["averages_per_model"][m]["held_out_improvement"] * 100
        sign = "+" if imp > 0 else ""
        md.append(f"| `{m}` | {base:.1f}% | {refined:.1f}% | **{sign}{imp:.1f} pp** |")

    if data.get("aggregate"):
        aggregate = data["aggregate"]
        base = aggregate["baseline_test_score"] * 100
        refined = aggregate["refined_test_score"] * 100
        imp = aggregate["held_out_improvement"] * 100
        sign = "+" if imp > 0 else ""
        md.append(f"| **Aggregate** | **{base:.1f}%** | **{refined:.1f}%** | **{sign}{imp:.1f} pp** |")
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def _load_report_data(run_dir: Path, experiment: str) -> dict:
    """Load either the repeated-run summary or the current artifact analysis."""
    benchmark_path = run_dir / "final_benchmark_summary.json"
    if benchmark_path.exists():
        with open(benchmark_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["experiment"] = experiment
        return data

    analysis_path = _PIPELINE / "Analysis" / "output" / experiment / "summary.json"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Could not find {benchmark_path} or artifact analysis summary {analysis_path}"
        )

    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    comparison = analysis.get("metrics", {}).get("held_out_comparison")
    if not comparison or not comparison.get("complete"):
        raise ValueError("Artifact analysis does not contain a complete held-out comparison.")

    averages = {
        row["model"]: {
            "baseline_test_score": row["base_score"],
            "refined_test_score": row["refined_score"],
            "held_out_improvement": row["absolute_delta"],
        }
        for row in comparison["models"]
    }
    return {
        "experiment": experiment,
        "models": analysis.get("model_name", ""),
        "repetitions": 1,
        "averages_per_model": averages,
        "aggregate": {
            "baseline_test_score": comparison["base_mean"],
            "refined_test_score": comparison["refined_mean"],
            "held_out_improvement": comparison["absolute_delta"],
        },
        "source": "artifact_analysis",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name (e.g., final_eval)")
    args = parser.parse_args()

    run_dir = _PIPELINE / "experiments" / "artifacts" / args.experiment
    try:
        data = _load_report_data(run_dir, args.experiment)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"[ERROR] {exc}")

    print(f"Generating Phase D assets for '{args.experiment}' in {run_dir}...")

    # 1. Consolidated benchmark comparison chart
    chart1_path = run_dir / "consolidated_comparison.png"
    generate_comparison_chart(data, chart1_path)
    print(f"  ✓ Created {chart1_path.name}")

    # 2. Improvement summary chart
    chart2_path = run_dir / "improvement_summary.png"
    generate_improvement_chart(data, chart2_path)
    print(f"  ✓ Created {chart2_path.name}")

    # 3. Compact comparison table
    table_path = run_dir / "comparison_table.md"
    generate_markdown_table(data, table_path)
    print(f"  ✓ Created {table_path.name}")

    if data.get("aggregate"):
        aggregate = data["aggregate"]
        print(
            "  Aggregate held-out result: "
            f"{aggregate['baseline_test_score'] * 100:.1f}% → "
            f"{aggregate['refined_test_score'] * 100:.1f}% "
            f"({aggregate['held_out_improvement'] * 100:+.1f} pp)"
        )
    
    print("\nDone! Phase D evaluation assets are ready.")

if __name__ == "__main__":
    main()
