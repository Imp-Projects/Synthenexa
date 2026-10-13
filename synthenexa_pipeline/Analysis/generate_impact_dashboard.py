#!/usr/bin/env python3
"""Generate a compact, offline impact dashboard from artifact analysis.

Unlike ``compare_baseline.py``, this script never opens SQLite and never calls
an LLM. It consumes the ``summary.json`` produced by ``generate_analysis.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_dashboard(summary_path: Path, output_path: Path) -> None:
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    comparison = summary.get("metrics", {}).get("held_out_comparison", {})
    if not comparison.get("complete"):
        raise ValueError("summary.json does not contain a complete held-out comparison")

    rows = comparison.get("models", [])
    if not rows:
        raise ValueError("held-out comparison contains no model rows")

    labels = [row["model"] for row in rows]
    base = [row["base_score"] * 100 for row in rows]
    refined = [row["refined_score"] * 100 for row in rows]
    deltas = [row["absolute_delta"] * 100 for row in rows]
    task_count = len(rows[0].get("task_ids", []))

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#1A1A2E")
    for axis in axes.flat:
        axis.set_facecolor("#16213E")
        axis.tick_params(colors="#E0E0E0", labelsize=9)
        for spine in axis.spines.values():
            spine.set_color("#2D2D4E")
        axis.title.set_color("#E0E0E0")
        axis.yaxis.label.set_color("#8E8EA0")

    x = list(range(len(labels)))
    width = 0.36
    axes[0, 0].bar([i - width / 2 for i in x], base, width, label="Base", color="#F7931E")
    axes[0, 0].bar([i + width / 2 for i in x], refined, width, label="Refined", color="#00C9A7")
    axes[0, 0].set_title("Held-out pass rate")
    axes[0, 0].set_ylabel("Pass rate (%)")
    axes[0, 0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0, 0].set_ylim(0, 110)
    axes[0, 0].legend(frameon=False, labelcolor="#E0E0E0")

    bars = axes[0, 1].bar(labels, deltas, color=["#00C9A7" if d >= 0 else "#FF6B6B" for d in deltas])
    axes[0, 1].axhline(0, color="#8E8EA0", linewidth=1)
    axes[0, 1].set_title("Improvement by model")
    axes[0, 1].set_ylabel("Delta (percentage points)")
    axes[0, 1].tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, deltas):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, value + (1 if value >= 0 else -3),
                         f"{value:+.1f} pp", ha="center", color="#E0E0E0", fontsize=9)

    axes[1, 0].plot(labels, base, marker="o", color="#F7931E", label="Base")
    axes[1, 0].plot(labels, refined, marker="o", color="#00C9A7", label="Refined")
    axes[1, 0].set_title("Cross-model consistency")
    axes[1, 0].set_ylabel("Pass rate (%)")
    axes[1, 0].set_ylim(0, 110)
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].legend(frameon=False, labelcolor="#E0E0E0")

    axes[1, 1].axis("off")
    aggregate = comparison
    text = (
        "HELD-OUT EVIDENCE\n\n"
        f"Tasks evaluated: {task_count or 'unknown'}\n"
        f"Models: {len(rows)}\n\n"
        f"Base mean:    {aggregate['base_mean'] * 100:.1f}%\n"
        f"Refined mean: {aggregate['refined_mean'] * 100:.1f}%\n"
        f"Improvement:  {aggregate['absolute_delta'] * 100:+.1f} pp\n"
        f"Model spread: {aggregate.get('delta_stddev', 0) * 100:.1f} pp"
    )
    axes[1, 1].text(0.08, 0.82, text, va="top", color="#E0E0E0", fontsize=14,
                    family="monospace", linespacing=1.6)

    figure.suptitle("Synthenexa Prompt Refinement Impact", color="#FFFFFF", fontsize=17, fontweight="bold")
    figure.text(0.5, 0.02, "Strict held-out comparison; no live model calls", ha="center", color="#8E8EA0")
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an offline artifact-native impact dashboard.")
    parser.add_argument("--experiment", required=True, help="Artifact experiment directory name")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    summary_path = root / "output" / args.experiment / "summary.json"
    output_path = root / "output" / args.experiment / "impact_dashboard.png"
    generate_dashboard(summary_path, output_path)
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
