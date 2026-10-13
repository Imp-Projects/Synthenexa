"""
Analysis/generate_analysis.py — Structured experiment analysis pipeline.

Builds a clean analysis package for one experiment:
  - extracts results from SQLite or artifact folders
  - computes prompt-refinement metrics
  - generates organized PNG visualizations
  - writes a JSON summary for downstream reporting/UI use

Usage:
    python Analysis/generate_analysis.py
    python Analysis/generate_analysis.py --experiment realistic_run
    python Analysis/generate_analysis.py --run-id c302c699 --source sqlite
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("[ERROR] matplotlib is required. Install it with: pip install matplotlib")

_HERE = Path(__file__).resolve().parent
_PIPELINE = _HERE.parent
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from Analysis.compare_baseline import build_test_comparison
from services.results_repository import default_repository


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_") or "experiment"


def _normalize_experiment(result) -> dict[str, Any]:
    prompt_lookup: dict[str, dict[str, Any]] = {}
    for row in result.prompts:
        pid = row.get("prompt_id") or row.get("id")
        if not pid:
            continue
        prompt_lookup[pid] = {
            "prompt_id": pid,
            "generation": row.get("generation") or _infer_generation(pid),
            "text": row.get("text", ""),
            "origin_model": row.get("origin_model") or _infer_origin_model(pid),
            "is_best": bool(row.get("is_best", False)),
        }

    prompt_model_scores: dict[str, dict[str, float]] = defaultdict(dict)
    prompt_eval_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result.evaluations:
        prompt_id = row.get("prompt_id")
        if not prompt_id:
            continue
        total = row.get("total", 0) or 0
        passed = row.get("passed", 0) or 0
        pass_rate = (passed / total) if total else 0.0
        model_name = row.get("evaluator_model") or ""
        if model_name:
            prompt_model_scores[prompt_id][model_name] = pass_rate
        prompt_eval_stats[prompt_id].append(
            {
                "model": model_name,
                "pass_rate": pass_rate,
                "passed": passed,
                "failed": row.get("failed", 0) or 0,
                "total": total,
                "syntax_errors": row.get("syntax_errors", 0) or 0,
                "avg_time_ms": row.get("avg_time_ms", 0.0) or 0.0,
            }
        )

    prompt_aggregate_scores: dict[str, float] = {}
    generation_rows = []
    for snap in result.snapshots:
        generation = snap.get("generation")
        scores = snap.get("scores") or {}
        if isinstance(scores, str):
            try:
                scores = json.loads(scores)
            except json.JSONDecodeError:
                scores = {}
        snap["scores"] = scores
        snap["ensemble_scores"] = snap.get("ensemble_scores") or {}
        snap["model_leaderboard"] = snap.get("model_leaderboard") or {}
        snap["best_ensemble_scores"] = snap.get("best_ensemble_scores") or {}
        if scores:
            best_prompt_id, best_score = max(scores.items(), key=lambda item: item[1])
            generation_rows.append(
                {
                    "generation": generation,
                    "candidate_count": len(scores),
                    "mean_score": sum(scores.values()) / len(scores),
                    "best_score": best_score,
                    "best_prompt_id": best_prompt_id,
                }
            )
            for prompt_id, score in scores.items():
                prompt_aggregate_scores[prompt_id] = score

    for snap in result.snapshots:
        ensemble_scores = snap.get("ensemble_scores") or {}
        if isinstance(ensemble_scores, str):
            try:
                ensemble_scores = json.loads(ensemble_scores)
            except json.JSONDecodeError:
                ensemble_scores = {}
        for prompt_id, model_scores in ensemble_scores.items():
            if not isinstance(model_scores, dict):
                continue
            for model_name, score in model_scores.items():
                prompt_model_scores[prompt_id][model_name] = float(score)

    prompt_rows = []
    for prompt_id, aggregate_score in prompt_aggregate_scores.items():
        model_scores = prompt_model_scores.get(prompt_id, {})
        generation = prompt_lookup.get(prompt_id, {}).get("generation", 0)
        stats = prompt_eval_stats.get(prompt_id, [])
        min_score = min(model_scores.values()) if model_scores else aggregate_score
        max_score = max(model_scores.values()) if model_scores else aggregate_score
        mean_score = (sum(model_scores.values()) / len(model_scores)) if model_scores else aggregate_score
        stddev = _stddev(list(model_scores.values()))
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "generation": generation,
                "text": prompt_lookup.get(prompt_id, {}).get("text", ""),
                "origin_model": prompt_lookup.get(prompt_id, {}).get("origin_model", ""),
                "aggregate_score": aggregate_score,
                "mean_model_score": mean_score,
                "min_model_score": min_score,
                "max_model_score": max_score,
                "score_stddev": stddev,
                "score_spread": max_score - min_score,
                "model_scores": model_scores,
                "eval_stats": stats,
                "is_best": prompt_id == result.best_prompt.get("prompt_id"),
            }
        )
    prompt_rows.sort(key=lambda row: (row["aggregate_score"], row["min_model_score"]), reverse=True)

    models = sorted({model for row in prompt_rows for model in row["model_scores"]})
    reward_history = result.best_prompt.get("reward_history", {})
    final_test_rows = list(getattr(result, "final_test_evaluations", []))
    return {
        "run_id": result.run_id,
        "experiment_name": result.experiment_name,
        "source": result.source,
        "model_name": result.model_name,
        "config": result.config,
        "best_prompt": result.best_prompt,
        "prompt_rows": prompt_rows,
        "generation_rows": generation_rows,
        "reward_history": reward_history,
        "models": models,
        "final_test_evaluations": final_test_rows,
    }


def _infer_origin_model(prompt_id: str) -> str:
    parts = prompt_id.split("_")
    if len(parts) < 5:
        return "base"
    return "_".join(parts[4:])


def _infer_generation(prompt_id: str) -> int:
    marker = "_gen"
    if marker not in prompt_id:
        return 0
    try:
        remainder = prompt_id.split(marker, 1)[1]
        return int(remainder.split("_", 1)[0])
    except (IndexError, ValueError):
        return 0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _build_metrics(data: dict[str, Any]) -> dict[str, Any]:
    generation_rows = data["generation_rows"]
    prompt_rows = data["prompt_rows"]
    if generation_rows:
        first_best = generation_rows[0]["best_score"]
        final_best = generation_rows[-1]["best_score"]
    else:
        first_best = 0.0
        final_best = data["best_prompt"].get("score", 0.0)

    top_prompts = prompt_rows[:10]
    origin_counter = Counter(row["origin_model"] or "unknown" for row in prompt_rows[: min(10, len(prompt_rows))])
    model_wins = Counter()
    for row in prompt_rows:
        if not row["model_scores"]:
            continue
        strongest_model = max(row["model_scores"].items(), key=lambda item: item[1])[0]
        model_wins[strongest_model] += 1

    try:
        test_comparison = build_test_comparison(
            data["final_test_evaluations"], data["best_prompt"].get("prompt_id", "")
        )
        comparison_status = "complete"
    except ValueError as exc:
        test_comparison = {"complete": False, "error": str(exc)}
        comparison_status = "incomplete"

    return {
        "final_best_score": data["best_prompt"].get("score", final_best),
        "generation_best_start": first_best,
        "generation_best_end": final_best,
        "generation_best_delta": final_best - first_best,
        "prompt_count": len(prompt_rows),
        "generation_count": len(generation_rows),
        "top_prompt_count": len(top_prompts),
        "average_top_prompt_score": mean([row["aggregate_score"] for row in top_prompts]) if top_prompts else 0.0,
        "average_top_prompt_floor": mean([row["min_model_score"] for row in top_prompts]) if top_prompts else 0.0,
        "origin_model_top10_counts": dict(origin_counter),
        "model_specific_win_counts": dict(model_wins),
        "held_out_comparison_status": comparison_status,
        "held_out_comparison": test_comparison,
        "baseline_test_score": test_comparison.get("base_mean"),
        "refined_test_score": test_comparison.get("refined_mean"),
        "held_out_improvement": test_comparison.get("absolute_delta"),
        "model_score_spread": test_comparison.get("delta_stddev"),
        "train_test_gap": (
            test_comparison.get("refined_mean") - data["best_prompt"].get("score", 0.0)
            if test_comparison.get("refined_mean") is not None
            else None
        ),
    }


def _plot_base_vs_refined(comparison: dict[str, Any], output_path: Path) -> None:
    rows = comparison.get("models", [])
    if not rows:
        return
    labels = [row["model"] for row in rows]
    base = [row["base_score"] for row in rows]
    refined = [row["refined_score"] for row in rows]
    positions = list(range(len(rows)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_aesthetics(ax, "Held-Out Test: Original vs Refined Prompt", ylabel="Pass Rate")
    ax.bar([p - width / 2 for p in positions], base, width=width, label="Base prompt", color="#F7931E")
    ax.bar([p + width / 2 for p in positions], refined, width=width, label="Refined prompt", color="#00C9A7")
    for position, row in zip(positions, rows):
        ax.text(
            position + width / 2,
            row["refined_score"] + 0.02,
            f"{row['absolute_delta']:+.1%}",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="#FAFAFA", bbox_inches="tight")
    plt.close(fig)


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


def _plot_error_topology(prompt_rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = list(reversed(prompt_rows[:10]))
    labels = [_prompt_label(row["prompt_id"]) for row in rows]

    pass_pcts, fail_pcts, syntax_pcts = [], [], []
    for row in rows:
        passed = sum(stat.get("passed", 0) for stat in row["eval_stats"])
        total = sum(stat.get("total", 0) for stat in row["eval_stats"])
        syntax = sum(stat.get("syntax_errors", 0) for stat in row["eval_stats"])
        failed = sum(stat.get("failed", 0) for stat in row["eval_stats"])
        logic_and_runtime = max(0, failed - syntax)

        if total == 0:
            pass_pcts.append(0)
            fail_pcts.append(0)
            syntax_pcts.append(0)
        else:
            pass_pcts.append(passed / total * 100)
            fail_pcts.append(logic_and_runtime / total * 100)
            syntax_pcts.append(syntax / total * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_aesthetics(ax, "Error Topology Breakdown (Top 10 Prompts)", xlabel="Percentage (%)")

    ax.barh(labels, pass_pcts, color="#10B981", label="Pass", height=0.6)
    ax.barh(labels, fail_pcts, left=pass_pcts, color="#F43F5E", label="Logic/Runtime Fail", height=0.6)
    left_syntax = [p + f for p, f in zip(pass_pcts, fail_pcts)]
    ax.barh(labels, syntax_pcts, left=left_syntax, color="#F59E0B", label="Syntax Error", height=0.6)

    ax.grid(axis="x", color="#EAEAEA", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=9, labelcolor="#555555")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="#FAFAFA", bbox_inches="tight")
    plt.close(fig)


def _plot_execution_time(prompt_rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = prompt_rows[:10]
    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_aesthetics(ax, "Execution Time Performance", ylabel="Average Time (ms)")

    labels = []
    has_data = False
    for i, row in enumerate(rows):
        times = [stat.get("avg_time_ms", 0) for stat in row["eval_stats"] if stat.get("avg_time_ms", 0) > 0]
        if not times:
            continue
        has_data = True
        labels.append(_prompt_label(row["prompt_id"]))

        x_vals = [i + 1] * len(times)
        ax.scatter(x_vals, times, color="#6366F1", alpha=0.6, s=60, edgecolors="white", linewidth=0.5, zorder=2)

        mean_time = sum(times) / len(times)
        ax.hlines(mean_time, i + 0.7, i + 1.3, color="#312E81", linewidth=3, zorder=3)

    if not has_data:
        plt.close(fig)
        return

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", color="#EAEAEA", linestyle="-", linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="#FAFAFA", bbox_inches="tight")
    plt.close(fig)


def _plot_cross_model_robustness(prompt_rows: list[dict[str, Any]], models: list[str], output_path: Path) -> None:
    if len(models) < 2:
        return

    rows = list(reversed(prompt_rows[:10]))
    labels = [_prompt_label(row["prompt_id"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_aesthetics(ax, "Cross-Model Robustness (Score Spread)", xlabel="Pass Rate")

    model_a, model_b = models[0], models[1]

    for i, row in enumerate(rows):
        scores = row["model_scores"]
        score_a = scores.get(model_a, 0)
        score_b = scores.get(model_b, 0)

        ax.plot([score_a, score_b], [i, i], color="#CBD5E1", linewidth=2, zorder=1)
        ax.scatter([score_a], [i], color="#3B82F6", s=80, zorder=2, label=_short_model_name(model_a) if i == 0 else "")
        ax.scatter([score_b], [i], color="#F59E0B", s=80, zorder=2, label=_short_model_name(model_b) if i == 0 else "")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", color="#EAEAEA", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=9, labelcolor="#555555")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="#FAFAFA", bbox_inches="tight")
    plt.close(fig)


def _plot_reward_progression(generation_rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_aesthetics(ax, "Reward Progression Over Generations", xlabel="Generation", ylabel="Best Score")

    gens = [row["generation"] for row in generation_rows]
    best_scores = [row["best_score"] for row in generation_rows]

    ax.plot(
        gens,
        best_scores,
        color="#0EA5E9",
        linewidth=3,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        zorder=3,
    )
    ax.fill_between(gens, best_scores, 0, color="#0EA5E9", alpha=0.1, zorder=2)

    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#EAEAEA", linestyle="-", linewidth=0.8, alpha=0.7)

    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="#FAFAFA", bbox_inches="tight")
    plt.close(fig)


def _plot_model_leaderboard(metrics: dict[str, Any], output_path: Path) -> None:
    counts = metrics.get("origin_model_top10_counts", {})
    if not counts:
        return

    sorted_models = sorted(counts.items(), key=lambda x: x[1])
    labels = [_short_model_name(k) for k, v in sorted_models]
    values = [v for k, v in sorted_models]

    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_aesthetics(ax, "Top 10 Prompts by Origin Model", xlabel="Number of Prompts")

    for i, (label, val) in enumerate(zip(labels, values)):
        ax.plot([0, val], [i, i], color="#94A3B8", linewidth=2, zorder=1)
        ax.scatter([val], [i], color="#8B5CF6", s=120, zorder=2)
        ax.text(val + 0.2, i, str(val), va="center", color="#555555", fontweight="bold", fontsize=10)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(values) + 1.5)

    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="#FAFAFA", bbox_inches="tight")
    plt.close(fig)


def _prompt_label(pid: str) -> str:
    """
    Convert a raw chained prompt ID into a compact, unique label.

    Examples:
        prompt_0_gen1_m1_gpt_oss_120b_cloud                      -> G1-M1 (gpt-oss-120b)
        prompt_0_gen1_m1_..._gen4_m0_nemotron_3_super_cloud      -> G4-M0 (nemotron-3-super)
    """
    import re

    segments = re.findall(r"gen(\d+)_m(\d+)_([^_](?:[^g]|g(?!en\d))*)", pid)
    if not segments:
        return pid.replace("prompt_0_", "").replace("_cloud", "")[:22]
    gen_num, mut_idx, model_part = segments[-1]
    model_clean = model_part.rstrip("_").replace("_cloud", "").replace("_", "-").strip("-")
    return f"G{gen_num}-M{mut_idx} ({model_clean})"


def _short_model_name(value: str) -> str:
    return value.replace(":cloud", "").replace("_", "-")


def _write_summary(data: dict[str, Any], metrics: dict[str, Any], output_path: Path) -> None:
    payload = {
        "run_id": data["run_id"],
        "experiment_name": data["experiment_name"],
        "source": data["source"],
        "model_name": data["model_name"],
        "config": data["config"],
        "best_prompt": data["best_prompt"],
        "metrics": metrics,
        "generation_rows": data["generation_rows"],
        "top_prompts": data["prompt_rows"][:10],
        "models": data["models"],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_output_dirs(base_output_dir: Path, experiment_name: str) -> dict[str, Path]:
    root = base_output_dir / _safe_slug(experiment_name)
    dirs = {
        "root": root,
        "overview": root / "overview",
        "prompt_wise": root / "prompt_wise",
        "model_wise": root / "model_wise",
        "ensemble_wise": root / "ensemble_wise",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate organized Synthenexa analysis outputs.")
    parser.add_argument("--experiment", help="Experiment name / artifact directory.")
    parser.add_argument("--run-id", help="Run identifier.")
    parser.add_argument("--source", choices=["auto", "sqlite", "artifacts"], default="auto")
    parser.add_argument("--output-dir", default=str(_HERE / "output"))
    parser.add_argument("--list-experiments", action="store_true")
    args = parser.parse_args()

    repo = default_repository()
    if args.list_experiments:
        print(json.dumps(repo.list_experiments(source=args.source), indent=2, default=str))
        return

    result = repo.load_experiment(
        experiment_name=args.experiment,
        run_id=args.run_id,
        source=args.source,
    )
    data = _normalize_experiment(result)
    metrics = _build_metrics(data)
    dirs = _build_output_dirs(Path(args.output_dir), data["experiment_name"])

    if data["generation_rows"]:
        _plot_reward_progression(data["generation_rows"], dirs["overview"] / "reward_progression.png")

    if data["prompt_rows"]:
        _plot_error_topology(data["prompt_rows"], dirs["prompt_wise"] / "error_topology.png")
        _plot_execution_time(data["prompt_rows"], dirs["prompt_wise"] / "execution_time.png")
        _plot_model_leaderboard(metrics, dirs["model_wise"] / "model_leaderboard.png")

    if data["models"] and data["prompt_rows"]:
        _plot_cross_model_robustness(
            data["prompt_rows"], data["models"], dirs["ensemble_wise"] / "cross_model_robustness.png"
        )

    if metrics["held_out_comparison_status"] == "complete":
        _plot_base_vs_refined(metrics["held_out_comparison"], dirs["overview"] / "held_out_base_vs_refined.png")
    else:
        print(f"[analysis] WARNING: held-out comparison incomplete: {metrics['held_out_comparison'].get('error')}")

    _write_summary(data, metrics, dirs["root"] / "summary.json")

    print(f"[analysis] Generated outputs for '{data['experiment_name']}'")
    print(f"[analysis] Source: {data['source']}")
    print(f"[analysis] Output: {dirs['root']}")


if __name__ == "__main__":
    main()
