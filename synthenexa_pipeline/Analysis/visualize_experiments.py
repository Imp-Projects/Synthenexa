"""
Analysis/visualize_experiments.py — Synthenexa Experiment Visualizations
==========================================================================

Generates publication-quality plots directly from the SQLite experiment
database (`experiments/synthenexa.db`).

Visualizations produced:
  1. Prompt Score Distribution     — Bar chart of all prompt ensemble scores
  2. Model Comparison Heatmap      — Per-prompt pass-rate across each evaluator model
  3. Cross-Model Generalization    — Scatter: score on Model A vs Model B per prompt
  4. Ensemble Score Leaderboard    — Ranked horizontal bars by origin model
  5. Per-Model Pass/Fail Breakdown — Stacked bar (passed / failed / syntax errors)
  6. Generation Evolution          — Line chart of best score per generation

Usage (run from synthenexa_pipeline/):
    python Analysis/visualize_experiments.py
    python Analysis/visualize_experiments.py --db experiments/synthenexa.db --experiment-id 2
    python Analysis/visualize_experiments.py --list-experiments

Requirements:
    pip install matplotlib seaborn
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from textwrap import shorten
from typing import Optional

# ── Graceful import check ────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("[ERROR] matplotlib is required. Run: pip install matplotlib seaborn")

try:
    import seaborn as sns
except ImportError:
    sys.exit("[ERROR] seaborn is required. Run: pip install matplotlib seaborn")

# ── Constants ────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
_PIPELINE   = _HERE.parent
_DEFAULT_DB = _PIPELINE / "experiments" / "synthenexa.db"
_OUTPUT_DIR = _HERE / "output"

PALETTE = {
    "primary":   "#6C63FF",
    "secondary": "#F7931E",
    "accent":    "#00C9A7",
    "danger":    "#FF6B6B",
    "neutral":   "#8E8EA0",
    "bg":        "#1A1A2E",
    "surface":   "#16213E",
    "text":      "#E0E0E0",
}

MODEL_COLORS = [
    "#6C63FF", "#F7931E", "#00C9A7", "#FF6B6B",
    "#A855F7", "#3B82F6", "#F59E0B", "#10B981",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        sys.exit(f"[ERROR] Database not found at: {db_path}\n"
                 "Run the pipeline with --storage sqlite first.")
    return sqlite3.connect(db_path)


def _pick_experiment(con: sqlite3.Connection, experiment_id: Optional[int]) -> dict:
    cur = con.cursor()
    if experiment_id is not None:
        cur.execute(
            "SELECT id, run_id, experiment_name, model_name, config FROM experiments WHERE id = ?",
            (experiment_id,),
        )
    else:
        cur.execute(
            "SELECT id, run_id, experiment_name, model_name, config "
            "FROM experiments WHERE status = 'completed' ORDER BY id DESC LIMIT 1"
        )
    row = cur.fetchone()
    if not row:
        sys.exit("[ERROR] No completed experiment found in the database.")
    return {
        "id":              row[0],
        "run_id":          row[1],
        "experiment_name": row[2],
        "model_name":      row[3],
        "config":          json.loads(row[4]) if row[4] else {},
    }


def _short_prompt(pid: str, max_len: int = 28) -> str:
    parts = pid.replace("prompt_0_", "").split("_")
    label = "_".join(parts[:4])
    return shorten(label, width=max_len, placeholder="…")


def _model_label(raw: str) -> str:
    return raw.replace("_", "-").replace(":cloud", "").strip("-")


def _apply_dark_style(fig, axes_list):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in axes_list:
        ax.set_facecolor(PALETTE["surface"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor("#2D2D4E")


# ═══════════════════════════════════════════════════════════════════════════
#  Plot functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_score_distribution(ax, scores: dict):
    """
    Plot 1 — Prompt Score Distribution
    Horizontal bar chart of ensemble scores per prompt variant.
    Answers: Which prompt variants performed best overall?
    """
    labels = [_short_prompt(k) for k in scores]
    values = list(scores.values())
    mean_v = sum(values) / len(values) if values else 0

    bar_colors = [PALETTE["accent"] if v == max(values) else PALETTE["primary"] for v in values]
    bars = ax.barh(labels, values, color=bar_colors, edgecolor="#2D2D4E", linewidth=0.6, height=0.6)

    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.2%}", va="center", ha="left",
                color=PALETTE["text"], fontsize=8.5, fontweight="bold")

    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Ensemble Score (Mean Pass Rate Across All Models)", fontsize=9)
    ax.set_title("① Prompt Ensemble Score Distribution", fontsize=11, fontweight="bold", pad=10)
    ax.axvline(mean_v, color=PALETTE["secondary"], linestyle="--", linewidth=1.2,
               label=f"Mean: {mean_v:.2%}")
    ax.legend(fontsize=8, labelcolor=PALETTE["text"], facecolor=PALETTE["surface"],
              edgecolor="#2D2D4E")
    ax.invert_yaxis()


def plot_model_heatmap(ax, eval_rows: list, prompt_ids: list, model_names: list):
    """
    Plot 2 — Cross-Model Evaluation Heatmap
    Rows = prompt variants, Cols = evaluator models.
    Answers: Does the winning prompt work well on ALL models?
    """
    import collections
    data = collections.defaultdict(dict)
    for row in eval_rows:
        pid, _, _, passed, failed, total, _, model = row
        data[pid][_model_label(model)] = passed / total if total > 0 else 0.0

    short_models = [_model_label(m) for m in model_names]
    matrix, ylabels = [], []
    for pid in prompt_ids:
        ylabels.append(_short_prompt(pid))
        matrix.append([data[pid].get(m, 0.0) for m in short_models])

    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    hm = sns.heatmap(
        matrix, ax=ax,
        xticklabels=short_models, yticklabels=ylabels,
        cmap=cmap, vmin=0, vmax=1,
        annot=True, fmt=".0%",
        linewidths=0.4, linecolor="#2D2D4E",
        annot_kws={"size": 8, "color": "#1A1A2E", "fontweight": "bold"},
        cbar_kws={"label": "Pass Rate", "shrink": 0.8},
    )
    hm.collections[0].colorbar.ax.yaxis.label.set_color(PALETTE["text"])
    hm.collections[0].colorbar.ax.tick_params(colors=PALETTE["text"])
    ax.set_title("② Cross-Model Evaluation Heatmap\n(High values across all columns = generalises well)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Evaluator Model", fontsize=9)
    ax.set_ylabel("Prompt Variant", fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    ax.tick_params(axis="y", rotation=0)


def plot_generalization_scatter(ax, eval_rows: list, prompt_ids: list, model_names: list, best_pid: str):
    """
    Plot 3 — Cross-Model Generalisation Scatter
    X = score on Model A, Y = score on Model B.
    Points near the diagonal generalise equally well on both models.
    Answers: Does the best prompt sit at the top-right (best on both)?
    """
    import collections
    if len(model_names) < 2:
        ax.text(0.5, 0.5, "Requires ≥2 models\nin the ensemble",
                ha="center", va="center", color=PALETTE["neutral"],
                transform=ax.transAxes, fontsize=11)
        ax.set_title("③ Cross-Model Generalisation Scatter", fontsize=11, fontweight="bold", pad=10)
        return

    data = collections.defaultdict(dict)
    for row in eval_rows:
        pid, _, _, passed, failed, total, _, model = row
        data[pid][model] = passed / total if total > 0 else 0.0

    m0, m1 = model_names[0], model_names[1]
    xs, ys, colors = [], [], []
    for pid in prompt_ids:
        xs.append(data[pid].get(m0, 0.0))
        ys.append(data[pid].get(m1, 0.0))
        colors.append(PALETTE["accent"] if pid == best_pid else PALETTE["primary"])

    ax.scatter(xs, ys, c=colors, s=130, edgecolors="#2D2D4E", linewidth=0.8, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["neutral"], linewidth=1.0,
            label="Perfect generalisation", zorder=2)

    for pid, x, y, c in zip(prompt_ids, xs, ys, colors):
        if c == PALETTE["accent"]:
            ax.annotate("★ Winner", (x, y), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=PALETTE["accent"], fontweight="bold")

    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel(_model_label(m0) + " Score", fontsize=9)
    ax.set_ylabel(_model_label(m1) + " Score", fontsize=9)
    ax.set_title("③ Cross-Model Generalisation\n(Top-right corner = best on both models)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=8, labelcolor=PALETTE["text"], facecolor=PALETTE["surface"],
              edgecolor="#2D2D4E")


def plot_leaderboard(ax, scores: dict, best_pid: str):
    """
    Plot 4 — Ensemble Leaderboard
    Ranked horizontal bars, coloured by origin model.
    Answers: Which model's mutations produced the best-scoring prompts?
    """
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1])
    raw_pids = [k for k, _ in sorted_items]
    labels   = [_short_prompt(k) for k in raw_pids]
    values   = [v for _, v in sorted_items]

    def _get_model_part(pid):
        parts = pid.split("_")
        return "_".join(parts[4:]) if len(parts) > 4 else "unknown"

    unique_models = sorted(set(_get_model_part(p) for p in raw_pids))
    color_map = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(unique_models)}
    bar_colors = [
        PALETTE["accent"] if pid == best_pid else color_map.get(_get_model_part(pid), PALETTE["primary"])
        for pid in raw_pids
    ]

    bars = ax.barh(labels, values, color=bar_colors, edgecolor="#2D2D4E", linewidth=0.5, height=0.55)
    for bar, val, pid in zip(bars, values, raw_pids):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.2%}", va="center", ha="left", color=PALETTE["text"], fontsize=8,
                fontweight="bold" if pid == best_pid else "normal")

    handles = [Patch(facecolor=color_map[m], label=_model_label(m)) for m in unique_models]
    handles.append(Patch(facecolor=PALETTE["accent"], label="Winner"))
    ax.legend(handles=handles, fontsize=7.5, labelcolor=PALETTE["text"],
              facecolor=PALETTE["surface"], edgecolor="#2D2D4E", loc="lower right")
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Ensemble Score", fontsize=9)
    ax.set_title("④ Ensemble Leaderboard — Prompt Rankings by Origin Model",
                 fontsize=11, fontweight="bold", pad=10)
    ax.invert_yaxis()


def plot_pass_fail_breakdown(ax, eval_rows: list, prompt_ids: list):
    """
    Plot 5 — Pass / Fail / Syntax Error Breakdown
    Stacked bar chart summed across all evaluator models.
    Answers: Why do some prompts score lower? (runtime errors vs syntax errors)
    """
    import collections
    data = collections.defaultdict(lambda: {"passed": 0, "failed": 0, "syntax": 0})
    for row in eval_rows:
        pid, _, _, passed, failed, total, syntax_err, _ = row
        data[pid]["passed"] += passed
        data[pid]["failed"] += max(0, failed - syntax_err)
        data[pid]["syntax"] += syntax_err

    labels  = [_short_prompt(p) for p in prompt_ids]
    passed  = [data[p]["passed"] for p in prompt_ids]
    failed  = [data[p]["failed"] for p in prompt_ids]
    syntax  = [data[p]["syntax"] for p in prompt_ids]
    x = range(len(labels))

    ax.bar(x, passed, label="Passed",       color=PALETTE["accent"],    edgecolor="#2D2D4E", lw=0.5)
    ax.bar(x, failed, label="Failed",       color=PALETTE["danger"],    edgecolor="#2D2D4E", lw=0.5,
           bottom=passed)
    ax.bar(x, syntax, label="Syntax Error", color=PALETTE["secondary"], edgecolor="#2D2D4E", lw=0.5,
           bottom=[p + f for p, f in zip(passed, failed)])

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("Test Cases (summed across all evaluator models)", fontsize=9)
    ax.set_title("⑤ Pass / Fail Breakdown per Prompt\n(Why do lower-ranked prompts fail?)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=8, labelcolor=PALETTE["text"], facecolor=PALETTE["surface"], edgecolor="#2D2D4E")


def plot_generation_evolution(ax, snapshots: list):
    """
    Plot 6 — Generation Evolution
    Line chart of best ensemble score per generation.
    This is the PRIMARY proof that the system actively refines prompts.
    """
    if not snapshots:
        ax.text(0.5, 0.5, "No snapshot data available",
                ha="center", va="center", color=PALETTE["neutral"], transform=ax.transAxes)
        ax.set_title("⑥ Prompt Evolution Over Generations", fontsize=11, fontweight="bold", pad=10)
        return

    gens   = [row[0] for row in snapshots]
    scores = [row[1] for row in snapshots]

    ax.plot(gens, scores, color=PALETTE["primary"], linewidth=2.5, marker="o",
            markersize=9, markerfacecolor=PALETTE["accent"], markeredgecolor="#2D2D4E",
            markeredgewidth=1.2, zorder=3)
    ax.fill_between(gens, scores, alpha=0.12, color=PALETTE["primary"])

    best_sc  = max(scores)
    best_gen = gens[scores.index(best_sc)]
    ax.annotate(f"Peak: {best_sc:.2%}", xy=(best_gen, best_sc),
                xytext=(best_gen + 0.1, best_sc + 0.02),
                arrowprops=dict(arrowstyle="->", color=PALETTE["accent"], lw=1.2),
                color=PALETTE["accent"], fontsize=9, fontweight="bold")

    ax.set_xlabel("Generation", fontsize=9)
    ax.set_ylabel("Best Ensemble Score", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title("⑥ Prompt Evolution Over Generations\n(Rising line = system is improving prompts)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.25, color=PALETTE["neutral"])


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(db_path: Path, experiment_id: Optional[int] = None):
    con = _connect(db_path)
    cur = con.cursor()

    exp = _pick_experiment(con, experiment_id)
    eid = exp["id"]
    cfg = exp["config"]
    model_names: list = cfg.get("ensemble_models", [])
    print(f"[Info] Visualising experiment #{eid}: '{exp['experiment_name']}' "
          f"(run_id={exp['run_id']})")
    print(f"[Info] Ensemble models: {model_names}")

    # Snapshots
    cur.execute(
        "SELECT generation, best_score, best_prompt_id, scores FROM snapshots "
        "WHERE experiment_id = ? ORDER BY generation",
        (eid,),
    )
    snap_rows = cur.fetchall()
    snapshots = [(r[0], r[1], r[2]) for r in snap_rows]

    scores_json = {}
    best_pid    = None
    if snap_rows:
        last = snap_rows[-1]
        best_pid    = last[2]
        scores_json = json.loads(last[3]) if last[3] else {}

    # Prompts
    cur.execute(
        "SELECT prompt_id FROM prompts WHERE experiment_id = ? ORDER BY id", (eid,)
    )
    prompt_ids = [r[0] for r in cur.fetchall()]
    if not prompt_ids:
        sys.exit("[ERROR] No prompt rows found for this experiment.")

    # Evaluations
    cur.execute(
        "SELECT prompt_id, experiment_id, generation, passed, failed, "
        "total, syntax_errors FROM evaluations WHERE experiment_id = ? ORDER BY id",
        (eid,),
    )
    raw_evals = cur.fetchall()

    def _infer_model(pid, candidates):
        pid_lower = pid.lower()
        for m in candidates:
            slug = m.replace(":", "_").replace("-", "_").replace(".", "_").lower()
            if slug in pid_lower:
                return m
        return candidates[0] if candidates else "unknown"

    eval_rows = [
        (r[0], r[1], r[2], r[3], r[4], r[5], r[6], _infer_model(r[0], model_names))
        for r in raw_evals
    ]
    con.close()

    # ── Layout ────────────────────────────────────────────────────────────
    sns.set_theme(style="dark", palette="muted")
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor(PALETTE["bg"])

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.38,
                           top=0.93, bottom=0.05, left=0.07, right=0.97)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    model_str = " · ".join(_model_label(m) for m in model_names) if model_names else exp["model_name"]
    fig.suptitle(
        f"Synthenexa — Prompt Evolution Analysis\n"
        f"Experiment: {exp['experiment_name']}  |  Run: {exp['run_id']}  |  Models: {model_str}",
        fontsize=13, fontweight="bold", color=PALETTE["text"], y=0.97,
    )

    plot_score_distribution(ax1, scores_json)
    plot_model_heatmap(ax2, eval_rows, prompt_ids, model_names)
    plot_generalization_scatter(ax3, eval_rows, prompt_ids, model_names, best_pid or "")
    plot_leaderboard(ax4, scores_json, best_pid or "")
    plot_pass_fail_breakdown(ax5, eval_rows, prompt_ids)
    plot_generation_evolution(ax6, snapshots)

    _apply_dark_style(fig, [ax1, ax2, ax3, ax4, ax5, ax6])

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / f"experiment_{exp['run_id']}_report.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"[✓] Dashboard saved → {out_path}")
    return out_path


def _list_experiments(db_path: Path):
    con = _connect(db_path)
    cur = con.cursor()
    cur.execute(
        "SELECT id, run_id, experiment_name, model_name, status, created_at "
        "FROM experiments ORDER BY id DESC"
    )
    rows = cur.fetchall()
    con.close()
    if not rows:
        print("No experiments found in the database.")
        return
    print(f"\n{'ID':>4}  {'RUN ID':<12}  {'NAME':<25}  {'STATUS':<12}  CREATED AT")
    print("─" * 80)
    for r in rows:
        print(f"{r[0]:>4}  {r[1]:<12}  {r[2]:<25}  {r[4]:<12}  {r[5]}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Synthenexa experiment visualizations from the SQLite database."
    )
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB,
                        help=f"Path to SQLite database (default: {_DEFAULT_DB})")
    parser.add_argument("--experiment-id", type=int, default=None,
                        help="Specific experiment ID to visualise (default: latest completed).")
    parser.add_argument("--list-experiments", action="store_true",
                        help="List all experiments in the database and exit.")
    args = parser.parse_args()

    if args.list_experiments:
        _list_experiments(args.db)
        return

    generate_report(args.db, args.experiment_id)


if __name__ == "__main__":
    main()
