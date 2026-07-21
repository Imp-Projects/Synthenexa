"""
Analysis/compare_baseline.py — Before vs. After Prompt Refinement Comparison
==============================================================================

This script answers the key question:
  "Did Synthenexa actually improve the prompt? By how much?"

It works in two phases:
  Phase 1 — Evaluate the original base prompt on the same models & dataset
             used in the given experiment (live LLM calls via OllamaAdapter).
  Phase 2 — Generate a focused 4-panel comparison dashboard:
             Panel A: Score delta bar chart (base vs all variants vs winner)
             Panel B: Per-model breakdown (base vs winner on each LLM)
             Panel C: Improvement summary table (numbers at a glance)
             Panel D: Prompt text side-by-side diff

Usage (from synthenexa_pipeline/):
    python Analysis/compare_baseline.py --experiment-id 2 --dataset Dataset/benchmark_mbpp_mini_b.jsonl
    python Analysis/compare_baseline.py --list-experiments

Requirements:
    pip install matplotlib seaborn
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from textwrap import wrap
from typing import Optional

_HERE     = Path(__file__).parent
_PIPELINE = _HERE.parent
_DEFAULT_DB = _PIPELINE / "experiments" / "synthenexa.db"
_OUTPUT_DIR = _HERE / "output"

# ── Import checks ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("[ERROR] Run: pip install matplotlib seaborn")

try:
    import seaborn as sns
except ImportError:
    sys.exit("[ERROR] Run: pip install matplotlib seaborn")

# ── Colour palette ─────────────────────────────────────────────────────────────
P = {
    "bg":        "#1A1A2E",
    "surface":   "#16213E",
    "text":      "#E0E0E0",
    "muted":     "#8E8EA0",
    "base":      "#F7931E",   # orange  = original / base prompt
    "winner":    "#00C9A7",   # teal    = best refined prompt
    "variants":  "#6C63FF",   # purple  = other mutated variants
    "danger":    "#FF6B6B",
    "grid":      "#2D2D4E",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DB helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        sys.exit(f"[ERROR] Database not found: {db_path}")
    return sqlite3.connect(db_path)


def _load_experiment(con, eid: Optional[int]) -> dict:
    cur = con.cursor()
    if eid is not None:
        cur.execute("SELECT id, run_id, experiment_name, model_name, config FROM experiments WHERE id=?", (eid,))
    else:
        cur.execute("SELECT id, run_id, experiment_name, model_name, config "
                    "FROM experiments WHERE status='completed' ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        sys.exit("[ERROR] No completed experiment found.")
    return {"id": row[0], "run_id": row[1], "name": row[2],
            "model_name": row[3], "config": json.loads(row[4]) if row[4] else {}}


def _list_experiments(db_path: Path):
    con = _connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT id, run_id, experiment_name, model_name, status, created_at FROM experiments ORDER BY id DESC")
    rows = cur.fetchall()
    con.close()
    if not rows:
        print("No experiments found.")
        return
    print(f"\n{'ID':>4}  {'RUN ID':<12}  {'NAME':<25}  {'STATUS':<12}  CREATED AT")
    print("─" * 80)
    for r in rows:
        print(f"{r[0]:>4}  {r[1]:<12}  {r[2]:<25}  {r[4]:<12}  {r[5]}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — Live evaluation of the base prompt
# ═══════════════════════════════════════════════════════════════════════════════

def _load_dataset(dataset_path: Path) -> list:
    tasks = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def _build_full_prompt(base_prompt_text: str, task: dict) -> str:
    """Combine the base prompt template with a concrete task description."""
    task_description = task.get("text") or task.get("prompt") or ""
    return f"{base_prompt_text}\n\nTask:\n{task_description}"


def _extract_code(response: str) -> str:
    """Extract the first Python code block from an LLM response."""
    import re
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    # Fallback: try to find def / return lines
    lines = response.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith("def ") or in_code:
            in_code = True
            code_lines.append(line)
    return "\n".join(code_lines).strip() if code_lines else response.strip()


def evaluate_base_prompt(
    base_prompt_text: str,
    model_name: str,
    dataset_path: Path,
    base_url: Optional[str] = None,
) -> dict:
    """
    Run the base prompt through the actual LLM + code evaluator.
    Returns a dict: {model_name: {passed, failed, total, syntax_errors, pass_rate}}
    """
    from services.llm.factory import LLMFactory
    from services.eval_service import EvalService
    from core.schema import PromptCandidate

    tasks = _load_dataset(dataset_path)
    eval_service = EvalService()

    print(f"  [BaseEval] Evaluating base prompt on {len(tasks)} tasks with model '{model_name}'...")
    t0 = time.time()

    # Create LLM adapter
    factory = LLMFactory()
    adapter = factory.create(model_name, base_url=base_url)

    passed_count = 0
    failed_count = 0
    syntax_count = 0
    total        = len(tasks)

    for i, task in enumerate(tasks):
        full_prompt = _build_full_prompt(base_prompt_text, task)
        try:
            response = adapter.generate(full_prompt)
            code     = _extract_code(response)
        except Exception as e:
            print(f"    [WARN] LLM call failed for task {i}: {e}")
            failed_count += 1
            continue

        # Evaluate using the same EvalService the pipeline uses
        result = eval_service.evaluate_single(
            prompt_candidate=PromptCandidate(id=f"base_{i}", text=base_prompt_text),
            task=task,
            generated_code=code,
        )
        if result.get("passed"):
            passed_count += 1
        else:
            if "syntax" in str(result.get("error_type", "")).lower():
                syntax_count += 1
            failed_count += 1

    elapsed = time.time() - t0
    pass_rate = passed_count / total if total > 0 else 0.0
    print(f"  [BaseEval] Done in {elapsed:.1f}s — {passed_count}/{total} passed ({pass_rate:.0%})")
    return {
        "passed":       passed_count,
        "failed":       failed_count,
        "syntax_errors": syntax_count,
        "total":        total,
        "pass_rate":    pass_rate,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Fallback: lightweight offline evaluation (no LLM) for demo / offline mode
# ═══════════════════════════════════════════════════════════════════════════════

def _offline_base_score(eval_rows: list) -> float:
    """
    Estimate what the base prompt score MIGHT be.
    Uses the minimum score across all variants as a conservative lower bound.
    This is used ONLY when --offline flag is set or LLM is unavailable.
    """
    scores = []
    for row in eval_rows:
        pid, _, _, passed, failed, total, _, _ = row
        if total > 0:
            scores.append(passed / total)
    return min(scores) if scores else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def _dark(fig, axes):
    fig.patch.set_facecolor(P["bg"])
    for ax in axes:
        ax.set_facecolor(P["surface"])
        ax.tick_params(colors=P["text"], labelsize=9)
        ax.xaxis.label.set_color(P["text"])
        ax.yaxis.label.set_color(P["text"])
        ax.title.set_color(P["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(P["grid"])


def _short(pid: str) -> str:
    from textwrap import shorten
    label = pid.replace("prompt_0_", "").replace("_cloud", "")
    return shorten(label, width=26, placeholder="…")


def _model_label(raw: str) -> str:
    return raw.replace("_", "-").replace(":cloud", "").strip("-")


def plot_delta_chart(ax, base_scores_per_model: dict, exp_scores: dict,
                     best_pid: str, model_names: list):
    """
    Panel A — Score Delta Chart
    Shows base prompt score vs every mutated variant vs winner.
    The gap between the base bar and the winner bar = improvement.
    """
    # Compute mean base score across all evaluator models
    base_mean = sum(base_scores_per_model[m]["pass_rate"] for m in model_names) / len(model_names)

    # All candidates + base
    labels = ["[BASE]"] + [_short(k) for k in exp_scores]
    values = [base_mean] + list(exp_scores.values())
    raw_keys = [None] + list(exp_scores.keys())

    colors = []
    for rk, v in zip(raw_keys, values):
        if rk is None:
            colors.append(P["base"])
        elif rk == best_pid:
            colors.append(P["winner"])
        else:
            colors.append(P["variants"])

    bars = ax.bar(labels, values, color=colors, edgecolor=P["grid"], linewidth=0.6, width=0.6)

    for bar, val, rk in zip(bars, values, raw_keys):
        face = bar.get_facecolor()
        weight = "bold" if (rk == best_pid or rk is None) else "normal"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f"{val:.2%}", ha="center", va="bottom",
                color=P["text"], fontsize=8.5, fontweight=weight)

    # Draw improvement arrow between base and winner
    winner_score = exp_scores.get(best_pid, base_mean)
    delta = winner_score - base_mean
    winner_idx = list(exp_scores.keys()).index(best_pid) + 1  # +1 for base bar
    ax.annotate(
        f"+{delta:.2%}",
        xy=(winner_idx, winner_score),
        xytext=(0, base_mean),
        arrowprops=dict(arrowstyle="->", color=P["winner"], lw=2),
        color=P["winner"], fontsize=10, fontweight="bold",
        ha="center", va="bottom",
    )

    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Ensemble Pass Rate", fontsize=9)
    ax.set_title(
        "[A]  Base Prompt vs. Refined Variants\n"
        "(Arrow = improvement achieved by Synthenexa)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.axhline(base_mean, color=P["base"], linestyle="--", linewidth=1.0, alpha=0.6)

    legend = [
        mpatches.Patch(color=P["base"],     label=f"Original base prompt ({base_mean:.2%})"),
        mpatches.Patch(color=P["variants"], label="Refined variants"),
        mpatches.Patch(color=P["winner"],   label=f"Best refined prompt ({winner_score:.2%})"),
    ]
    ax.legend(handles=legend, fontsize=8, labelcolor=P["text"],
              facecolor=P["surface"], edgecolor=P["grid"], loc="upper left")


def plot_per_model_comparison(ax, base_scores_per_model: dict, eval_rows: list,
                               best_pid: str, model_names: list):
    """
    Panel B — Per-Model Comparison
    For each evaluator model: base score vs winner score.
    Proves the refined prompt is universally better, not model-specific.
    """
    # Winner scores per model
    winner_per_model = {}
    for row in eval_rows:
        pid, _, _, passed, failed, total, _, model = row
        if pid == best_pid:
            winner_per_model[model] = passed / total if total > 0 else 0.0

    short_models = [_model_label(m) for m in model_names]
    base_vals    = [base_scores_per_model[m]["pass_rate"] for m in model_names]
    winner_vals  = [winner_per_model.get(m, 0.0) for m in model_names]

    x = range(len(short_models))
    w = 0.35
    ax.bar([i - w/2 for i in x], base_vals,   width=w, color=P["base"],   edgecolor=P["grid"],
           lw=0.5, label="Original base prompt")
    ax.bar([i + w/2 for i in x], winner_vals, width=w, color=P["winner"], edgecolor=P["grid"],
           lw=0.5, label="Best refined prompt")

    # Annotate deltas
    for i, (bv, wv) in enumerate(zip(base_vals, winner_vals)):
        delta = wv - bv
        sign  = "+" if delta >= 0 else ""
        col   = P["winner"] if delta >= 0 else P["danger"]
        ax.text(i + w/2, wv + 0.015, f"{sign}{delta:.0%}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=col)

    ax.set_xticks(list(x))
    ax.set_xticklabels(short_models, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Pass Rate", fontsize=9)
    ax.set_title(
        "[B]  Per-Model Performance: Base vs. Winner\n"
        "(Δ labels = how much the winner improved on each model)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=8, labelcolor=P["text"], facecolor=P["surface"], edgecolor=P["grid"])


def plot_summary_table(ax, base_scores_per_model: dict, exp_scores: dict,
                        best_pid: str, base_prompt: str, winner_text: str, model_names: list):
    """
    Panel C — Improvement Summary Table
    Clean numeric table: all the key metrics at a glance.
    """
    ax.axis("off")

    base_mean   = sum(base_scores_per_model[m]["pass_rate"] for m in model_names) / len(model_names)
    winner_score = exp_scores.get(best_pid, 0.0)
    delta        = winner_score - base_mean
    rel_improve  = (delta / base_mean * 100) if base_mean > 0 else 0.0

    # Build table data
    rows = [
        ["Metric",                                "Original Prompt", "Best Refined Prompt"],
        ["Ensemble Score (mean pass rate)",        f"{base_mean:.2%}",   f"{winner_score:.2%}"],
        ["Absolute Score Improvement",             "—",                  f"+{delta:.2%}"],
        ["Relative Improvement",                   "—",                  f"+{rel_improve:.1f}%"],
        ["# Models Evaluated On",                  str(len(model_names)), str(len(model_names))],
    ]
    for m in model_names:
        label = _model_label(m)
        bv = base_scores_per_model[m]["pass_rate"]
        wv = exp_scores.get(best_pid, 0.0)  # placeholder; per-model winner already shown in B
        rows.append([f"Pass Rate — {label}", f"{bv:.2%}", "see Panel B"])

    col_widths = [0.48, 0.26, 0.26]
    header_color = "#2C2C5C"
    row_colors   = [P["surface"], "#1E1E38"]

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(P["grid"])
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color=P["text"], fontweight="bold")
        else:
            cell.set_facecolor(row_colors[r % 2])
            cell.set_text_props(color=P["text"])
            # Highlight improvement cells
            if c == 2 and "+" in cell.get_text().get_text():
                cell.set_text_props(color=P["winner"], fontweight="bold")

    ax.set_title("[C]  Improvement Summary (Numbers at a Glance)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.title.set_color(P["text"])


def plot_prompt_diff(ax, base_text: str, winner_text: str):
    """
    Panel D — Prompt Text Side-by-Side
    Shows the textual difference between original and winning prompt.
    """
    ax.axis("off")

    def _wrapped(text, width=52):
        return "\n".join(wrap(text, width))

    base_w   = _wrapped(base_text)
    winner_w = _wrapped(winner_text)

    # Left side — base
    ax.text(0.01, 0.97, "Original Base Prompt", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=P["base"], va="top")
    ax.text(0.01, 0.88, base_w, transform=ax.transAxes,
            fontsize=8.5, color=P["text"], va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=P["surface"],
                      edgecolor=P["base"], linewidth=1.5))

    # Right side — winner
    ax.text(0.51, 0.97, "Best Refined Prompt  ★", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=P["winner"], va="top")
    ax.text(0.51, 0.88, winner_w, transform=ax.transAxes,
            fontsize=8.5, color=P["text"], va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=P["surface"],
                      edgecolor=P["winner"], linewidth=1.5))

    ax.set_title("[D]  Prompt Text: Before → After",
                 fontsize=11, fontweight="bold", pad=10)
    ax.title.set_color(P["text"])


# ═══════════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_comparison(
    db_path: Path,
    dataset_path: Path,
    experiment_id: Optional[int] = None,
    base_url: Optional[str] = None,
    offline: bool = False,
):
    con = _connect(db_path)
    exp = _load_experiment(con, experiment_id)
    eid = exp["id"]
    cfg = exp["config"]
    model_names: list = cfg.get("ensemble_models", [])

    print(f"\n[Info] Experiment #{eid}: '{exp['name']}' (run={exp['run_id']})")
    print(f"[Info] Models: {model_names}")

    # Load base prompt
    base_prompt_path = _PIPELINE / "Dataset" / "base_prompt.jsonl"
    if base_prompt_path.exists():
        with open(base_prompt_path) as f:
            base_text = json.loads(f.read().strip()).get("prompt", "Write a Python function to solve the given task.")
    else:
        base_text = "Write a Python function to solve the given task."

    # Load snapshot (winner info)
    cur = con.cursor()
    cur.execute("SELECT generation, best_score, best_prompt_id, scores, best_prompt_text "
                "FROM snapshots WHERE experiment_id=? ORDER BY generation DESC LIMIT 1", (eid,))
    snap = cur.fetchone()
    if not snap:
        sys.exit("[ERROR] No snapshot found for this experiment.")

    best_pid   = snap[2]
    winner_text = snap[4] or ""
    exp_scores  = json.loads(snap[3]) if snap[3] else {}

    # Load prompt IDs and eval rows
    cur.execute("SELECT prompt_id FROM prompts WHERE experiment_id=? ORDER BY id", (eid,))
    prompt_ids = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT prompt_id, experiment_id, generation, passed, failed, total, "
                "syntax_errors FROM evaluations WHERE experiment_id=? ORDER BY id", (eid,))
    raw_evals = cur.fetchall()

    def _infer_model(pid, candidates):
        pid_lower = pid.lower()
        for m in candidates:
            slug = m.replace(":", "_").replace("-", "_").replace(".", "_").lower()
            if slug in pid_lower:
                return m
        return candidates[0] if candidates else "unknown"

    eval_rows = [(*r, _infer_model(r[0], model_names)) for r in raw_evals]
    con.close()

    # ── Phase 1: Evaluate base prompt ────────────────────────────────────────
    base_scores_per_model = {}

    if offline:
        print("[WARN] --offline mode: using minimum variant score as base estimate.")
        min_score = _offline_base_score(eval_rows)
        for m in model_names:
            base_scores_per_model[m] = {"pass_rate": min_score, "passed": 0, "failed": 0,
                                         "total": 0, "syntax_errors": 0}
    else:
        for model in model_names:
            print(f"\n[Phase 1] Evaluating BASE prompt on model: {model}")
            result = evaluate_base_prompt(base_text, model, dataset_path, base_url)
            base_scores_per_model[model] = result

    # ── Phase 2: Build dashboard ─────────────────────────────────────────────
    sns.set_theme(style="dark", palette="muted")
    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor(P["bg"])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35,
                           top=0.92, bottom=0.05, left=0.07, right=0.97)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    base_mean    = sum(base_scores_per_model[m]["pass_rate"] for m in model_names) / len(model_names)
    winner_score = exp_scores.get(best_pid, 0.0)
    delta        = winner_score - base_mean
    mode_str     = "(offline estimate)" if offline else "(live LLM evaluation)"

    fig.suptitle(
        f"Synthenexa — Prompt Refinement Impact Analysis  {mode_str}\n"
        f"Experiment: {exp['name']}  |  Run: {exp['run_id']}  |  "
        f"Improvement: {base_mean:.2%} → {winner_score:.2%}  (+{delta:.2%})",
        fontsize=13, fontweight="bold", color=P["text"], y=0.97,
    )

    plot_delta_chart(axA, base_scores_per_model, exp_scores, best_pid, model_names)
    plot_per_model_comparison(axB, base_scores_per_model, eval_rows, best_pid, model_names)
    plot_summary_table(axC, base_scores_per_model, exp_scores, best_pid, base_text, winner_text, model_names)
    plot_prompt_diff(axD, base_text, winner_text)

    _dark(fig, [axA, axB, axC, axD])

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / f"comparison_{exp['run_id']}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    print(f"\n[✓] Comparison dashboard saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate a before-vs-after prompt refinement comparison from Synthenexa results."
    )
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--experiment-id", type=int, default=None,
                        help="Experiment ID to compare (default: latest completed).")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Dataset JSONL used during the experiment (for live base evaluation).")
    parser.add_argument("--ollama-base-url", type=str, default=None,
                        help="Ollama base URL override.")
    parser.add_argument("--offline", action="store_true",
                        help="Skip live LLM evaluation; use minimum variant score as a conservative baseline estimate.")
    parser.add_argument("--list-experiments", action="store_true",
                        help="List all stored experiments and exit.")
    args = parser.parse_args()

    if args.list_experiments:
        _list_experiments(args.db)
        return

    # Resolve dataset path
    if args.offline:
        dataset_path = None
    else:
        if args.dataset is None:
            # Default to the mini dataset
            dataset_path = _PIPELINE / "Dataset" / "benchmark_mbpp_mini_b.jsonl"
            if not dataset_path.exists():
                dataset_path = _PIPELINE / "Dataset" / "benchmark_mbpp_small.jsonl"
        else:
            dataset_path = args.dataset
        if not dataset_path.exists():
            sys.exit(f"[ERROR] Dataset not found: {dataset_path}\n"
                     "Use --offline to skip live evaluation.")
        print(f"[Info] Dataset: {dataset_path} ({sum(1 for _ in open(dataset_path))} tasks)")

    generate_comparison(
        db_path=args.db,
        dataset_path=dataset_path,
        experiment_id=args.experiment_id,
        base_url=args.ollama_base_url,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
