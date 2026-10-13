import html
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Synthenexa Results Viewer")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "synthenexa_pipeline", "experiments", "artifacts")
OUTPUT_DIR = os.path.join(ROOT_DIR, "synthenexa_pipeline", "Analysis", "output")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "synthenexa_pipeline", "templates")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@dataclass
class ChartCard:
    category: str
    title: str
    filename: str
    caption: str


def _latest_experiment_dir(root: str) -> Optional[Path]:
    base = Path(root)
    if not base.is_dir():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _discover_generation_numbers(exp_dir: Path) -> list[int]:
    gens = set()
    for path in exp_dir.glob("snapshot_gen_*.json"):
        try:
            gens.add(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(gens)


def _build_summary(exp_dir: Path) -> dict[str, Any]:
    best = _read_json(exp_dir / "best_prompt.json")
    snapshots = [_read_json(path) for path in sorted(exp_dir.glob("snapshot_gen_*.json"))]
    prompt_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for gen in _discover_generation_numbers(exp_dir):
        prompt_rows.extend(_read_jsonl(exp_dir / f"prompts_gen_{gen}.jsonl"))
        eval_rows.extend(_read_jsonl(exp_dir / f"evaluations_gen_{gen}.jsonl"))

    prompt_by_id = {row.get("id"): row for row in prompt_rows if row.get("id")}
    evals_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        pid = row.get("prompt_id")
        if pid:
            evals_by_prompt[pid].append(row)

    best_prompt_id = best.get("prompt_id") or best.get("best_prompt_id")
    reward_history = best.get("reward_history") or {
        str(s.get("generation")): s.get("best_score", 0.0) for s in snapshots if s.get("generation") is not None
    }

    generation_best: list[dict[str, Any]] = []
    generation_candidates: list[dict[str, Any]] = []
    best_prompt_model_scores: dict[str, float] = {}
    model_leaderboard: dict[str, float] = {}
    ensemble_model_scores: dict[str, dict[str, float]] = {}

    for snap in snapshots:
        generation = snap.get("generation")
        scores = snap.get("scores") or {}
        if isinstance(scores, dict) and scores:
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            top_id, top_score = sorted_scores[0]
            generation_best.append({"generation": generation, "prompt_id": top_id, "score": top_score})
            generation_candidates.append(
                {
                    "generation": generation,
                    "count": len(scores),
                    "mean_score": sum(scores.values()) / len(scores),
                    "best_score": top_score,
                }
            )

        if isinstance(snap.get("model_leaderboard"), dict):
            for model, score in snap["model_leaderboard"].items():
                model = str(model)
                model_leaderboard[model] = max(model_leaderboard.get(model, 0.0), float(score))

        if isinstance(snap.get("ensemble_scores"), dict):
            for pid, model_scores in snap["ensemble_scores"].items():
                if isinstance(model_scores, dict):
                    ensemble_model_scores[pid] = {str(m): float(s) for m, s in model_scores.items()}

    if best_prompt_id and best_prompt_id in ensemble_model_scores:
        best_prompt_model_scores = ensemble_model_scores[best_prompt_id]

    prompt_scores = []
    for pid, model_rows in evals_by_prompt.items():
        total = max((row.get("total", 0) or 0) for row in model_rows) or 1
        passed = max((row.get("passed", 0) or 0) for row in model_rows)
        avg_pass_rate = sum((row.get("passed", 0) or 0) / (row.get("total", 1) or 1) for row in model_rows) / len(model_rows)
        prompt_scores.append(
            {
                "prompt_id": pid,
                "generation": prompt_by_id.get(pid, {}).get("generation", 0),
                "score": avg_pass_rate,
                "passed": passed,
                "total": total,
                "is_best": pid == best_prompt_id,
                "spread": (
                    max(ensemble_model_scores.get(pid, {}).values()) - min(ensemble_model_scores.get(pid, {}).values())
                )
                if ensemble_model_scores.get(pid)
                else 0.0,
            }
        )
    prompt_scores.sort(key=lambda item: (item["generation"], item["score"]), reverse=True)

    model_wise = {
        "best_prompt_model_scores": best_prompt_model_scores,
        "model_leaderboard": model_leaderboard,
    }

    ensemble = {
        "prompt_vs_consistency": [
            {
                "prompt_id": pid,
                "mean": sum(scores.values()) / len(scores),
                "worst": min(scores.values()),
                "best": max(scores.values()),
                "is_best": pid == best_prompt_id,
            }
            for pid, scores in ensemble_model_scores.items()
            if scores
        ],
    }

    summary = {
        "run_id": best.get("run_id") or exp_dir.name,
        "experiment_name": exp_dir.name,
        "model": ", ".join(best.get("ensemble_models", [])) if isinstance(best.get("ensemble_models"), list) else best.get("model", "ensemble"),
        "best_score": best.get("score") or best.get("best_score") or 0.0,
        "best_prompt": best.get("text", ""),
        "best_prompt_id": best_prompt_id,
        "reward_history": reward_history,
        "snapshots": snapshots,
        "generation_best": generation_best,
        "generation_candidates": generation_candidates,
        "prompt_scores": prompt_scores[:12],
        "model_wise": model_wise,
        "ensemble": ensemble,
    }
    return summary


def _save_svg(path: str, body: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)


def _horizontal_bar_chart(path: str, labels: list[str], values: list[float], title: str, color: str, xlabel: str = "Score") -> None:
    width, height = 900, 420
    pad_x_left, pad_x_right, pad_y = 200, 50, 55
    chart_w = width - pad_x_left - pad_x_right
    chart_h = height - pad_y * 2
    max_v = max(values) if values else 1.0
    max_v = max(max_v, 1e-9)
    bar_h = chart_h / max(len(values), 1)
    
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    lines.append('<rect width="100%" height="100%" rx="20" fill="#ffffff"/>')
    lines.append(f'<text x="{pad_x_left}" y="28" fill="#0f172a" font-size="20" font-weight="700">{html.escape(title)}</text>')
    lines.append(f'<text x="{pad_x_left + chart_w/2}" y="{height - 15}" fill="#475569" font-size="12" text-anchor="middle">{html.escape(xlabel)}</text>')
    
    for tick in range(0, 5):
        x = pad_x_left + (chart_w / 4) * tick
        score = max_v * (tick / 4)
        lines.append(f'<line x1="{x:.1f}" y1="{pad_y}" x2="{x:.1f}" y2="{pad_y + chart_h}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{pad_y + chart_h + 16}" fill="#475569" font-size="12" text-anchor="middle">{score:.2f}</text>')
        
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = pad_y + idx * bar_h + bar_h * 0.15
        bar_w = (value / max_v) * chart_w
        lines.append(f'<rect x="{pad_x_left:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h * 0.7:.1f}" rx="4" fill="{color}"/>')
        short_label = label if len(label) <= 25 else label[:24] + "…"
        lines.append(f'<text x="{pad_x_left - 10}" y="{y + bar_h * 0.35 + 4:.1f}" fill="#0f172a" font-size="12" text-anchor="end">{html.escape(short_label)}</text>')
        lines.append(f'<text x="{pad_x_left + bar_w + 5:.1f}" y="{y + bar_h * 0.35 + 4:.1f}" fill="#475569" font-size="11" text-anchor="start">{value:.2f}</text>')
        
    lines.append("</svg>")
    _save_svg(path, "\n".join(lines))


def _multi_line_chart(path: str, labels: list[Any], lines_data: dict[str, dict], title: str) -> None:
    width, height = 900, 420
    pad_x, pad_y = 70, 50
    chart_w = width - pad_x * 2
    chart_h = height - pad_y * 2
    
    all_values = []
    for line_info in lines_data.values():
        all_values.extend(line_info["values"])
        
    max_v = max(all_values) if all_values else 1.0
    max_v = max(max_v, 1e-9)
    
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    lines.append('<rect width="100%" height="100%" rx="20" fill="#ffffff"/>')
    lines.append(f'<text x="{pad_x}" y="28" fill="#0f172a" font-size="20" font-weight="700">{html.escape(title)}</text>')
    
    legend_x = pad_x + chart_w - 150
    for i, (name, info) in enumerate(lines_data.items()):
        lines.append(f'<circle cx="{legend_x}" cy="{20 + i*15}" r="4" fill="{info["color"]}"/>')
        lines.append(f'<text x="{legend_x + 10}" y="{24 + i*15}" fill="#0f172a" font-size="11">{html.escape(name)}</text>')

    for tick in range(0, 5):
        y = pad_y + chart_h - (chart_h / 4) * tick
        score = max_v * (tick / 4)
        lines.append(f'<line x1="{pad_x}" y1="{y:.1f}" x2="{pad_x + chart_w}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="45" y="{y + 4:.1f}" fill="#475569" font-size="12" text-anchor="end">{score:.2f}</text>')
        
    step = chart_w / max(len(labels) - 1, 1)
    
    for name, info in lines_data.items():
        points = []
        for idx, value in enumerate(info["values"]):
            x = pad_x + idx * step
            y = pad_y + chart_h - (value / max_v) * chart_h
            points.append((x, y))
            
        if len(points) >= 2:
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            lines.append(f'<polyline fill="none" stroke="{info["color"]}" stroke-width="3" points="{poly}"/>')
            
        for (x, y), value in zip(points, info["values"]):
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{info["color"]}"/>')
            
    for idx, label in enumerate(labels):
        x = pad_x + idx * step
        lines.append(f'<text x="{x:.1f}" y="{height - 16}" fill="#475569" font-size="12" text-anchor="middle">{html.escape(str(label))}</text>')
        
    lines.append("</svg>")
    _save_svg(path, "\n".join(lines))


def _scatter_chart(path: str, points: list[dict[str, Any]], title: str) -> None:
    width, height = 900, 420
    pad_x, pad_y = 70, 55
    chart_w = width - pad_x * 2
    chart_h = height - pad_y * 2
    max_v = max([max(p["mean"], p["worst"], p["best"]) for p in points], default=1.0)
    max_v = max(max_v, 1e-9)
    
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    lines.append('<rect width="100%" height="100%" rx="20" fill="#ffffff"/>')
    lines.append(f'<text x="{pad_x}" y="28" fill="#0f172a" font-size="20" font-weight="700">{html.escape(title)}</text>')
    
    for tick in range(0, 5):
        y = pad_y + chart_h - (chart_h / 4) * tick
        score = max_v * (tick / 4)
        lines.append(f'<line x1="{pad_x}" y1="{y:.1f}" x2="{pad_x + chart_w}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="45" y="{y + 4:.1f}" fill="#475569" font-size="12" text-anchor="end">{score:.2f}</text>')
        x = pad_x + (chart_w / 4) * tick
        lines.append(f'<line x1="{x:.1f}" y1="{pad_y}" x2="{x:.1f}" y2="{pad_y + chart_h}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{pad_y + chart_h + 16}" fill="#475569" font-size="12" text-anchor="middle">{score:.2f}</text>')

    lines.append(f'<line x1="{pad_x}" y1="{pad_y + chart_h}" x2="{pad_x + chart_w}" y2="{pad_y}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="5,5"/>')
        
    best_point = None
    for point in points:
        x = pad_x + (point["mean"] / max_v) * chart_w
        y = pad_y + chart_h - (point["worst"] / max_v) * chart_h
        if point.get("is_best"):
            best_point = (x, y)
            color = "#10b981"
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}" opacity="1.0" stroke="#047857" stroke-width="2"/>')
        else:
            color = "#3b82f6"
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" opacity="0.7"/>')
            
    if best_point:
        bx, by = best_point
        lines.append(f'<text x="{bx + 12:.1f}" y="{by - 12:.1f}" fill="#047857" font-size="12" font-weight="bold">Best Prompt</text>')
        
    lines.append(f'<text x="{width/2}" y="{height - 5}" fill="#475569" font-size="12" text-anchor="middle">X = Mean Ensemble Score, Y = Worst-Model Score (Higher is better, diagonal = perfect consistency)</text>')
    lines.append(f'<text x="15" y="{height/2}" fill="#475569" font-size="12" text-anchor="middle" transform="rotate(-90 15 {height/2})">Worst-Model Score</text>')
    lines.append("</svg>")
    _save_svg(path, "\n".join(lines))


def _grouped_bar_chart(path: str, labels: list[str], series1: list[float], series2: list[float], name1: str, name2: str, title: str) -> None:
    width, height = 900, 420
    pad_x, pad_y = 70, 55
    chart_w = width - pad_x * 2
    chart_h = height - pad_y * 2
    
    max_v = max(max(series1) if series1 else [0], max(series2) if series2 else [0])
    max_v = max(max_v, 1e-9)
    group_w = chart_w / max(len(labels), 1)
    bar_w = group_w * 0.35
    
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    lines.append('<rect width="100%" height="100%" rx="20" fill="#ffffff"/>')
    lines.append(f'<text x="{pad_x}" y="28" fill="#0f172a" font-size="20" font-weight="700">{html.escape(title)}</text>')
    
    legend_x = pad_x + chart_w - 200
    lines.append(f'<rect x="{legend_x}" y="15" width="12" height="12" rx="2" fill="#3b82f6"/>')
    lines.append(f'<text x="{legend_x + 20}" y="25" fill="#0f172a" font-size="11">{html.escape(name1)}</text>')
    lines.append(f'<rect x="{legend_x}" y="35" width="12" height="12" rx="2" fill="#10b981"/>')
    lines.append(f'<text x="{legend_x + 20}" y="45" fill="#0f172a" font-size="11">{html.escape(name2)}</text>')

    for tick in range(0, 5):
        y = pad_y + chart_h - (chart_h / 4) * tick
        score = max_v * (tick / 4)
        lines.append(f'<line x1="{pad_x}" y1="{y:.1f}" x2="{pad_x + chart_w}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="45" y="{y + 4:.1f}" fill="#475569" font-size="12" text-anchor="end">{score:.2f}</text>')
        
    for idx, (label, v1, v2) in enumerate(zip(labels, series1, series2)):
        group_x = pad_x + idx * group_w
        x1 = group_x + group_w * 0.1
        x2 = group_x + group_w * 0.5
        
        h1 = (v1 / max_v) * chart_h
        y1 = pad_y + chart_h - h1
        lines.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{bar_w:.1f}" height="{h1:.1f}" rx="4" fill="#3b82f6"/>')
        lines.append(f'<text x="{x1 + bar_w/2:.1f}" y="{y1 - 5:.1f}" fill="#475569" font-size="10" text-anchor="middle">{v1:.2f}</text>')
        
        h2 = (v2 / max_v) * chart_h
        y2 = pad_y + chart_h - h2
        lines.append(f'<rect x="{x2:.1f}" y="{y2:.1f}" width="{bar_w:.1f}" height="{h2:.1f}" rx="4" fill="#10b981"/>')
        lines.append(f'<text x="{x2 + bar_w/2:.1f}" y="{y2 - 5:.1f}" fill="#475569" font-size="10" text-anchor="middle">{v2:.2f}</text>')
        
        lines.append(f'<text x="{group_x + group_w/2:.1f}" y="{height - 16}" fill="#0f172a" font-size="12" text-anchor="middle">{html.escape(str(label))}</text>')
        
    lines.append("</svg>")
    _save_svg(path, "\n".join(lines))


def _build_charts(summary: dict[str, Any], exp_dir: Path) -> list[ChartCard]:
    cards: list[ChartCard] = []
    out_root = Path(OUTPUT_DIR) / exp_dir.name
    model_dir = out_root / "model_wise"
    prompt_dir = out_root / "prompt_wise"
    ensemble_dir = out_root / "ensemble"
    for d in (model_dir, prompt_dir, ensemble_dir):
        d.mkdir(parents=True, exist_ok=True)

    reward_history = summary.get("reward_history") or {}
    generation_candidates = summary.get("generation_candidates") or []
    
    if reward_history and generation_candidates:
        gens = [int(g) for g in sorted(reward_history.keys(), key=lambda g: int(g))]
        best_scores = [float(reward_history[str(g)]) for g in gens]
        mean_scores = []
        for g in gens:
            m_score = 0.0
            for item in generation_candidates:
                if item["generation"] == g:
                    m_score = float(item["mean_score"])
                    break
            mean_scores.append(m_score)
            
        filename = "prompt_progression.svg"
        lines_data = {
            "Best Score": {"values": best_scores, "color": "#10b981"},
            "Mean Score": {"values": mean_scores, "color": "#3b82f6"}
        }
        _multi_line_chart(str(prompt_dir / filename), gens, lines_data, "Prompt Refinement Progression")
        cards.append(ChartCard("prompt_wise", "Score Progression", f"{exp_dir.name}/prompt_wise/{filename}", "Pipeline improvement over generations."))

    prompt_scores = summary.get("prompt_scores") or []
    if prompt_scores:
        labels = [item["prompt_id"] for item in prompt_scores]
        values = [float(item["score"]) for item in prompt_scores]
        filename = "top_prompt_scores.svg"
        _horizontal_bar_chart(str(prompt_dir / filename), labels, values, "Top Prompt Scores", "#f59e0b")
        cards.append(ChartCard("prompt_wise", "Top Prompts", f"{exp_dir.name}/prompt_wise/{filename}", "Strongest prompts discovered in the run."))

    model_wise = summary.get("model_wise") or {}
    best_scores = model_wise.get("best_prompt_model_scores") or {}
    leaderboard = model_wise.get("model_leaderboard") or {}
    
    if best_scores and leaderboard:
        models = sorted(list(set(best_scores.keys()) | set(leaderboard.keys())))
        labels = [m.replace(":", "\n") for m in models]
        leaderboard_vals = [float(leaderboard.get(m, 0.0)) for m in models]
        winner_vals = [float(best_scores.get(m, 0.0)) for m in models]
        
        filename = "model_wise_comparison.svg"
        _grouped_bar_chart(str(model_dir / filename), labels, leaderboard_vals, winner_vals, "Max Score (Any Prompt)", "Final Winner Score", "Model Leaderboard vs Winner")
        cards.append(ChartCard("model_wise", "Model Leaderboard vs Winner", f"{exp_dir.name}/model_wise/{filename}", "Comparing the absolute best seen on each model vs the final overall winner."))

    ensemble = summary.get("ensemble") or {}
    scatter_points = ensemble.get("prompt_vs_consistency") or []
    if scatter_points:
        filename = "mean_vs_worst.svg"
        _scatter_chart(str(ensemble_dir / filename), scatter_points, "Mean vs Worst-Model Score")
        cards.append(ChartCard("ensemble", "Robustness Scatter", f"{exp_dir.name}/ensemble/{filename}", "Shows whether a prompt is strong on average and on its weakest model."))

    return cards


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    exp_dir = _latest_experiment_dir(ARTIFACTS_DIR)
    if exp_dir is None:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"summary": None, "charts_by_category": {}, "message": "No experiment artifacts found yet."},
        )

    summary = _build_summary(exp_dir)
    cards = _build_charts(summary, exp_dir)
    charts_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        charts_by_category[card.category].append(asdict(card))
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "summary": summary,
            "charts_by_category": dict(charts_by_category),
            "message": None,
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Avoid noisy 404s when browsers request their default favicon."""
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    print("Starting Synthenexa Results Viewer on http://localhost:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
