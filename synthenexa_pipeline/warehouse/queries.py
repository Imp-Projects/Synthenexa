"""Reusable analytical SQL for the Synthenexa DuckDB warehouse."""

BEST_SCORE_TREND_SQL = """
SELECT
    e.run_id,
    e.experiment_name,
    e.model_name,
    s.generation,
    s.best_score,
    s.created_at
FROM sqlite_source.snapshots AS s
JOIN sqlite_source.experiments AS e
    ON e.id = s.experiment_id
ORDER BY e.created_at DESC, s.generation ASC
"""

MODEL_COMPARISON_SQL = """
SELECT
    e.model_name,
    COUNT(DISTINCT e.id) AS experiment_count,
    MAX(s.best_score) AS peak_score,
    AVG(s.best_score) AS avg_generation_best_score
FROM sqlite_source.experiments AS e
JOIN sqlite_source.snapshots AS s
    ON s.experiment_id = e.id
GROUP BY e.model_name
ORDER BY peak_score DESC NULLS LAST, experiment_count DESC
"""

PASS_AT_ONE_BY_GENERATION_SQL = """
SELECT
    e.run_id,
    e.model_name,
    ev.generation,
    AVG(CASE WHEN ev.total > 0 THEN CAST(ev.passed AS DOUBLE) / ev.total ELSE 0 END) AS pass_at_1
FROM sqlite_source.evaluations AS ev
JOIN sqlite_source.experiments AS e
    ON e.id = ev.experiment_id
GROUP BY e.run_id, e.model_name, ev.generation
ORDER BY e.run_id DESC, ev.generation ASC
"""

SCORE_IMPROVEMENT_DELTA_SQL = """
WITH ordered_scores AS (
    SELECT
        e.run_id,
        e.model_name,
        s.generation,
        s.best_score,
        LAG(s.best_score) OVER (
            PARTITION BY e.id
            ORDER BY s.generation
        ) AS previous_best_score
    FROM sqlite_source.snapshots AS s
    JOIN sqlite_source.experiments AS e
        ON e.id = s.experiment_id
)
SELECT
    run_id,
    model_name,
    generation,
    best_score,
    COALESCE(best_score - previous_best_score, 0) AS improvement_delta
FROM ordered_scores
ORDER BY run_id DESC, generation ASC
"""

PROMPT_MODEL_HEATMAP_SQL = """
SELECT
    p.prompt_id,
    substr(p.text, 1, 120) AS prompt_preview,
    COALESCE(ev.evaluator_model, e.model_name) AS model_name,
    AVG(CASE WHEN ev.total > 0 THEN CAST(ev.passed AS DOUBLE) / ev.total ELSE 0 END) AS pass_rate
FROM sqlite_source.prompts AS p
JOIN sqlite_source.experiments AS e
    ON e.id = p.experiment_id
LEFT JOIN sqlite_source.evaluations AS ev
    ON ev.experiment_id = p.experiment_id
   AND ev.prompt_id = p.prompt_id
   AND ev.generation = p.generation
GROUP BY 1, 2, 3
ORDER BY pass_rate DESC NULLS LAST, prompt_id
"""
