"""Prometheus-backed metrics for Synthenexa pipeline runs.

The helpers in this module are intentionally lightweight so the existing CLI
can import them without changing its calling convention.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator, Optional

try:
    from prometheus_client import Counter, Gauge, Histogram
except Exception:  # pragma: no cover - graceful fallback when dependency is absent
    Counter = Gauge = Histogram = None


class _NoopMetric:
    def labels(self, **_: str) -> "_NoopMetric":
        return self

    def inc(self, amount: float = 1.0) -> None:
        return None

    def set(self, value: float) -> None:
        return None

    def observe(self, value: float) -> None:
        return None


class PipelineMetrics:
    """Wraps the small set of Prometheus metrics used by the pipeline."""

    def __init__(self) -> None:
        metric_factory = _NoopMetric if Histogram is None else None
        if metric_factory:
            self.generation_duration = metric_factory()
            self.stage_duration = metric_factory()
            self.prompt_score = metric_factory()
            self.pipeline_runs = metric_factory()
            return

        self.generation_duration = Histogram(
            "synthenexa_generation_duration_seconds",
            "Wall-clock duration for a single generation execution.",
            buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
        )
        self.stage_duration = Histogram(
            "synthenexa_stage_duration_seconds",
            "Wall-clock duration per stage in the prompt evolution DAG.",
            labelnames=("stage",),
            buckets=(0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60),
        )
        self.prompt_score = Gauge(
            "synthenexa_prompt_score",
            "Latest best prompt score recorded by model.",
            labelnames=("model",),
        )
        self.pipeline_runs = Counter(
            "synthenexa_pipeline_runs_total",
            "Total number of pipeline runs by terminal status.",
            labelnames=("status",),
        )

    @contextmanager
    def track_generation(self) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.generation_duration.observe(perf_counter() - start)

    @contextmanager
    def track_stage(self, stage_name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.stage_duration.labels(stage=stage_name).observe(perf_counter() - start)

    def record_best_score(self, model: Optional[str], score: Optional[float]) -> None:
        if score is None:
            return
        self.prompt_score.labels(model=model or "unknown").set(float(score))

    def record_run(self, status: str) -> None:
        self.pipeline_runs.labels(status=status).inc()


_PIPELINE_METRICS: Optional[PipelineMetrics] = None


def get_pipeline_metrics() -> PipelineMetrics:
    global _PIPELINE_METRICS
    if _PIPELINE_METRICS is None:
        _PIPELINE_METRICS = PipelineMetrics()
    return _PIPELINE_METRICS
