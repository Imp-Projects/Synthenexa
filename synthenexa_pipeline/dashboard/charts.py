"""Plotly chart helpers for the Streamlit analytics dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def score_progression_chart(dataframe: pd.DataFrame):
    if dataframe.empty:
        return None
    return px.line(
        dataframe,
        x="generation",
        y="best_score",
        color="model_name",
        line_group="run_id",
        markers=True,
        title="Best Score by Generation",
    )


def model_comparison_chart(dataframe: pd.DataFrame):
    if dataframe.empty:
        return None
    return px.bar(
        dataframe,
        x="model_name",
        y="peak_score",
        color="experiment_count",
        title="Peak Score by Model",
    )


def base_refined_comparison_chart(dataframe: pd.DataFrame):
    """Plot matched held-out base and refined scores per model/run."""
    required = {"model_name", "base_score", "refined_score"}
    if dataframe.empty or not required.issubset(dataframe.columns):
        return None
    long_frame = dataframe.melt(
        id_vars=[column for column in ("model_name", "run_id") if column in dataframe.columns],
        value_vars=["base_score", "refined_score"],
        var_name="prompt_version",
        value_name="pass_rate",
    )
    long_frame["prompt_version"] = long_frame["prompt_version"].map(
        {"base_score": "Original base prompt", "refined_score": "Refined prompt"}
    )
    return px.bar(
        long_frame,
        x="model_name",
        y="pass_rate",
        color="prompt_version",
        barmode="group",
        facet_col="run_id" if "run_id" in long_frame.columns else None,
        title="Held-Out Base vs Refined Performance",
    )


def evaluation_heatmap_chart(dataframe: pd.DataFrame):
    if dataframe.empty:
        return None
    matrix = dataframe.pivot_table(
        index="prompt_preview",
        columns="model_name",
        values="pass_rate",
        aggfunc="mean",
        fill_value=0,
    )
    return px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="Prompt x Model Pass Rate",
    )
