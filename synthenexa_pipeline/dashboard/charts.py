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
