"""Streamlit application for browsing Synthenexa analytics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.charts import (
    evaluation_heatmap_chart,
    model_comparison_chart,
    score_progression_chart,
)
from warehouse import queries
from warehouse.duckdb_client import DuckDBWarehouseClient


st.set_page_config(page_title="Synthenexa Analytics", layout="wide")


@st.cache_resource
def get_warehouse() -> DuckDBWarehouseClient:
    return DuckDBWarehouseClient()


@st.cache_data(ttl=60)
def load_dataframe(sql: str) -> pd.DataFrame:
    return get_warehouse().to_df(sql)


def safe_load_dataframe(sql: str) -> pd.DataFrame:
    try:
        return load_dataframe(sql)
    except Exception as exc:
        st.warning(f"Query unavailable yet: {exc}")
        return pd.DataFrame()


def main() -> None:
    st.title("Synthenexa Analytics Dashboard")
    warehouse = get_warehouse()
    st.caption(f"SQLite source: `{warehouse.metadata()['sqlite_path']}`")

    experiments_df = safe_load_dataframe(
        """
        SELECT run_id, experiment_name, model_name, status, created_at
        FROM sqlite_source.experiments
        ORDER BY created_at DESC
        """
    )
    progression_df = safe_load_dataframe(queries.BEST_SCORE_TREND_SQL)
    comparison_df = safe_load_dataframe(queries.MODEL_COMPARISON_SQL)
    heatmap_df = safe_load_dataframe(queries.PROMPT_MODEL_HEATMAP_SQL)
    health_df = safe_load_dataframe(queries.SCORE_IMPROVEMENT_DELTA_SQL)

    st.subheader("Experiment Browser")
    if experiments_df.empty:
        st.info("No experiment records found in SQLite yet.")
    else:
        experiment_names = experiments_df["run_id"].tolist()
        selected_run = st.selectbox("Select a run", options=experiment_names)
        selected = experiments_df.loc[experiments_df["run_id"] == selected_run].head(1)
        st.dataframe(selected, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Score Progression")
        progression_chart = score_progression_chart(progression_df)
        if progression_chart is None:
            st.info("No snapshot data available yet.")
        else:
            st.plotly_chart(progression_chart, use_container_width=True)
    with right:
        st.subheader("Model Comparison")
        comparison_chart = model_comparison_chart(comparison_df)
        if comparison_chart is None:
            st.info("No model comparison data available yet.")
        else:
            st.plotly_chart(comparison_chart, use_container_width=True)

    st.subheader("Prompt Explorer")
    prompts_df = safe_load_dataframe(
        """
        SELECT prompt_id, generation, is_best, substr(text, 1, 160) AS prompt_preview, created_at
        FROM sqlite_source.prompts
        ORDER BY created_at DESC
        """
    )
    st.dataframe(prompts_df, use_container_width=True, hide_index=True)

    st.subheader("Evaluation Heatmap")
    heatmap_chart = evaluation_heatmap_chart(heatmap_df)
    if heatmap_chart is None:
        st.info("No evaluation data available yet.")
    else:
        st.plotly_chart(heatmap_chart, use_container_width=True)

    st.subheader("Pipeline Health")
    if health_df.empty:
        st.info("No pipeline health metrics available yet.")
    else:
        latest_health = (
            health_df.sort_values(["run_id", "generation"], ascending=[False, False])
            .groupby("run_id", as_index=False)
            .head(1)
        )
        st.dataframe(latest_health, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
