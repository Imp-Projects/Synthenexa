"""Airflow DAG wrapping the existing Synthenexa CLI and dbt transforms."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "synthenexa",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="synthenexa_pipeline",
    default_args=default_args,
    description="Run Synthenexa optimization and dbt analytics transforms.",
    schedule="@daily",
    start_date=datetime(2026, 8, 1),
    catchup=False,
    max_active_runs=1,
    tags=["synthenexa", "data-engineering"],
) as dag:
    run_pipeline = BashOperator(
        task_id="run_pipeline",
        cwd="/opt/synthenexa",
        bash_command="python synthenexa_pipeline/scripts/run_optimization.py --generations 1",
        env={"PYTHONPATH": "/opt/synthenexa/synthenexa_pipeline"},
    )

    validate_output = BashOperator(
        task_id="validate_output",
        cwd="/opt/synthenexa",
        bash_command="python -c \"from warehouse.duckdb_client import DuckDBWarehouseClient; print(DuckDBWarehouseClient().metadata())\"",
        env={"PYTHONPATH": "/opt/synthenexa/synthenexa_pipeline"},
    )

    run_dbt_transform = BashOperator(
        task_id="run_dbt_transform",
        cwd="/opt/synthenexa/dbt_synthenexa",
        bash_command="dbt run --profiles-dir . && dbt test --profiles-dir .",
    )

    notify_complete = BashOperator(
        task_id="notify_complete",
        cwd="/opt/synthenexa",
        bash_command="echo 'Synthenexa pipeline and dbt transform completed successfully.'",
    )

    run_pipeline >> validate_output >> run_dbt_transform >> notify_complete
