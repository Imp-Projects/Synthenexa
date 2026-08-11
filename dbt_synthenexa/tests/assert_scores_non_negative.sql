select *
from {{ ref('fact_generation_summary') }}
where best_score < 0 or avg_score < 0
