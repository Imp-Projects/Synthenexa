select *
from {{ ref('fact_evaluation') }}
where pass_rate < 0 or pass_rate > 1
