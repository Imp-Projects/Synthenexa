select
  e.experiment_id,
  e.run_id,
  ev.generation,
  ev.prompt_id,
  coalesce(nullif(ev.evaluator_model, ''), e.model_name) as model_name,
  ev.passed,
  ev.failed,
  ev.total,
  ev.syntax_errors,
  case when ev.total > 0 then cast(ev.passed as double) / ev.total else 0 end as pass_rate,
  ev.total_time_ms,
  ev.avg_time_ms,
  ev.created_at
from {{ ref('stg_evaluations') }} ev
join {{ ref('stg_experiments') }} e
  on e.experiment_id = ev.experiment_id
