select
  id as evaluation_id,
  prompt_id,
  experiment_id,
  generation,
  coalesce(evaluator_model, '') as evaluator_model,
  passed,
  failed,
  total,
  syntax_errors,
  errors,
  total_time_ms,
  min_time_ms,
  max_time_ms,
  avg_time_ms,
  created_at
from sqlite_source.evaluations
