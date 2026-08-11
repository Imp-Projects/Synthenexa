select
  id as experiment_id,
  run_id,
  experiment_name,
  model_name,
  base_prompt_id,
  created_at,
  updated_at,
  status,
  config
from sqlite_source.experiments
