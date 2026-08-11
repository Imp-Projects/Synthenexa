select distinct
  p.prompt_id,
  p.prompt_text,
  p.created_at as first_seen_at,
  e.model_name as origin_model
from {{ ref('stg_prompts') }} p
join {{ ref('stg_experiments') }} e
  on e.experiment_id = p.experiment_id
