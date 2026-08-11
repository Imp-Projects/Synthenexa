select
  id as prompt_key,
  prompt_id,
  experiment_id,
  generation,
  text as prompt_text,
  is_base,
  is_best,
  created_at
from sqlite_source.prompts
