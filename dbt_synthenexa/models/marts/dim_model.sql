select distinct
  model_name,
  case
    when model_name like 'openai:%' then 'tier_1_openai'
    when model_name like 'gemini:%' then 'tier_2_gemini'
    when model_name like 'nemotron%' then 'tier_3_nemotron'
    else 'tier_4_other'
  end as model_tier
from {{ ref('stg_experiments') }}
