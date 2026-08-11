select
  id as snapshot_id,
  experiment_id,
  generation,
  best_score,
  best_prompt_id,
  best_prompt_text,
  scores,
  reward_history,
  best_ensemble_scores,
  model_leaderboard,
  ensemble_scores,
  created_at
from sqlite_source.snapshots
