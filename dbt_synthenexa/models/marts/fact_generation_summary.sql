with generation_scores as (
  select
    e.experiment_id,
    e.run_id,
    e.model_name,
    s.generation,
    s.best_score,
    lag(s.best_score) over (
      partition by e.experiment_id
      order by s.generation
    ) as previous_best_score
  from {{ ref('stg_snapshots') }} s
  join {{ ref('stg_experiments') }} e
    on e.experiment_id = s.experiment_id
),
evaluation_rollup as (
  select
    experiment_id,
    generation,
    avg(pass_rate) as avg_score
  from {{ ref('fact_evaluation') }}
  group by 1, 2
)
select
  gs.experiment_id,
  gs.run_id,
  gs.model_name,
  gs.generation,
  gs.best_score,
  coalesce(er.avg_score, 0) as avg_score,
  coalesce(gs.best_score - gs.previous_best_score, 0) as improvement_delta
from generation_scores gs
left join evaluation_rollup er
  on er.experiment_id = gs.experiment_id
 and er.generation = gs.generation
