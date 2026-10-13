# Prompt Refinement Improvement Plan

## Purpose

This plan improves the existing Synthenexa prototype without replacing its architecture. The target is a more reliable and cost-efficient demonstration that automatically refines prompts across Ollama-hosted models.

The plan is intentionally constrained to:

- Ollama models only.
- A maximum of three generations.
- Three to five mutations per generation.
- The existing `PipelineContext`, stage DAG, artifact storage, and analysis flow.
- A college-level prototype budget of approximately $60.

## Current baseline

The current workflow is:

```text
run_optimization.py
    → mutation
    → cross-model execution
    → local evaluation
    → ensemble scoring
    → selection
    → artifact reports
```

The latest three-model experiment demonstrated a useful improvement, but the optimization and held-out results differ. The next implementation must prevent overfitting to the selection tasks and make repeated model calls cheaper.

## Phase 1 — Documentation and workflow cleanup

Status: completed

Tasks:

- Reduce `commands.md` to the Ollama-only workflow.
- Document the main ensemble run.
- Document structured analysis and final report generation.
- Retain the four-panel impact dashboard workflow.
- Explain the purpose of `greedy`, `epsilon`, `generations`, `mutations`, and model selection.

Acceptance criteria:

- No OpenAI or Gemini commands appear in the primary workflow.
- A new user can run an experiment and generate reports using the documented commands.

## Phase 2 — Mutation quality

Status: completed

Existing improvements:

- Mutation roles for clarity, correctness, output discipline, reasoning, and concise wording.
- Compact parent-score feedback for later generations.
- Prompt normalization and hashing.
- Duplicate suppression within and across model mutation pools.
- Candidate metadata for role and prompt hash.

Next refinements:

- Include validation failures, not only aggregate parent errors.
- Add a maximum prompt length to avoid verbose mutations.
- Preserve the parent prompt as an explicit candidate when mutations are weak.

Acceptance criteria:

- Candidate roles are visible in artifacts.
- Duplicate candidates are not cross-evaluated twice.
- Mutation requests remain within the configured 3–5 limit.

## Phase 3 — Cost control and caching

Status: completed

Existing improvements:

- In-memory code-generation cache.
- Generation call, cache-hit, and failure counters.
- Complete ensemble matrix validation.

Implementation details:

1. Add persistent cache storage under an experiment-independent cache directory.
2. Cache mutation outputs using model, parent hash, role, generation, and mutation settings.
3. Cache generated code using model, prompt hash, task ID, and generation parameters.
4. Record cache hits and misses in manifests.
5. Add a pre-run estimated-call report.
6. Refuse or warn when the estimated run exceeds a configured budget.

Implemented in Phase 3:

- Persistent JSON response cache under `synthenexa_pipeline/experiments/cache/`.
- Mutation-output caching keyed by model, parent prompt, mutation role, feedback, and temperature.
- Code-generation caching keyed by model, prompt, task, task prompt, and temperature.
- Persistent cache hit/miss statistics in run metadata.
- `--max-llm-calls` pre-run worst-case call estimate and guardrail.

Acceptance criteria:

- Repeating an identical experiment does not repeat identical Ollama calls.
- Cache keys include all parameters that can affect output.
- Cache use is visible in the run manifest.

## Phase 4 — Validation-aware selection

Objective: reduce overfitting to the optimization tasks.

Tasks:

1. Preserve separate task groups:
   - Search tasks.
   - Validation tasks.
   - Final held-out test tasks.
2. Evaluate generated code on search and validation subsets locally.
3. Compute a combined selection score:

```text
selection_score = 0.7 × search_score + 0.3 × validation_score
```

4. Keep final test tasks untouched until the workflow finishes.
5. Report search, validation, and held-out scores separately.

Implementation scope:

- Extend scoring/context data rather than redesigning the DAG.
- Reuse generated code where task outputs are already available.
- Do not add model calls solely for validation.

Acceptance criteria:

- Candidate selection uses validation information.
- Final test results are generated only after the winner is frozen.
- Artifacts clearly identify each task split.

Status: implemented. Evaluation now preserves per-task outcomes, derives
search/validation rewards from the existing generated code, and applies the
0.7/0.3 selection weighting without additional Ollama calls. Held-out test
evaluation remains a final workflow step after selection.

## Phase 5 — Robust ensemble scoring

Objective: favor prompts that work across all models, not only the strongest model.

Tasks:

1. Preserve the existing per-model score matrix.
2. Add a robust aggregate score:

```text
robust_score = 0.7 × mean_model_score + 0.3 × minimum_model_score
```

3. Continue reporting mean, minimum, maximum, and model spread.
4. Use deterministic tie-breaking.
5. Keep `mean` available for comparison, but use robust scoring for the primary final experiment.

Acceptance criteria:

- A prompt cannot receive a complete score when a model result is missing.
- The selected candidate’s weakest-model score is visible.
- Offline analysis can recompute alternative aggregates without new Ollama calls.

Status: implemented. The robust strategy is available through
`--ensemble-strategy robust`, score matrices remain per-model, and each
candidate now records mean, minimum, maximum, and spread diagnostics in the
generation snapshots. Missing-model protection remains enforced by the
ensemble execution stage before scoring.

## Phase 6 — Elite retention and parent protection

Objective: prevent later generations from discarding a strong prompt.

Tasks:

- Preserve the current parent.
- Preserve the best aggregate candidate.
- Preserve the best worst-model candidate.
- Select the next parent from this protected pool and the new mutations.
- Prefer the existing parent on exact score ties.

Acceptance criteria:

- A later generation cannot overwrite the global best with a weaker candidate.
- The final report distinguishes global best, selected parent, and final refined prompt.

Status: implemented. Selection now protects the current parent and global
best prompt, and includes the strongest weakest-model candidate when ensemble
diagnostics are available. Exact score ties prefer the existing parent.

## Phase 7 — Reporting improvements

Tasks:

1. Keep `generate_analysis.py` for the structured analysis package.
2. Keep `generate_final_report.py` for compact charts and Markdown tables.
3. Add an artifact-native wrapper for the four-panel dashboard currently produced through `compare_baseline.py`.
4. Ensure the four-panel report uses strict held-out results when available.
5. Label small test sets prominently in chart subtitles and captions.

Preferred reporting flow:

```text
generate_analysis.py
    → generate_final_report.py
    → generate_impact_dashboard.py
```

Acceptance criteria:

- No temporary SQLite import is required for file-backed experiments.
- Reports never perform live model calls.
- Baseline and refined prompts use the same held-out task IDs.

Status: implemented. The artifact-native dashboard reads `summary.json`, uses
the strict held-out comparison, reports the held-out task count, and performs
no SQLite import or live model call.

## Phase 8 — Controlled experiment strategy

Status: execution protocol implemented. The two-model tuning and three-model
confirmation commands, call guards, cache checks, and evidence criteria are
documented in `docs/phase8_experiment_protocol.md`. The actual Ollama runs
remain intentionally pending user execution and budget availability.

### Tuning run

Use two Ollama models first:

```text
nemotron-3-ultra:cloud
gpt-oss:120b-cloud
```

Recommended settings:

```text
generations: 3
mutations: 3
policy: greedy
aggregation: robust score
```

### Confirmation run

If the tuning run improves validation without model regression, evaluate the selected approach with:

```text
nemotron-3-ultra:cloud
gpt-oss:120b-cloud
neeoggl/glm-5.2:latest
```

Use three generations and three or four mutations only after caching and call accounting are verified.

## Evaluation criteria

Every report must include:

- Baseline search score.
- Refined search score.
- Baseline validation score.
- Refined validation score.
- Baseline held-out score.
- Refined held-out score.
- Per-model deltas.
- Mean improvement.
- Worst-model improvement.
- Number of tasks in each split.
- Number of Ollama calls and cache hits.

A strong result should improve the aggregate held-out score without materially regressing any model. A 30% improvement is only credible if it is visible across more than one task partition or repeated run.

## Budget and efficiency rules

- Do not run paid-provider models.
- Do not exceed three generations or five mutations.
- Do not rerun identical model/prompt/task combinations.
- Prefer offline score recomputation over new model calls.
- Run two-model tuning before three-model confirmation.
- Keep local evaluation and reporting free of Ollama calls.

## Completion definition

The improvement effort is complete when:

- The pipeline uses persistent caching.
- Candidate selection includes validation and robust model scoring.
- Strong candidates are retained across generations.
- Reports use strict baseline-versus-refined held-out comparisons.
- A controlled Ollama experiment demonstrates whether the revised approach improves consistently.
