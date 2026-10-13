from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.schema import EvalResult, ExecutionResult, PromptCandidate


@dataclass
class PipelineContext:
    """
    Runtime state container for one full RL optimization run.
    One episode = one evolving prompt system state across N generations.
    """

    # --- run metadata ---
    model_name: str = "nemotron-3-super:cloud"
    run_id: str = ""
    dataset_split: str = "all"
    seed: Optional[int] = None
    partition_metadata: Dict[str, Any] = field(default_factory=dict)

    # --- lifecycle ---
    generation: int = 1
    max_generations: int = 1

    # --- prompt state ---
    base_prompt: Optional[PromptCandidate] = None
    current_prompt: Optional[PromptCandidate] = None
    candidate_prompts: List[PromptCandidate] = field(default_factory=list)

    # --- execution state ---
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    test_tasks: List[Dict[str, Any]] = field(default_factory=list)
    generated_codes: Dict[str, List[ExecutionResult]] = field(default_factory=dict)
    # Cross-generation cache for deterministic code-generation requests.
    execution_cache: Dict[str, str] = field(default_factory=dict)
    call_stats: Dict[str, int] = field(default_factory=lambda: {
        "mutation_calls": 0,
        "generation_calls": 0,
        "generation_cache_hits": 0,
        "generation_failures": 0,
        "persistent_cache_hits": 0,
        "persistent_cache_misses": 0,
    })

    # --- evaluation state ---
    eval_results: Dict[str, EvalResult] = field(default_factory=dict)
    eval_results_by_model: Dict[str, Dict[str, EvalResult]] = field(default_factory=dict)
    baseline_eval_results_by_model: Dict[str, Dict[str, EvalResult]] = field(default_factory=dict)
    final_test_eval_results_by_model: Dict[str, Dict[str, EvalResult]] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    # ensemble_scores: {prompt_id: {model_name: score}} — populated by EnsembleScoringStage
    ensemble_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Per-task sandbox outcomes and split-aware ensemble rewards. These are
    # derived from existing generated code, so they add no model calls.
    task_outcomes: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    ensemble_split_scores: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    # Per-candidate aggregate diagnostics used by reports and offline analysis.
    ensemble_score_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # --- RL state ---
    best_prompt: Optional[PromptCandidate] = None
    global_best_prompt: Optional[PromptCandidate] = None
    selected_next_prompt: Optional[PromptCandidate] = None
    best_score: float = 0.0
    best_ensemble_scores: Dict[str, float] = field(default_factory=dict)
    model_leaderboard: Dict[str, float] = field(default_factory=dict)
    reward_history: Dict[int, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Lifecycle helpers                                                   #
    # ------------------------------------------------------------------ #

    def update_best(self, prompt: PromptCandidate, score: float) -> None:
        """Update the all-time best prompt only if the new score exceeds it."""
        if score > self.best_score:
            self.best_score = score
            self.best_prompt = prompt
            self.global_best_prompt = prompt

    def add_candidate(self, prompt: PromptCandidate) -> None:
        """Append a new candidate prompt to the current generation pool."""
        self.candidate_prompts.append(prompt)

    def reset_for_next_generation(self) -> None:
        """
        Clear per-generation state so the next generation starts clean.
        Preserves: global_best_prompt, best_prompt, best_score, reward_history,
        tasks, base_prompt, current_prompt, and baseline_eval_results_by_model.
        """
        self.candidate_prompts = []
        self.generated_codes = {}
        self.eval_results = {}
        self.eval_results_by_model = {}
        self.scores = {}
        self.ensemble_scores = {}
        self.task_outcomes = {}
        self.ensemble_split_scores = {}
        self.ensemble_score_summary = {}
        self.selected_next_prompt = None
        # current_prompt is intentionally kept — it seeds the next mutation round.
        # Preserve the baseline and the all-time winner across generations.
