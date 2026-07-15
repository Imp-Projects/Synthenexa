from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from core.schema import PromptCandidate, EvalResult, ExecutionResult


@dataclass
class PipelineContext:
    """
    Runtime state container for one full RL optimization run.
    One episode = one evolving prompt system state across N generations.
    """

    # --- run metadata ---
    model_name: str = "nemotron-3-super:cloud"
    run_id: str = ""

    # --- lifecycle ---
    generation: int = 1
    max_generations: int = 1

    # --- prompt state ---
    base_prompt: Optional[PromptCandidate] = None
    current_prompt: Optional[PromptCandidate] = None
    candidate_prompts: List[PromptCandidate] = field(default_factory=list)

    # --- execution state ---
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    generated_codes: Dict[str, List[ExecutionResult]] = field(default_factory=dict)

    # --- evaluation state ---
    eval_results: Dict[str, EvalResult] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    # ensemble_scores: {prompt_id: {model_name: score}} — populated by EnsembleScoringStage
    ensemble_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # --- RL state ---
    best_prompt: Optional[PromptCandidate] = None
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

    def add_candidate(self, prompt: PromptCandidate) -> None:
        """Append a new candidate prompt to the current generation pool."""
        self.candidate_prompts.append(prompt)

    def reset_for_next_generation(self) -> None:
        """
        Clear per-generation state so the next generation starts clean.
        Preserves: best_prompt, best_score, reward_history, tasks, base_prompt.
        """
        self.candidate_prompts = []
        self.generated_codes = {}
        self.eval_results = {}
        self.scores = {}
        self.ensemble_scores = {}
        # current_prompt is intentionally kept — it seeds the next mutation round
