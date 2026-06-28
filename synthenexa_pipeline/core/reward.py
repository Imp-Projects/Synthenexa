from core.schema import EvalResult


# ------------------------------------------------------------------ #
#  Simple pass@1 metric                                               #
# ------------------------------------------------------------------ #

def pass_at_1(passed: int, total: int) -> float:
    """Returns the fraction of tasks passed — the simplest reward signal."""
    if total == 0:
        return 0.0
    return passed / total


# ------------------------------------------------------------------ #
#  Full reward computation                                            #
# ------------------------------------------------------------------ #

def compute_reward(eval_result: EvalResult) -> float:
    """
    Converts an EvalResult into a scalar RL reward in [0.0, 1.0].

    Reward shaping:
      base     = pass@1  (fraction of tasks passed)
      penalty  += 0.2    if any syntax errors detected
      (future) weighted_difficulty: multiply base by task difficulty weight
      (future) diversity_bonus: reward for producing structurally different solutions
    """
    base = pass_at_1(eval_result.passed, eval_result.total)

    penalty = 0.0

    # Syntax error penalty — deducts 0.2 if any syntax errors occurred
    syntax_errors = getattr(eval_result, "syntax_errors", 0)
    if syntax_errors:
        penalty += 0.2

    # --- future extension hooks (stubbed at 0.0) ---
    # weighted_difficulty_bonus = 0.0   # e.g. harder tasks give more reward
    # diversity_bonus           = 0.0   # e.g. reward unique solution structure
    # error_clustering_penalty  = 0.0   # e.g. penalise systematic failure modes

    return max(0.0, base - penalty)
