"""
core/metrics.py — Pure evaluation metrics (NO RL logic).

All functions are stateless and operate on raw lists of values.
"""
from typing import List
import math


def pass_at_k(results: List[bool], k: int) -> float:
    """
    Unbiased pass@k estimator (Chen et al., HumanEval 2021).

    Given n total results with c passing, estimates the probability that
    at least one of k randomly sampled solutions passes.

    Formula: 1 - C(n-c, k) / C(n, k)

    Falls back to simple top-k rate when n < k (not enough samples).
    """
    n = len(results)
    if n == 0 or k <= 0:
        return 0.0

    c = sum(1 for r in results if r)

    if n < k:
        # Not enough samples for full estimator — use observed rate
        return c / n

    # Combinatorial estimator: avoid overflow with log-space calculation
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0

    # 1 - product_{i=0}^{k-1} (n - c - i) / (n - i)
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def success_rate(passed: int, total: int) -> float:
    """Returns overall task success ratio (passed / total)."""
    if total == 0:
        return 0.0
    return passed / total


def avg_completion_length(outputs: List[str]) -> float:
    """Returns the average character length of generated code outputs."""
    if not outputs:
        return 0.0
    return sum(len(o) for o in outputs) / len(outputs)
