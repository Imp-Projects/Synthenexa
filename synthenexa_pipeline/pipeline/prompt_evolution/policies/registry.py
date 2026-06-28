"""
pipeline/prompt_evolution/policies/registry.py — Maps policy names → policy classes.

Usage:
    from pipeline.prompt_evolution.policies.registry import PolicyRegistry
    from core.config import RLConfig

    policy = PolicyRegistry.get_policy(RLConfig(policy_type="epsilon_greedy", epsilon=0.15))
"""
from typing import Dict, Type

from core.config import RLConfig
from pipeline.prompt_evolution.policies.base import Policy
from pipeline.prompt_evolution.policies.greedy import GreedySelectionPolicy
from pipeline.prompt_evolution.policies.epsilon_greedy import EpsilonGreedySelectionPolicy


class PolicyRegistry:
    """
    Dynamic registry mapping string names to selection policy classes.
    New policies can be registered at runtime via PolicyRegistry.register().
    """

    _registry: Dict[str, Type[Policy]] = {
        "greedy":         GreedySelectionPolicy,
        "epsilon_greedy": EpsilonGreedySelectionPolicy,
    }

    @classmethod
    def register(cls, name: str, policy_class: Type[Policy]) -> None:
        """Register a custom policy dynamically."""
        cls._registry[name] = policy_class

    @classmethod
    def get_policy(cls, config: RLConfig) -> Policy:
        """
        Instantiate and return the configured selection policy.

        Args:
            config: RLConfig carrying policy_type and any policy-specific params.

        Returns:
            A fully-constructed Policy instance.

        Raises:
            ValueError: If the policy_type is not registered.
        """
        name = config.policy_type.lower()

        if name not in cls._registry:
            raise ValueError(
                f"Unknown policy type: '{name}'. "
                f"Available policies: {sorted(cls._registry.keys())}"
            )

        policy_class = cls._registry[name]

        if name == "epsilon_greedy":
            #return policy_class(epsilon=config.epsilon)
            # Explicitly cast to the concrete type so the type checker can verify
            # the `epsilon` kwarg — a dict lookup alone can't narrow Type[Policy].
            return EpsilonGreedySelectionPolicy(epsilon=config.epsilon)

        # GreedySelectionPolicy and any other zero-arg policies
        return policy_class()
