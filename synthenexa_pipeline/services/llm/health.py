"""Provider health tracking for the LLM subsystem."""

class ProviderHealthTracker:
    """Tracks global health status for LLM providers.
    
    Used to quickly fail over and skip providers that have hit hard 
    limits like zero quota across the entire pipeline run.
    """
    
    _disabled_providers: set[str] = set()

    @classmethod
    def disable(cls, provider: str) -> None:
        """Mark a provider as globally disabled due to a hard failure."""
        cls._disabled_providers.add(provider)

    @classmethod
    def is_enabled(cls, provider: str) -> bool:
        """Check if a provider is currently enabled and healthy."""
        return provider not in cls._disabled_providers
