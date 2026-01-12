"""Tests for agent state types."""

from typing import get_type_hints

from sintra.agents.state import SintraState
from sintra.profiles.models import (
    ExperimentResult,
    HardwareProfile,
    LLMConfig,
    ModelRecipe,
)


class TestSintraState:
    """Tests for SintraState TypedDict."""

    def test_state_has_required_keys(self) -> None:
        """Test that SintraState defines all required keys."""
        hints = get_type_hints(SintraState)

        required_keys = [
            "profile",
            "llm_config",
            "current_recipe",
            "history",
            "critic_feedback",
            "best_recipe",
            "iteration",
            "is_converged",
            "use_debug",
        ]

        for key in required_keys:
            assert key in hints, f"Missing required key: {key}"

    def test_profile_type(self) -> None:
        """Test profile field type."""
        hints = get_type_hints(SintraState)
        assert hints["profile"] == HardwareProfile

    def test_llm_config_type(self) -> None:
        """Test llm_config field type."""
        hints = get_type_hints(SintraState)
        assert hints["llm_config"] == LLMConfig
