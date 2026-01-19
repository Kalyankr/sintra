"""Tests for agent node functions."""

import pytest

from sintra.agents.nodes import (
    DEFAULT_OUTPUT_FILE,
    MAX_ITERATIONS,
    architect_node,
    benchmarker_node,
    critic_node,
    critic_router,
)
from sintra.agents.state import SintraState
from sintra.profiles.models import (
    Constraints,
    ExperimentResult,
    HardwareProfile,
    LLMConfig,
    LLMProvider,
    ModelRecipe,
    Targets,
)


@pytest.fixture
def base_profile() -> HardwareProfile:
    """Create a test hardware profile."""
    return HardwareProfile(
        name="Test Device",
        constraints=Constraints(vram_gb=8.0),
        targets=Targets(min_tokens_per_second=50.0, min_accuracy_score=0.7),
    )


@pytest.fixture
def base_state(base_profile: HardwareProfile) -> SintraState:
    """Create a base state for testing."""
    return SintraState(
        profile=base_profile,
        llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
        current_recipe=None,
        history=[],
        critic_feedback="",
        best_recipe=None,
        iteration=0,
        is_converged=False,
        use_debug=True,  # Use debug mode to skip LLM calls
    )


class TestArchitectNode:
    """Tests for architect_node function."""

    def test_debug_mode_returns_fixed_recipe(self, base_state: SintraState) -> None:
        """Test that debug mode returns a predetermined recipe."""
        result = architect_node(base_state)

        assert "current_recipe" in result
        assert isinstance(result["current_recipe"], ModelRecipe)
        assert result["current_recipe"].bits == 2
        assert result["iteration"] == 1

    def test_increments_iteration(self, base_state: SintraState) -> None:
        """Test that iteration counter is incremented."""
        base_state["iteration"] = 5
        result = architect_node(base_state)

        assert result["iteration"] == 6

    def test_sets_converged_in_debug(self, base_state: SintraState) -> None:
        """Test that debug mode sets is_converged."""
        result = architect_node(base_state)

        assert result["is_converged"] is True


class TestBenchmarkerNode:
    """Tests for benchmarker_node function."""

    def test_returns_history_entry(self, base_state: SintraState) -> None:
        """Test that benchmarker returns a history entry."""
        base_state["current_recipe"] = ModelRecipe(bits=4, pruning_ratio=0.1)

        result = benchmarker_node(base_state)

        assert "history" in result
        assert len(result["history"]) == 1
        assert "recipe" in result["history"][0]
        assert "metrics" in result["history"][0]

    def test_uses_mock_executor_in_debug(self, base_state: SintraState) -> None:
        """Test that debug mode uses MockExecutor."""
        base_state["current_recipe"] = ModelRecipe()

        result = benchmarker_node(base_state)

        # MockExecutor always succeeds
        assert result["history"][0]["metrics"].was_successful is True


class TestCriticNode:
    """Tests for critic_node function."""

    def test_initial_attempt_feedback(self, base_state: SintraState) -> None:
        """Test feedback for initial attempt with no history."""
        result = critic_node(base_state)

        assert result["critic_feedback"] == "Initial attempt."

    def test_identifies_speed_failure(self, base_state: SintraState) -> None:
        """Test that critic identifies speed failures."""
        base_state["history"] = [
            {
                "recipe": ModelRecipe(bits=8),
                "metrics": ExperimentResult(
                    actual_tps=30.0,  # Below target of 50
                    actual_vram_usage=4.0,
                    accuracy_score=0.8,
                    was_successful=True,
                ),
            }
        ]

        result = critic_node(base_state)

        assert "SPEED FAIL" in result["critic_feedback"]

    def test_tracks_best_recipe(self, base_state: SintraState) -> None:
        """Test that critic tracks the best recipe."""
        successful_run = {
            "recipe": ModelRecipe(bits=4),
            "metrics": ExperimentResult(
                actual_tps=60.0,
                actual_vram_usage=4.0,
                accuracy_score=0.75,
                was_successful=True,
            ),
        }
        base_state["history"] = [successful_run]

        result = critic_node(base_state)

        assert result.get("best_recipe") is not None

    def test_detects_repeated_recipe(self, base_state: SintraState) -> None:
        """Test that critic warns about repeated recipes."""
        same_recipe = ModelRecipe(bits=4, pruning_ratio=0.1)
        base_state["history"] = [
            {
                "recipe": same_recipe,
                "metrics": ExperimentResult(
                    actual_tps=40.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.7,
                    was_successful=True,
                ),
            },
            {
                "recipe": same_recipe,
                "metrics": ExperimentResult(
                    actual_tps=40.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.7,
                    was_successful=True,
                ),
            },
        ]

        result = critic_node(base_state)

        assert "repeated the exact same recipe" in result["critic_feedback"]


class TestCriticRouter:
    """Tests for critic_router function."""

    def test_routes_to_reporter_in_debug(self, base_state: SintraState) -> None:
        """Test routing to reporter in debug mode."""
        route = critic_router(base_state)
        assert route == "reporter"

    def test_routes_to_reporter_when_converged(self, base_state: SintraState) -> None:
        """Test routing to reporter when converged."""
        base_state["use_debug"] = False
        base_state["is_converged"] = True

        route = critic_router(base_state)
        assert route == "reporter"

    def test_routes_to_reporter_at_max_iterations(
        self, base_state: SintraState
    ) -> None:
        """Test routing to reporter at max iterations."""
        base_state["use_debug"] = False
        base_state["iteration"] = MAX_ITERATIONS

        route = critic_router(base_state)
        assert route == "reporter"

    def test_routes_to_architect_with_no_history(self, base_state: SintraState) -> None:
        """Test routing to architect when history is empty."""
        base_state["use_debug"] = False
        base_state["history"] = []

        route = critic_router(base_state)
        assert route == "architect"

    def test_routes_to_architect_after_crash(self, base_state: SintraState) -> None:
        """Test routing to architect after a crash."""
        base_state["use_debug"] = False
        base_state["history"] = [
            {
                "recipe": ModelRecipe(),
                "metrics": ExperimentResult(
                    actual_tps=0.0,
                    actual_vram_usage=0.0,
                    accuracy_score=0.0,
                    was_successful=False,
                    error_log="Out of memory",
                ),
            }
        ]

        route = critic_router(base_state)
        assert route == "architect"

    def test_routes_to_reporter_when_targets_met(self, base_state: SintraState) -> None:
        """Test routing to reporter when all targets are met."""
        base_state["use_debug"] = False
        base_state["history"] = [
            {
                "recipe": ModelRecipe(bits=4),
                "metrics": ExperimentResult(
                    actual_tps=60.0,  # Above target of 50
                    actual_vram_usage=6.0,  # Below VRAM limit of 8
                    accuracy_score=0.75,  # Above target of 0.7
                    was_successful=True,
                ),
            }
        ]

        route = critic_router(base_state)
        assert route == "reporter"

    def test_routes_to_architect_when_targets_not_met(
        self, base_state: SintraState
    ) -> None:
        """Test routing to architect when targets are not met."""
        base_state["use_debug"] = False
        base_state["history"] = [
            {
                "recipe": ModelRecipe(bits=8),
                "metrics": ExperimentResult(
                    actual_tps=40.0,  # Below target of 50
                    actual_vram_usage=6.0,
                    accuracy_score=0.8,
                    was_successful=True,
                ),
            }
        ]

        route = critic_router(base_state)
        assert route == "architect"


class TestConstants:
    """Tests for module constants."""

    def test_max_iterations_is_reasonable(self) -> None:
        """Test MAX_ITERATIONS is a reasonable value."""
        assert 5 <= MAX_ITERATIONS <= 20

    def test_default_output_file_is_json(self) -> None:
        """Test DEFAULT_OUTPUT_FILE has .json extension."""
        assert str(DEFAULT_OUTPUT_FILE).endswith(".json")
