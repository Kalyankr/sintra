"""Tests for the reflector module."""

import pytest

from sintra.agents.reflector import (
    FailureAnalysis,
    Reflection,
    StrategyAdjustment,
    _all_targets_met,
    _analyze_history,
    reflector_node,
)
from sintra.profiles.models import (
    Constraints,
    ExperimentResult,
    HardwareProfile,
    LLMConfig,
    ModelRecipe,
    Targets,
)


@pytest.fixture
def sample_profile():
    """Create a sample hardware profile."""
    return HardwareProfile(
        name="test-device",
        constraints=Constraints(
            vram_gb=8.0,
            cpu_arch="x86_64",
            has_cuda=False,
        ),
        targets=Targets(
            min_tokens_per_second=30.0,
            min_accuracy_score=0.7,
        ),
    )


@pytest.fixture
def sample_state(sample_profile):
    """Create a sample state."""
    return {
        "profile": sample_profile,
        "llm_config": LLMConfig(),
        "target_model_id": "test-model",
        "backend": "gguf",
        "run_id": "test-run",
        "iteration": 0,
        "history": [],
        "is_converged": False,
        "use_debug": False,
        "use_mock": True,
        "current_recipe": None,
        "critic_feedback": "",
        "best_recipe": None,
    }


class TestFailureAnalysis:
    """Tests for FailureAnalysis model."""

    def test_create_failure_analysis(self):
        failure = FailureAnalysis(
            failure_type="speed",
            root_cause="TPS too low",
            severity="medium",
        )
        assert failure.failure_type == "speed"
        assert failure.severity == "medium"


class TestStrategyAdjustment:
    """Tests for StrategyAdjustment model."""

    def test_create_strategy_adjustment(self):
        adjustment = StrategyAdjustment(
            parameter="bits",
            direction="decrease",
            magnitude="small",
            reasoning="Need more speed",
        )
        assert adjustment.parameter == "bits"
        assert adjustment.direction == "decrease"


class TestReflection:
    """Tests for Reflection model."""

    def test_create_empty_reflection(self):
        reflection = Reflection(iteration_analyzed=1)
        assert reflection.iteration_analyzed == 1
        assert reflection.failures == []
        assert reflection.patterns == []

    def test_create_full_reflection(self):
        reflection = Reflection(
            iteration_analyzed=3,
            failures=[
                FailureAnalysis(
                    failure_type="speed", root_cause="too slow", severity="high"
                )
            ],
            patterns=["consistent speed failures"],
            adjustments=[
                StrategyAdjustment(
                    parameter="bits",
                    direction="decrease",
                    magnitude="medium",
                    reasoning="need more speed",
                )
            ],
            confidence=0.8,
            summary="Speed is the main issue",
        )
        assert len(reflection.failures) == 1
        assert len(reflection.adjustments) == 1
        assert reflection.confidence == 0.8


class TestAllTargetsMet:
    """Tests for _all_targets_met helper."""

    def test_returns_false_for_unsuccessful(self, sample_profile):
        metrics = ExperimentResult(
            actual_tps=50.0,
            actual_vram_usage=4.0,
            accuracy_score=0.8,
            was_successful=False,
        )
        assert not _all_targets_met(metrics, sample_profile)

    def test_returns_true_when_all_met(self, sample_profile):
        metrics = ExperimentResult(
            actual_tps=35.0,  # > 30
            actual_vram_usage=6.0,  # < 8
            accuracy_score=0.75,  # > 0.7
            was_successful=True,
        )
        assert _all_targets_met(metrics, sample_profile)

    def test_returns_false_when_tps_low(self, sample_profile):
        metrics = ExperimentResult(
            actual_tps=25.0,  # < 30
            actual_vram_usage=4.0,
            accuracy_score=0.8,
            was_successful=True,
        )
        assert not _all_targets_met(metrics, sample_profile)

    def test_returns_false_when_accuracy_low(self, sample_profile):
        metrics = ExperimentResult(
            actual_tps=35.0,
            actual_vram_usage=4.0,
            accuracy_score=0.5,  # < 0.7
            was_successful=True,
        )
        assert not _all_targets_met(metrics, sample_profile)


class TestAnalyzeHistory:
    """Tests for _analyze_history function."""

    def test_identifies_speed_failures(self, sample_profile, sample_state):
        history = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.1),
                "metrics": ExperimentResult(
                    actual_tps=20.0,  # Below target 30
                    actual_vram_usage=4.0,
                    accuracy_score=0.8,
                    was_successful=True,
                ),
            }
        ]

        reflection = _analyze_history(history, sample_profile, sample_state)

        assert any(f.failure_type == "speed" for f in reflection.failures)

    def test_identifies_accuracy_failures(self, sample_profile, sample_state):
        history = [
            {
                "recipe": ModelRecipe(bits=2, pruning_ratio=0.3),
                "metrics": ExperimentResult(
                    actual_tps=40.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.5,  # Below target 0.7
                    was_successful=True,
                ),
            }
        ]

        reflection = _analyze_history(history, sample_profile, sample_state)

        assert any(f.failure_type == "accuracy" for f in reflection.failures)

    def test_identifies_crash_failures(self, sample_profile, sample_state):
        history = [
            {
                "recipe": ModelRecipe(bits=2, pruning_ratio=0.5),
                "metrics": ExperimentResult(
                    actual_tps=0.0,
                    actual_vram_usage=0.0,
                    accuracy_score=0.0,
                    was_successful=False,
                    error_log="OOM error",
                ),
            }
        ]

        reflection = _analyze_history(history, sample_profile, sample_state)

        assert any(f.failure_type == "crash" for f in reflection.failures)

    def test_generates_adjustments_for_speed_failures(
        self, sample_profile, sample_state
    ):
        history = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.1),
                "metrics": ExperimentResult(
                    actual_tps=15.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.8,
                    was_successful=True,
                ),
            },
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.2),
                "metrics": ExperimentResult(
                    actual_tps=18.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.75,
                    was_successful=True,
                ),
            },
        ]

        reflection = _analyze_history(history, sample_profile, sample_state)

        # Should recommend decreasing bits
        bit_adjustment = next(
            (a for a in reflection.adjustments if a.parameter == "bits"),
            None,
        )
        assert bit_adjustment is not None
        assert bit_adjustment.direction == "decrease"

    def test_identifies_oscillation_pattern(self, sample_profile, sample_state):
        history = [
            {
                "recipe": ModelRecipe(bits=2, pruning_ratio=0.3),
                "metrics": ExperimentResult(
                    actual_tps=40.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.5,  # Accuracy failure
                    was_successful=True,
                ),
            },
            {
                "recipe": ModelRecipe(bits=6, pruning_ratio=0.1),
                "metrics": ExperimentResult(
                    actual_tps=15.0,  # Speed failure
                    actual_vram_usage=4.0,
                    accuracy_score=0.85,
                    was_successful=True,
                ),
            },
        ]

        reflection = _analyze_history(history, sample_profile, sample_state)

        assert any("oscillating" in p.lower() for p in reflection.patterns)


class TestReflectorNode:
    """Tests for reflector_node function."""

    def test_returns_none_for_empty_history(self, sample_state):
        sample_state["history"] = []

        result = reflector_node(sample_state)

        assert result["reflection"] is None

    def test_returns_none_in_debug_mode(self, sample_state):
        sample_state["use_debug"] = True
        sample_state["history"] = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.1),
                "metrics": ExperimentResult(
                    actual_tps=20.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.8,
                    was_successful=True,
                ),
            }
        ]

        result = reflector_node(sample_state)

        assert result["reflection"] is None

    def test_returns_none_when_successful(self, sample_state):
        sample_state["history"] = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.1),
                "metrics": ExperimentResult(
                    actual_tps=35.0,  # Meets target
                    actual_vram_usage=4.0,
                    accuracy_score=0.8,  # Meets target
                    was_successful=True,
                ),
            }
        ]

        result = reflector_node(sample_state)

        assert result["reflection"] is None

    def test_returns_reflection_when_failed(self, sample_state):
        sample_state["history"] = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.1),
                "metrics": ExperimentResult(
                    actual_tps=20.0,  # Below target
                    actual_vram_usage=4.0,
                    accuracy_score=0.8,
                    was_successful=True,
                ),
            }
        ]

        result = reflector_node(sample_state)

        assert result["reflection"] is not None
        assert isinstance(result["reflection"], Reflection)
