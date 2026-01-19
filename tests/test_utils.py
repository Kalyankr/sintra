"""Tests for history formatting utilities."""


from sintra.agents.utils import (
    format_history_for_llm,
    get_untried_variations,
    is_duplicate_recipe,
)
from sintra.profiles.models import ExperimentResult, ModelRecipe


class TestFormatHistoryForLlm:
    """Tests for format_history_for_llm function."""

    def test_empty_history(self) -> None:
        """Test formatting with no history."""
        result = format_history_for_llm([])
        assert "No previous experiments" in result
        assert "baseline run" in result

    def test_single_successful_entry(self) -> None:
        """Test formatting a single successful experiment."""
        history = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.1, method="GGUF"),
                "metrics": ExperimentResult(
                    actual_tps=55.0,
                    actual_vram_usage=4.5,
                    accuracy_score=0.75,
                    was_successful=True,
                ),
            }
        ]

        result = format_history_for_llm(history)

        assert "Attempt #1" in result
        assert "SUCCESS" in result
        assert "4-bit" in result
        assert "GGUF" in result
        assert "55.00 TPS" in result
        assert "0.7500" in result

    def test_failed_entry_includes_error(self) -> None:
        """Test that failed entries include error log."""
        history = [
            {
                "recipe": ModelRecipe(bits=2),
                "metrics": ExperimentResult(
                    actual_tps=0.0,
                    actual_vram_usage=0.0,
                    accuracy_score=0.0,
                    was_successful=False,
                    error_log="Model too compressed",
                ),
            }
        ]

        result = format_history_for_llm(history)

        assert "FAILED" in result
        assert "ERROR:" in result
        assert "Model too compressed" in result

    def test_multiple_entries(self) -> None:
        """Test formatting multiple history entries."""
        history = [
            {
                "recipe": ModelRecipe(bits=8),
                "metrics": ExperimentResult(
                    actual_tps=30.0,
                    actual_vram_usage=8.0,
                    accuracy_score=0.9,
                    was_successful=True,
                ),
            },
            {
                "recipe": ModelRecipe(bits=4),
                "metrics": ExperimentResult(
                    actual_tps=60.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.75,
                    was_successful=True,
                ),
            },
        ]

        result = format_history_for_llm(history)

        assert "Attempt #1" in result
        assert "Attempt #2" in result
        assert "8-bit" in result
        assert "4-bit" in result

    def test_pruning_percentage_format(self) -> None:
        """Test that pruning ratio is formatted as percentage."""
        history = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.25),
                "metrics": ExperimentResult(
                    actual_tps=60.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.7,
                    was_successful=True,
                ),
            }
        ]

        result = format_history_for_llm(history)

        assert "25.0%" in result


class TestIsDuplicateRecipe:
    """Tests for is_duplicate_recipe function."""

    def test_empty_history_not_duplicate(self) -> None:
        """Test that any recipe is not duplicate with empty history."""
        recipe = ModelRecipe(bits=4, pruning_ratio=0.2)
        assert is_duplicate_recipe(recipe, []) is False

    def test_exact_match_is_duplicate(self) -> None:
        """Test that exact match is detected as duplicate."""
        history = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.2, layers_to_drop=[]),
                "metrics": ExperimentResult(
                    actual_tps=50.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.7,
                    was_successful=True,
                ),
            }
        ]
        recipe = ModelRecipe(bits=4, pruning_ratio=0.2, layers_to_drop=[])
        assert is_duplicate_recipe(recipe, history) is True

    def test_different_bits_not_duplicate(self) -> None:
        """Test that different bits means not duplicate."""
        history = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.2),
                "metrics": ExperimentResult(
                    actual_tps=50.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.7,
                    was_successful=True,
                ),
            }
        ]
        recipe = ModelRecipe(bits=8, pruning_ratio=0.2)
        assert is_duplicate_recipe(recipe, history) is False

    def test_different_pruning_not_duplicate(self) -> None:
        """Test that significantly different pruning is not duplicate."""
        history = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.2),
                "metrics": ExperimentResult(
                    actual_tps=50.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.7,
                    was_successful=True,
                ),
            }
        ]
        recipe = ModelRecipe(bits=4, pruning_ratio=0.4)
        assert is_duplicate_recipe(recipe, history) is False

    def test_within_tolerance_is_duplicate(self) -> None:
        """Test that pruning within tolerance counts as duplicate."""
        history = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.20),
                "metrics": ExperimentResult(
                    actual_tps=50.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.7,
                    was_successful=True,
                ),
            }
        ]
        # 0.22 is within 0.05 tolerance of 0.20
        recipe = ModelRecipe(bits=4, pruning_ratio=0.22)
        assert is_duplicate_recipe(recipe, history) is True

    def test_different_layers_not_duplicate(self) -> None:
        """Test that different layers_to_drop is not duplicate."""
        history = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.2, layers_to_drop=[]),
                "metrics": ExperimentResult(
                    actual_tps=50.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.7,
                    was_successful=True,
                ),
            }
        ]
        recipe = ModelRecipe(bits=4, pruning_ratio=0.2, layers_to_drop=[1, 2])
        assert is_duplicate_recipe(recipe, history) is False


class TestGetUntriedVariations:
    """Tests for get_untried_variations function."""

    def test_empty_history_returns_all_options(self) -> None:
        """Test that empty history returns all possible options."""
        result = get_untried_variations([])
        assert 4 in result["untried_bits"]
        assert 8 in result["untried_bits"]
        assert 0.2 in result["untried_pruning"]

    def test_filters_tried_bits(self) -> None:
        """Test that tried bits are excluded from suggestions."""
        history = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.2),
                "metrics": ExperimentResult(
                    actual_tps=50.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.7,
                    was_successful=True,
                ),
            }
        ]
        result = get_untried_variations(history)
        assert 4 not in result["untried_bits"]
        assert 8 in result["untried_bits"]

    def test_tracks_tried_combinations(self) -> None:
        """Test that tried combinations are listed."""
        history = [
            {
                "recipe": ModelRecipe(bits=4, pruning_ratio=0.2),
                "metrics": ExperimentResult(
                    actual_tps=50.0,
                    actual_vram_usage=4.0,
                    accuracy_score=0.7,
                    was_successful=True,
                ),
            }
        ]
        result = get_untried_variations(history)
        assert len(result["tried_combinations"]) == 1
        assert "4-bit" in result["tried_combinations"][0]
