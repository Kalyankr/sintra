"""Tests for history formatting utilities."""

import pytest

from sintra.agents.utils import format_history_for_llm
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
