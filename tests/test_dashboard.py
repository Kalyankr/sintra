"""Tests for the Gradio web UI dashboard module."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from sintra.profiles.models import (
    ExperimentResult,
    ModelRecipe,
)
from sintra.ui.dashboard import (
    check_gradio_available,
    format_experiment_table,
    format_run_info,
)


@pytest.fixture
def sample_experiments():
    """Create mock experiment records for table formatting."""
    records = []
    for i in range(3):
        record = MagicMock()
        record.iteration = i
        record.backend = "gguf"
        record.created_at = datetime(2025, 1, 1, 10, i, 0)
        record.recipe = MagicMock(spec=ModelRecipe)
        record.recipe.bits = 4
        record.recipe.pruning_ratio = 0.1 * i
        record.recipe.layers_to_drop = list(range(i))
        record.result = MagicMock(spec=ExperimentResult)
        record.result.was_successful = i != 1  # Middle one is failure
        record.result.actual_tps = 25.0 + i * 5
        record.result.accuracy_score = 0.8 - i * 0.1
        record.result.actual_vram_usage = 3.0 + i * 0.5
        records.append(record)
    return records


@pytest.fixture
def sample_run_info():
    """Create a sample run metadata dictionary."""
    return {
        "run_id": "abcdef1234567890abcdef1234567890",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "hardware_name": "test-device",
        "backend": "gguf",
        "status": "completed",
        "started_at": "2025-01-01T10:00:00",
        "finished_at": "2025-01-01T10:05:00",
        "final_iteration": 3,
        "is_converged": True,
    }


class TestCheckGradioAvailable:
    def test_returns_bool(self):
        result = check_gradio_available()
        assert isinstance(result, bool)


class TestFormatExperimentTable:
    def test_format_basic(self, sample_experiments):
        rows = format_experiment_table(sample_experiments)
        assert len(rows) == 3
        # Each row should have 10 columns
        for row in rows:
            assert len(row) == 10

    def test_success_status(self, sample_experiments):
        rows = format_experiment_table(sample_experiments)
        assert rows[0][0] == "✓"  # First is successful
        assert rows[1][0] == "✗"  # Second is failure
        assert rows[2][0] == "✓"  # Third is successful

    def test_iteration_column(self, sample_experiments):
        rows = format_experiment_table(sample_experiments)
        assert rows[0][1] == "0"
        assert rows[1][1] == "1"
        assert rows[2][1] == "2"

    def test_bits_column(self, sample_experiments):
        rows = format_experiment_table(sample_experiments)
        for row in rows:
            assert row[2] == "4"

    def test_backend_column(self, sample_experiments):
        rows = format_experiment_table(sample_experiments)
        for row in rows:
            assert row[8] == "gguf"

    def test_empty_experiments(self):
        rows = format_experiment_table([])
        assert rows == []


class TestFormatRunInfo:
    def test_format_basic(self, sample_run_info):
        result = format_run_info(sample_run_info)
        assert "Run ID" in result
        assert "abcdef123456" in result  # Truncated ID
        assert "TinyLlama" in result
        assert "test-device" in result
        assert "gguf" in result
        assert "completed" in result

    def test_converged_yes(self, sample_run_info):
        result = format_run_info(sample_run_info)
        assert "Yes" in result

    def test_converged_no(self, sample_run_info):
        sample_run_info["is_converged"] = False
        result = format_run_info(sample_run_info)
        assert "No" in result

    def test_in_progress(self, sample_run_info):
        sample_run_info["finished_at"] = None
        result = format_run_info(sample_run_info)
        assert "In progress" in result

    def test_empty_run(self):
        result = format_run_info({})
        assert "No run selected" in result

    def test_no_run(self):
        result = format_run_info(None)
        assert "No run selected" in result

    def test_returns_string(self, sample_run_info):
        result = format_run_info(sample_run_info)
        assert isinstance(result, str)
