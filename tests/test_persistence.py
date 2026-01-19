"""Tests for SQLite persistence module."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from sintra.persistence import (
    ExperimentRecord,
    HistoryDB,
    format_history_from_db,
)
from sintra.profiles.models import (
    Constraints,
    ExperimentResult,
    HardwareProfile,
    ModelRecipe,
    Targets,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_history.db"
        db = HistoryDB(db_path)
        yield db


@pytest.fixture
def sample_recipe():
    """Create a sample recipe for testing."""
    return ModelRecipe(
        bits=4,
        pruning_ratio=0.2,
        layers_to_drop=[10, 11, 12],
        method="GGUF",
    )


@pytest.fixture
def sample_result():
    """Create a sample result for testing."""
    return ExperimentResult(
        actual_tps=25.5,
        actual_vram_usage=4.2,
        accuracy_score=0.72,
        was_successful=True,
        error_log=None,
    )


@pytest.fixture
def sample_profile():
    """Create a sample hardware profile."""
    return HardwareProfile(
        name="Test Hardware",
        constraints=Constraints(vram_gb=8.0, cpu_arch="x86_64", has_cuda=False),
        targets=Targets(min_tokens_per_second=20.0, min_accuracy_score=0.65),
    )


class TestHistoryDB:
    """Tests for HistoryDB class."""

    def test_creates_database(self, temp_db: HistoryDB):
        """Should create database file."""
        assert temp_db.db_path.exists()

    def test_save_experiment(self, temp_db: HistoryDB, sample_recipe, sample_result):
        """Should save experiment to database."""
        exp_id = temp_db.save_experiment(
            run_id="test-run-1",
            model_id="TinyLlama/TinyLlama-1.1B",
            hardware_name="Test Hardware",
            recipe=sample_recipe,
            result=sample_result,
            backend="gguf",
            iteration=0,
        )

        assert exp_id > 0

    def test_find_similar_experiments(
        self, temp_db: HistoryDB, sample_recipe, sample_result
    ):
        """Should find experiments for same model."""
        # Save an experiment
        temp_db.save_experiment(
            run_id="test-run-1",
            model_id="TinyLlama/TinyLlama-1.1B",
            hardware_name="Test Hardware",
            recipe=sample_recipe,
            result=sample_result,
            backend="gguf",
        )

        # Find similar
        results = temp_db.find_similar_experiments("TinyLlama/TinyLlama-1.1B")

        assert len(results) == 1
        assert results[0].recipe.bits == 4
        assert results[0].result.actual_tps == 25.5

    def test_find_similar_filters_by_hardware(
        self, temp_db: HistoryDB, sample_recipe, sample_result
    ):
        """Should filter by hardware name."""
        # Save for Hardware A
        temp_db.save_experiment(
            run_id="run-1",
            model_id="TinyLlama/TinyLlama-1.1B",
            hardware_name="Hardware A",
            recipe=sample_recipe,
            result=sample_result,
            backend="gguf",
        )

        # Save for Hardware B
        temp_db.save_experiment(
            run_id="run-2",
            model_id="TinyLlama/TinyLlama-1.1B",
            hardware_name="Hardware B",
            recipe=sample_recipe,
            result=sample_result,
            backend="gguf",
        )

        # Find for Hardware A only
        results = temp_db.find_similar_experiments(
            "TinyLlama/TinyLlama-1.1B",
            hardware_name="Hardware A",
        )

        assert len(results) == 1
        assert results[0].hardware_name == "Hardware A"

    def test_find_successful_only(self, temp_db: HistoryDB, sample_recipe):
        """Should filter by success status."""
        # Save successful
        temp_db.save_experiment(
            run_id="run-1",
            model_id="TinyLlama/TinyLlama-1.1B",
            hardware_name="Test",
            recipe=sample_recipe,
            result=ExperimentResult(
                actual_tps=25.0,
                actual_vram_usage=4.0,
                accuracy_score=0.7,
                was_successful=True,
            ),
            backend="gguf",
        )

        # Save failed
        temp_db.save_experiment(
            run_id="run-2",
            model_id="TinyLlama/TinyLlama-1.1B",
            hardware_name="Test",
            recipe=sample_recipe,
            result=ExperimentResult(
                actual_tps=5.0,
                actual_vram_usage=8.0,
                accuracy_score=0.3,
                was_successful=False,
                error_log="OOM",
            ),
            backend="gguf",
        )

        # Find successful only
        results = temp_db.find_similar_experiments(
            "TinyLlama/TinyLlama-1.1B",
            successful_only=True,
        )

        assert len(results) == 1
        assert results[0].result.was_successful is True

    def test_get_best_recipe(self, temp_db: HistoryDB, sample_recipe):
        """Should return best recipe by accuracy."""
        # Save low accuracy
        temp_db.save_experiment(
            run_id="run-1",
            model_id="TinyLlama/TinyLlama-1.1B",
            hardware_name="Test",
            recipe=ModelRecipe(bits=4, pruning_ratio=0.1, method="GGUF"),
            result=ExperimentResult(
                actual_tps=25.0,
                actual_vram_usage=4.0,
                accuracy_score=0.6,
                was_successful=True,
            ),
            backend="gguf",
        )

        # Save high accuracy
        temp_db.save_experiment(
            run_id="run-2",
            model_id="TinyLlama/TinyLlama-1.1B",
            hardware_name="Test",
            recipe=ModelRecipe(bits=8, pruning_ratio=0.0, method="GGUF"),
            result=ExperimentResult(
                actual_tps=15.0,
                actual_vram_usage=6.0,
                accuracy_score=0.85,
                was_successful=True,
            ),
            backend="gguf",
        )

        best = temp_db.get_best_recipe_for_hardware("TinyLlama/TinyLlama-1.1B", "Test")

        assert best is not None
        recipe, result = best
        assert recipe.bits == 8
        assert result.accuracy_score == 0.85

    def test_get_failed_recipes(self, temp_db: HistoryDB):
        """Should return failed recipes."""
        failed_recipe = ModelRecipe(bits=2, pruning_ratio=0.5, method="GGUF")

        temp_db.save_experiment(
            run_id="run-1",
            model_id="TinyLlama/TinyLlama-1.1B",
            hardware_name="Test",
            recipe=failed_recipe,
            result=ExperimentResult(
                actual_tps=0.0,
                actual_vram_usage=0.0,
                accuracy_score=0.0,
                was_successful=False,
                error_log="Too aggressive compression",
            ),
            backend="gguf",
        )

        failed = temp_db.get_failed_recipes("TinyLlama/TinyLlama-1.1B")

        assert len(failed) == 1
        assert failed[0].bits == 2
        assert failed[0].pruning_ratio == 0.5


class TestRunManagement:
    """Tests for run start/finish tracking."""

    def test_start_and_finish_run(
        self, temp_db: HistoryDB, sample_profile, sample_recipe
    ):
        """Should track run lifecycle."""
        run_id = "test-run-123"

        # Start run
        temp_db.start_run(
            run_id=run_id,
            model_id="TinyLlama/TinyLlama-1.1B",
            profile=sample_profile,
            backend="gguf",
        )

        # Check it's running
        run = temp_db.get_run(run_id)
        assert run is not None
        assert run["status"] == "running"

        # Finish run
        temp_db.finish_run(
            run_id=run_id,
            final_iteration=5,
            is_converged=True,
            best_recipe=sample_recipe,
            status="completed",
        )

        # Check it's completed
        run = temp_db.get_run(run_id)
        assert run["status"] == "completed"
        assert run["is_converged"] == 1
        assert run["final_iteration"] == 5

    def test_get_incomplete_runs(self, temp_db: HistoryDB, sample_profile):
        """Should find runs that didn't complete."""
        # Start run but don't finish
        temp_db.start_run(
            run_id="incomplete-run",
            model_id="TinyLlama/TinyLlama-1.1B",
            profile=sample_profile,
            backend="gguf",
        )

        incomplete = temp_db.get_incomplete_runs()

        assert len(incomplete) == 1
        assert incomplete[0]["run_id"] == "incomplete-run"


class TestStatistics:
    """Tests for statistics gathering."""

    def test_get_statistics(self, temp_db: HistoryDB, sample_recipe, sample_result):
        """Should calculate correct statistics."""
        # Save some experiments
        for i in range(5):
            temp_db.save_experiment(
                run_id=f"run-{i}",
                model_id="TinyLlama/TinyLlama-1.1B",
                hardware_name="Test",
                recipe=sample_recipe,
                result=sample_result
                if i < 3
                else ExperimentResult(
                    actual_tps=0,
                    actual_vram_usage=0,
                    accuracy_score=0,
                    was_successful=False,
                ),
                backend="gguf",
            )

        stats = temp_db.get_statistics()

        assert stats["total_experiments"] == 5
        assert stats["successful_experiments"] == 3
        assert stats["success_rate"] == 0.6
        assert stats["unique_models"] == 1


class TestExperimentRecord:
    """Tests for ExperimentRecord class."""

    def test_to_dict(self, sample_recipe, sample_result):
        """Should convert to dictionary."""
        record = ExperimentRecord(
            id=1,
            run_id="test-run",
            model_id="TinyLlama/TinyLlama-1.1B",
            hardware_name="Test",
            recipe=sample_recipe,
            result=sample_result,
            backend="gguf",
            created_at=datetime(2026, 1, 14, 12, 0, 0),
            iteration=0,
        )

        d = record.to_dict()

        assert d["id"] == 1
        assert d["run_id"] == "test-run"
        assert d["recipe"]["bits"] == 4
        assert d["result"]["actual_tps"] == 25.5


class TestFormatHistory:
    """Tests for history formatting."""

    def test_format_empty_history(self, temp_db: HistoryDB):
        """Should handle no history gracefully."""
        # Use the temp_db's path
        # Set up global DB to use temp
        import sintra.persistence as persistence_module

        persistence_module._global_db = temp_db

        result = format_history_from_db("nonexistent/model", "Test")

        assert "No previous experiments" in result

        # Reset
        persistence_module._global_db = None

    def test_format_with_history(
        self, temp_db: HistoryDB, sample_recipe, sample_result
    ):
        """Should format history entries."""
        import sintra.persistence as persistence_module

        persistence_module._global_db = temp_db

        temp_db.save_experiment(
            run_id="run-1",
            model_id="TinyLlama/TinyLlama-1.1B",
            hardware_name="Test",
            recipe=sample_recipe,
            result=sample_result,
            backend="gguf",
        )

        result = format_history_from_db("TinyLlama/TinyLlama-1.1B", "Test")

        assert "bits=4" in result
        assert "TPS=25.5" in result
        assert "✓" in result  # Successful

        persistence_module._global_db = None
