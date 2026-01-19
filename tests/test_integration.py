"""Integration tests for Sintra.

These tests verify that components work together correctly.
Some tests require external resources (models, llama.cpp) and are
marked to be skipped in CI environments.

Markers:
    - @pytest.mark.integration: All integration tests
    - @pytest.mark.slow: Tests that take >10 seconds
    - @pytest.mark.requires_model: Tests that need to download models
    - @pytest.mark.requires_llama_cpp: Tests that need llama.cpp installed
"""

import json
from pathlib import Path

import pytest

from sintra.agents.nodes import architect_node, benchmarker_node, critic_node
from sintra.checkpoint import (
    delete_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from sintra.persistence import HistoryDB
from sintra.profiles.models import (
    Constraints,
    HardwareProfile,
    LLMConfig,
    LLMProvider,
    ModelRecipe,
    Targets,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_profile():
    """Create a test hardware profile."""
    return HardwareProfile(
        name="integration-test-device",
        constraints=Constraints(vram_gb=4.0, cpu_arch="x86_64", has_cuda=False),
        targets=Targets(min_tokens_per_second=20, min_accuracy_score=0.6),
        supported_quantizations=["GGUF"],
    )


@pytest.fixture
def test_llm_config():
    """Create a test LLM config (won't be used in mock mode)."""
    return LLMConfig(provider=LLMProvider.OLLAMA, model_name="qwen3:8b")


@pytest.fixture
def test_state(test_profile, test_llm_config):
    """Create a complete test state for workflow."""
    return {
        "profile": test_profile,
        "llm_config": test_llm_config,
        "use_debug": True,  # Skip LLM calls
        "use_mock": True,  # Use MockExecutor
        "target_model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "run_id": "integration-test-run",
        "backend": "gguf",
        "iteration": 0,
        "history": [],
        "is_converged": False,
        "current_recipe": None,
        "critic_feedback": "",
        "best_recipe": None,
    }


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_history.db"
    return HistoryDB(db_path=db_path)


@pytest.fixture
def temp_checkpoint_dir(tmp_path, monkeypatch):
    """Use a temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    monkeypatch.setattr(
        "sintra.checkpoint.get_checkpoint_dir",
        lambda: checkpoint_dir,
    )
    return checkpoint_dir


# =============================================================================
# Mock Workflow Integration Tests
# =============================================================================


@pytest.mark.integration
class TestMockWorkflowIntegration:
    """Test the full workflow with mock executor (no real compression)."""

    def test_architect_produces_recipe(self, test_state):
        """Architect node should produce a valid recipe in debug mode."""
        result = architect_node(test_state)

        assert "current_recipe" in result
        assert isinstance(result["current_recipe"], ModelRecipe)
        assert result["current_recipe"].bits in [2, 3, 4, 5, 6, 8]
        assert result["iteration"] == 1

    def test_benchmarker_produces_metrics(self, test_state):
        """Benchmarker should produce metrics from mock executor."""
        # First get a recipe from architect
        architect_result = architect_node(test_state)
        test_state.update(architect_result)

        # Then run benchmarker
        result = benchmarker_node(test_state)

        assert "history" in result
        assert len(result["history"]) == 1

        entry = result["history"][0]
        assert "recipe" in entry
        assert "metrics" in entry
        assert entry["metrics"].actual_tps > 0
        assert 0 <= entry["metrics"].accuracy_score <= 1

    def test_critic_evaluates_results(self, test_state):
        """Critic should evaluate benchmark results."""
        # Get recipe and benchmark it
        architect_result = architect_node(test_state)
        test_state.update(architect_result)

        bench_result = benchmarker_node(test_state)
        test_state["history"] = bench_result["history"]

        # Run critic
        result = critic_node(test_state)

        assert "critic_feedback" in result

    def test_full_iteration_cycle(self, test_state):
        """Test a complete architect -> benchmarker -> critic cycle."""
        # Iteration 1: Architect
        arch_result = architect_node(test_state)
        test_state.update(arch_result)
        assert test_state["current_recipe"] is not None

        # Iteration 1: Benchmarker
        bench_result = benchmarker_node(test_state)
        test_state["history"] = bench_result["history"]
        assert len(test_state["history"]) == 1

        # Iteration 1: Critic
        critic_result = critic_node(test_state)
        test_state.update(critic_result)

        # Verify state is consistent
        assert test_state["iteration"] == 1
        assert len(test_state["history"]) == 1
        assert test_state["current_recipe"].bits == 2  # Debug mode uses 2-bit


# =============================================================================
# Persistence Integration Tests
# =============================================================================


@pytest.mark.integration
class TestPersistenceIntegration:
    """Test that persistence works across workflow runs."""

    def test_save_and_retrieve_experiments(self, temp_db, test_profile):
        """Experiments saved should be retrievable."""
        recipe = ModelRecipe(bits=4, pruning_ratio=0.1)
        from sintra.profiles.models import ExperimentResult

        result = ExperimentResult(
            actual_tps=25.5,
            actual_vram_usage=3.2,
            accuracy_score=0.72,
            was_successful=True,
        )

        # Save experiment
        temp_db.save_experiment(
            run_id="test-run-1",
            model_id="test/model",
            hardware_name="test-device",
            recipe=recipe,
            result=result,
            backend="gguf",
        )

        # Retrieve it
        experiments = temp_db.find_similar_experiments(
            model_id="test/model",
            hardware_name="test-device",
        )

        assert len(experiments) == 1
        assert experiments[0].recipe.bits == 4

    def test_best_recipe_selection(self, temp_db, test_profile):
        """Should find the best recipe based on accuracy."""
        from sintra.profiles.models import ExperimentResult

        # Save multiple experiments with different accuracy scores
        # get_best_recipe_for_hardware sorts by accuracy_score DESC
        for bits, tps, accuracy in [(4, 20.0, 0.9), (3, 35.0, 0.75), (2, 50.0, 0.6)]:
            recipe = ModelRecipe(bits=bits, pruning_ratio=0.0)
            result = ExperimentResult(
                actual_tps=tps,
                actual_vram_usage=2.0,
                accuracy_score=accuracy,
                was_successful=True,
            )
            temp_db.save_experiment(
                run_id=f"test-run-{bits}",
                model_id="test/model",
                hardware_name="test-device",
                recipe=recipe,
                result=result,
                backend="gguf",
            )

        # Get best recipe
        best = temp_db.get_best_recipe_for_hardware(
            model_id="test/model",
            hardware_name="test-device",
        )

        assert best is not None
        best_recipe, best_result = best
        assert best_recipe.bits == 4  # Highest accuracy (0.9)

    def test_run_lifecycle(self, temp_db, test_profile):
        """Test start_run and finish_run tracking."""
        run_id = "lifecycle-test"

        # Start run
        temp_db.start_run(
            run_id=run_id,
            model_id="test/model",
            profile=test_profile,
            backend="gguf",
        )

        # Check it's running
        incomplete = temp_db.get_incomplete_runs()
        assert any(r["run_id"] == run_id for r in incomplete)

        # Finish run
        temp_db.finish_run(
            run_id=run_id,
            final_iteration=5,
            is_converged=True,
            status="completed",
        )

        # Check it's no longer incomplete
        incomplete = temp_db.get_incomplete_runs()
        assert not any(r["run_id"] == run_id for r in incomplete)


# =============================================================================
# Checkpoint Integration Tests
# =============================================================================


@pytest.mark.integration
class TestCheckpointIntegration:
    """Test checkpoint save/load/resume functionality."""

    def test_checkpoint_save_load_cycle(self, test_state, temp_checkpoint_dir):
        """Checkpoint should preserve state across save/load."""
        run_id = "checkpoint-test-run"
        test_state["run_id"] = run_id

        # Run architect to get a recipe
        arch_result = architect_node(test_state)
        test_state.update(arch_result)

        # Run benchmarker to get history
        bench_result = benchmarker_node(test_state)
        test_state["history"] = bench_result["history"]

        # Save checkpoint
        save_checkpoint(run_id, test_state, iteration=1)

        # Verify checkpoint exists
        checkpoints = list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0]["run_id"] == run_id

        # Load checkpoint
        loaded = load_checkpoint(run_id)
        assert loaded is not None
        assert loaded["iteration"] == 1

        # Verify state was preserved
        loaded_state = loaded["state"]
        assert loaded_state["target_model_id"] == test_state["target_model_id"]
        assert len(loaded_state["history"]) == 1
        assert loaded_state["current_recipe"].bits == test_state["current_recipe"].bits

    def test_resume_continues_from_checkpoint(self, test_state, temp_checkpoint_dir):
        """Resuming should continue from saved iteration."""
        from sintra.profiles.models import ExperimentResult

        run_id = "resume-test-run"
        test_state["run_id"] = run_id
        test_state["iteration"] = 3

        # Create proper ExperimentResult objects for history
        mock_result = ExperimentResult(
            actual_tps=25.0,
            actual_vram_usage=2.5,
            accuracy_score=0.7,
            was_successful=True,
        )
        test_state["history"] = [
            {"recipe": ModelRecipe(bits=4), "metrics": mock_result},
            {"recipe": ModelRecipe(bits=3), "metrics": mock_result},
            {"recipe": ModelRecipe(bits=2), "metrics": mock_result},
        ]

        # Save at iteration 3
        save_checkpoint(run_id, test_state, iteration=3)

        # Load and verify iteration is preserved
        loaded = load_checkpoint(run_id)
        assert loaded["iteration"] == 3

    def test_checkpoint_cleanup(self, test_state, temp_checkpoint_dir):
        """Delete checkpoint should work."""
        run_id = "cleanup-test"
        test_state["run_id"] = run_id

        save_checkpoint(run_id, test_state, iteration=1)
        assert len(list_checkpoints()) == 1

        delete_checkpoint(run_id)
        assert len(list_checkpoints()) == 0


# =============================================================================
# CLI Integration Tests
# =============================================================================


@pytest.mark.integration
class TestCLIIntegration:
    """Test CLI commands work end-to-end."""

    def test_dry_run_produces_output(self, tmp_path):
        """--dry-run should produce config files without running."""
        import subprocess

        result = subprocess.run(
            [
                "uv",
                "run",
                "sintra",
                "--auto-detect",
                "--dry-run",
                "--output-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # Should succeed
        assert result.returncode == 0

        # Should create dry run config
        config_path = tmp_path / "dry_run_config.json"
        assert config_path.exists()

        # Config should be valid JSON
        with open(config_path) as f:
            config = json.load(f)
        assert config["mode"] == "dry-run"

    def test_debug_mock_completes(self, tmp_path, temp_checkpoint_dir):
        """--debug --mock should complete full workflow."""
        import subprocess

        result = subprocess.run(
            [
                "uv",
                "run",
                "sintra",
                "--auto-detect",
                "--debug",
                "--mock",
                "--output-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=60,
        )

        # Should succeed
        assert result.returncode == 0
        assert "OPTIMIZATION COMPLETE" in result.stdout


# =============================================================================
# Slow Integration Tests (require external resources)
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_model
class TestRealModelIntegration:
    """Tests that require downloading real models.

    These are skipped by default. Run with:
        pytest -m "requires_model" --run-slow
    """

    @pytest.mark.skip(reason="Requires model download - run manually")
    def test_tinyllama_download(self, tmp_path):
        """Test downloading TinyLlama from HuggingFace."""
        from sintra.compression.downloader import ModelDownloader

        downloader = ModelDownloader(cache_dir=tmp_path)
        model_path = downloader.download("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        assert model_path.exists()
        assert (model_path / "config.json").exists()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_llama_cpp
class TestGGUFIntegration:
    """Tests that require llama.cpp installed.

    These are skipped by default. Run with:
        pytest -m "requires_llama_cpp" --run-slow
    """

    @pytest.mark.skip(reason="Requires llama.cpp - run manually")
    def test_gguf_quantization(self, tmp_path):
        """Test GGUF quantization with real model."""
        from sintra.compression.quantizer import GGUFQuantizer

        # This would require a real model to be downloaded first
        quantizer = GGUFQuantizer(cache_dir=tmp_path)
        # quantizer.quantize(model_path, bits=4)
