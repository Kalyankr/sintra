"""Tests for checkpoint module."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from sintra.checkpoint import (
    cleanup_old_checkpoints,
    delete_checkpoint,
    deserialize_state,
    find_latest_checkpoint,
    get_checkpoint_dir,
    get_checkpoint_path,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
    serialize_state,
)
from sintra.profiles.models import (
    Constraints,
    HardwareProfile,
    LLMConfig,
    LLMProvider,
    ModelRecipe,
    Targets,
)


@pytest.fixture
def sample_profile():
    """Create a sample hardware profile."""
    return HardwareProfile(
        name="test-device",
        constraints=Constraints(vram_gb=8.0, cpu_arch="arm64", has_cuda=False),
        targets=Targets(min_tokens_per_second=30, min_accuracy_score=0.7),
        supported_quantizations=["GGUF", "ONNX"],
    )


@pytest.fixture
def sample_state(sample_profile):
    """Create a sample workflow state."""
    return {
        "profile": sample_profile,
        "llm_config": LLMConfig(provider=LLMProvider.OPENAI, model_name="gpt-4o"),
        "use_debug": False,
        "use_mock": True,
        "target_model_id": "test/model",
        "run_id": "test-run-123",
        "backend": "gguf",
        "iteration": 3,
        "history": [],
        "is_converged": False,
        "current_recipe": ModelRecipe(bits=4, pruning_ratio=0.1),
        "critic_feedback": "Test feedback",
        "best_recipe": None,
    }


@pytest.fixture
def checkpoint_dir(tmp_path, monkeypatch):
    """Use a temporary directory for checkpoints."""
    checkpoint_path = tmp_path / "checkpoints"
    monkeypatch.setattr(
        "sintra.checkpoint.get_checkpoint_dir",
        lambda: checkpoint_path,
    )
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


class TestGetCheckpointDir:
    """Tests for get_checkpoint_dir function."""
    
    def test_returns_path(self):
        """Should return a Path object."""
        result = get_checkpoint_dir()
        assert isinstance(result, Path)
    
    def test_default_location(self):
        """Should be under ~/.sintra/checkpoints."""
        result = get_checkpoint_dir()
        assert "sintra" in str(result)
        assert "checkpoints" in str(result)


class TestGetCheckpointPath:
    """Tests for get_checkpoint_path function."""
    
    def test_includes_run_id(self):
        """Should include run_id in filename."""
        result = get_checkpoint_path("my-run-id")
        assert "my-run-id" in str(result)
    
    def test_json_extension(self):
        """Should have .json extension."""
        result = get_checkpoint_path("test-run")
        assert result.suffix == ".json"


class TestSerializeState:
    """Tests for serialize_state function."""
    
    def test_serializes_profile(self, sample_state):
        """Should convert HardwareProfile to dict."""
        result = serialize_state(sample_state)
        assert isinstance(result["profile"], dict)
        assert result["profile"]["name"] == "test-device"
    
    def test_serializes_llm_config(self, sample_state):
        """Should convert LLMConfig to dict."""
        result = serialize_state(sample_state)
        assert isinstance(result["llm_config"], dict)
        assert result["llm_config"]["provider"] == "openai"
    
    def test_serializes_recipe(self, sample_state):
        """Should convert ModelRecipe to dict."""
        result = serialize_state(sample_state)
        assert isinstance(result["current_recipe"], dict)
        assert result["current_recipe"]["bits"] == 4
    
    def test_preserves_simple_values(self, sample_state):
        """Should preserve simple values unchanged."""
        result = serialize_state(sample_state)
        assert result["iteration"] == 3
        assert result["target_model_id"] == "test/model"
        assert result["use_mock"] is True


class TestDeserializeState:
    """Tests for deserialize_state function."""
    
    def test_deserializes_profile(self, sample_state):
        """Should convert dict back to HardwareProfile."""
        serialized = serialize_state(sample_state)
        result = deserialize_state(serialized)
        assert isinstance(result["profile"], HardwareProfile)
        assert result["profile"].name == "test-device"
    
    def test_deserializes_llm_config(self, sample_state):
        """Should convert dict back to LLMConfig."""
        serialized = serialize_state(sample_state)
        result = deserialize_state(serialized)
        assert isinstance(result["llm_config"], LLMConfig)
        assert result["llm_config"].provider == LLMProvider.OPENAI
    
    def test_deserializes_recipe(self, sample_state):
        """Should convert dict back to ModelRecipe."""
        serialized = serialize_state(sample_state)
        result = deserialize_state(serialized)
        assert isinstance(result["current_recipe"], ModelRecipe)
        assert result["current_recipe"].bits == 4
    
    def test_round_trip(self, sample_state):
        """Serialize then deserialize should preserve data."""
        serialized = serialize_state(sample_state)
        result = deserialize_state(serialized)
        
        assert result["profile"].name == sample_state["profile"].name
        assert result["iteration"] == sample_state["iteration"]
        assert result["current_recipe"].bits == sample_state["current_recipe"].bits


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""
    
    def test_creates_file(self, sample_state, checkpoint_dir):
        """Should create a checkpoint file."""
        save_checkpoint("test-run", sample_state, 1)
        
        checkpoint_file = checkpoint_dir / "test-run.json"
        assert checkpoint_file.exists()
    
    def test_file_contains_valid_json(self, sample_state, checkpoint_dir):
        """Should write valid JSON."""
        save_checkpoint("test-run", sample_state, 1)
        
        checkpoint_file = checkpoint_dir / "test-run.json"
        with open(checkpoint_file) as f:
            data = json.load(f)
        
        assert "run_id" in data
        assert "state" in data
    
    def test_includes_metadata(self, sample_state, checkpoint_dir):
        """Should include run_id, iteration, and timestamp."""
        save_checkpoint("test-run", sample_state, 5)
        
        checkpoint_file = checkpoint_dir / "test-run.json"
        with open(checkpoint_file) as f:
            data = json.load(f)
        
        assert data["run_id"] == "test-run"
        assert data["iteration"] == 5
        assert "timestamp" in data
    
    def test_returns_path(self, sample_state, checkpoint_dir):
        """Should return the path to the checkpoint file."""
        result = save_checkpoint("test-run", sample_state, 1)
        assert isinstance(result, Path)
        assert result.exists()


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""
    
    def test_loads_saved_checkpoint(self, sample_state, checkpoint_dir):
        """Should load a previously saved checkpoint."""
        save_checkpoint("test-run", sample_state, 3)
        
        result = load_checkpoint("test-run")
        
        assert result is not None
        assert result["run_id"] == "test-run"
        assert result["iteration"] == 3
    
    def test_returns_none_for_missing(self, checkpoint_dir):
        """Should return None if checkpoint doesn't exist."""
        result = load_checkpoint("nonexistent-run")
        assert result is None
    
    def test_deserializes_state(self, sample_state, checkpoint_dir):
        """Should deserialize state objects."""
        save_checkpoint("test-run", sample_state, 1)
        
        result = load_checkpoint("test-run")
        
        assert isinstance(result["state"]["profile"], HardwareProfile)
        assert isinstance(result["state"]["current_recipe"], ModelRecipe)


class TestListCheckpoints:
    """Tests for list_checkpoints function."""
    
    def test_empty_when_no_checkpoints(self, checkpoint_dir):
        """Should return empty list when no checkpoints."""
        result = list_checkpoints()
        assert result == []
    
    def test_lists_all_checkpoints(self, sample_state, checkpoint_dir):
        """Should list all saved checkpoints."""
        save_checkpoint("run-1", sample_state, 1)
        save_checkpoint("run-2", sample_state, 2)
        
        result = list_checkpoints()
        
        assert len(result) == 2
    
    def test_includes_metadata(self, sample_state, checkpoint_dir):
        """Should include run_id, model_id, iteration, timestamp."""
        save_checkpoint("test-run", sample_state, 5)
        
        result = list_checkpoints()
        
        assert result[0]["run_id"] == "test-run"
        assert result[0]["iteration"] == 5
        assert "timestamp" in result[0]
    
    def test_sorted_by_timestamp(self, sample_state, checkpoint_dir):
        """Should be sorted most recent first."""
        import time
        
        save_checkpoint("run-old", sample_state, 1)
        time.sleep(0.01)  # Ensure different timestamps
        save_checkpoint("run-new", sample_state, 2)
        
        result = list_checkpoints()
        
        assert result[0]["run_id"] == "run-new"
        assert result[1]["run_id"] == "run-old"


class TestDeleteCheckpoint:
    """Tests for delete_checkpoint function."""
    
    def test_deletes_existing(self, sample_state, checkpoint_dir):
        """Should delete an existing checkpoint."""
        save_checkpoint("test-run", sample_state, 1)
        
        result = delete_checkpoint("test-run")
        
        assert result is True
        assert not (checkpoint_dir / "test-run.json").exists()
    
    def test_returns_false_for_missing(self, checkpoint_dir):
        """Should return False if checkpoint doesn't exist."""
        result = delete_checkpoint("nonexistent")
        assert result is False


class TestFindLatestCheckpoint:
    """Tests for find_latest_checkpoint function."""
    
    def test_returns_none_when_empty(self, checkpoint_dir):
        """Should return None when no checkpoints."""
        result = find_latest_checkpoint()
        assert result is None
    
    def test_finds_most_recent(self, sample_state, checkpoint_dir):
        """Should find the most recent checkpoint."""
        import time
        
        save_checkpoint("run-old", sample_state, 1)
        time.sleep(0.01)
        sample_state["iteration"] = 5
        save_checkpoint("run-new", sample_state, 5)
        
        result = find_latest_checkpoint()
        
        assert result["run_id"] == "run-new"
    
    def test_filters_by_model(self, sample_state, checkpoint_dir):
        """Should filter by model_id when specified."""
        import time
        
        sample_state["target_model_id"] = "model-a"
        save_checkpoint("run-a", sample_state, 1)
        
        time.sleep(0.01)
        sample_state["target_model_id"] = "model-b"
        save_checkpoint("run-b", sample_state, 2)
        
        result = find_latest_checkpoint(model_id="model-a")
        
        assert result["run_id"] == "run-a"


class TestCleanupOldCheckpoints:
    """Tests for cleanup_old_checkpoints function."""
    
    def test_respects_max_count(self, sample_state, checkpoint_dir):
        """Should delete checkpoints over max_count."""
        for i in range(5):
            save_checkpoint(f"run-{i}", sample_state, i)
        
        deleted = cleanup_old_checkpoints(max_age_days=365, max_count=2)
        
        assert deleted == 3
        remaining = list_checkpoints()
        assert len(remaining) == 2
    
    def test_returns_zero_when_empty(self, checkpoint_dir):
        """Should return 0 when no checkpoints to delete."""
        result = cleanup_old_checkpoints()
        assert result == 0
