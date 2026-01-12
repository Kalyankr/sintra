"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from sintra.profiles.models import (
    Constraints,
    ExperimentResult,
    HardwareProfile,
    LLMConfig,
    LLMProvider,
    ModelRecipe,
    Targets,
)


class TestModelRecipe:
    """Tests for ModelRecipe model."""

    def test_default_values(self) -> None:
        """Test default recipe values."""
        recipe = ModelRecipe()
        assert recipe.bits == 4
        assert recipe.pruning_ratio == 0.0
        assert recipe.layers_to_drop == []
        assert recipe.method == "GGUF"

    def test_bits_validation(self) -> None:
        """Test bits must be between 2 and 8."""
        with pytest.raises(ValidationError):
            ModelRecipe(bits=1)
        with pytest.raises(ValidationError):
            ModelRecipe(bits=16)

        # Valid edge cases
        assert ModelRecipe(bits=2).bits == 2
        assert ModelRecipe(bits=8).bits == 8

    def test_pruning_ratio_validation(self) -> None:
        """Test pruning ratio bounds - negative values should fail."""
        with pytest.raises(ValidationError):
            ModelRecipe(pruning_ratio=-0.1)
        # Note: Values > 1.0 are auto-scaled (e.g., 110 → 1.1 which is clamped or fails)
        # Values that can't scale to valid range should fail
        with pytest.raises(ValidationError):
            ModelRecipe(pruning_ratio=-50.0)  # Negative values always fail

    def test_pruning_ratio_auto_scaling(self) -> None:
        """Test that pruning ratios > 1 are automatically scaled."""
        # If AI sends 50 instead of 0.5, it should be fixed
        recipe = ModelRecipe(pruning_ratio=50.0)
        assert recipe.pruning_ratio == 0.5

        recipe = ModelRecipe(pruning_ratio=25)
        assert recipe.pruning_ratio == 0.25

    def test_layers_to_drop(self) -> None:
        """Test layers_to_drop accepts list of integers."""
        recipe = ModelRecipe(layers_to_drop=[0, 1, 5, 10])
        assert recipe.layers_to_drop == [0, 1, 5, 10]


class TestConstraints:
    """Tests for Constraints model."""

    def test_required_fields(self) -> None:
        """Test vram_gb is required."""
        with pytest.raises(ValidationError):
            Constraints()

        constraints = Constraints(vram_gb=8.0)
        assert constraints.vram_gb == 8.0

    def test_optional_fields(self) -> None:
        """Test optional fields have defaults."""
        constraints = Constraints(vram_gb=8.0)
        assert constraints.cpu_arch is None
        assert constraints.has_cuda is False

    def test_all_fields(self) -> None:
        """Test all fields can be set."""
        constraints = Constraints(vram_gb=16.0, cpu_arch="x86_64", has_cuda=True)
        assert constraints.cpu_arch == "x86_64"
        assert constraints.has_cuda is True


class TestTargets:
    """Tests for Targets model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        targets = Targets(min_tokens_per_second=50.0, min_accuracy_score=0.7)
        assert targets.min_tokens_per_second == 50.0
        assert targets.min_accuracy_score == 0.7

    def test_optional_latency(self) -> None:
        """Test max_latency_ms is optional."""
        targets = Targets(min_tokens_per_second=50.0, min_accuracy_score=0.7)
        assert targets.max_latency_ms is None

        targets = Targets(
            min_tokens_per_second=50.0, min_accuracy_score=0.7, max_latency_ms=300.0
        )
        assert targets.max_latency_ms == 300.0


class TestHardwareProfile:
    """Tests for HardwareProfile model."""

    def test_complete_profile(self) -> None:
        """Test creating a complete hardware profile."""
        profile = HardwareProfile(
            name="Test Device",
            constraints=Constraints(vram_gb=8.0),
            targets=Targets(min_tokens_per_second=50.0, min_accuracy_score=0.7),
        )
        assert profile.name == "Test Device"
        assert profile.supported_quantizations is None

    def test_with_quantizations(self) -> None:
        """Test profile with supported quantizations."""
        profile = HardwareProfile(
            name="GPU Device",
            constraints=Constraints(vram_gb=16.0, has_cuda=True),
            targets=Targets(min_tokens_per_second=100.0, min_accuracy_score=0.8),
            supported_quantizations=["GGUF", "AWQ", "GPTQ"],
        )
        assert profile.supported_quantizations == ["GGUF", "AWQ", "GPTQ"]


class TestExperimentResult:
    """Tests for ExperimentResult model."""

    def test_successful_result(self) -> None:
        """Test creating a successful experiment result."""
        result = ExperimentResult(
            actual_tps=85.5,
            actual_vram_usage=6.2,
            accuracy_score=0.78,
            was_successful=True,
        )
        assert result.was_successful is True
        assert result.error_log is None

    def test_failed_result(self) -> None:
        """Test creating a failed experiment result."""
        result = ExperimentResult(
            actual_tps=0.0,
            actual_vram_usage=0.0,
            accuracy_score=0.0,
            was_successful=False,
            error_log="Out of memory",
        )
        assert result.was_successful is False
        assert result.error_log == "Out of memory"


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_default_values(self) -> None:
        """Test default LLM configuration."""
        config = LLMConfig()
        assert config.provider == LLMProvider.OLLAMA
        assert config.model_name == "qwen3:8b"
        assert config.temperature == 0.8

    def test_custom_config(self) -> None:
        """Test custom LLM configuration."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI, model_name="gpt-4o", temperature=0.5
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.model_name == "gpt-4o"


class TestLLMProvider:
    """Tests for LLMProvider enum."""

    def test_all_providers(self) -> None:
        """Test all providers are defined."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GOOGLE.value == "google"
        assert LLMProvider.OLLAMA.value == "ollama"
