"""Tests for benchmark executors."""

import pytest

from sintra.benchmarks.executor import BenchmarkExecutor, MockExecutor
from sintra.profiles.models import (
    Constraints,
    ExperimentResult,
    HardwareProfile,
    ModelRecipe,
    Targets,
)


class TestMockExecutor:
    """Tests for MockExecutor."""

    @pytest.fixture
    def profile(self) -> HardwareProfile:
        """Create a test hardware profile."""
        return HardwareProfile(
            name="Test Device",
            constraints=Constraints(vram_gb=8.0),
            targets=Targets(min_tokens_per_second=50.0, min_accuracy_score=0.7),
        )

    def test_is_benchmark_executor(self) -> None:
        """Test MockExecutor inherits from BenchmarkExecutor."""
        executor = MockExecutor()
        assert isinstance(executor, BenchmarkExecutor)

    def test_reproducible_with_seed(self, profile: HardwareProfile) -> None:
        """Test that seeded executor produces reproducible results."""
        executor1 = MockExecutor(seed=42)
        executor2 = MockExecutor(seed=42)
        recipe = ModelRecipe(bits=4, pruning_ratio=0.1)

        result1 = executor1.run_benchmark(recipe, profile)
        result2 = executor2.run_benchmark(recipe, profile)

        assert result1.actual_tps == result2.actual_tps
        assert result1.accuracy_score == result2.accuracy_score

    def test_different_seeds_different_results(self, profile: HardwareProfile) -> None:
        """Test that different seeds produce different results."""
        executor1 = MockExecutor(seed=42)
        executor2 = MockExecutor(seed=123)
        recipe = ModelRecipe(bits=4, pruning_ratio=0.1)

        result1 = executor1.run_benchmark(recipe, profile)
        result2 = executor2.run_benchmark(recipe, profile)

        # Results should differ (with very high probability)
        assert result1.actual_tps != result2.actual_tps

    def test_4bit_faster_than_8bit(self, profile: HardwareProfile) -> None:
        """Test that 4-bit quantization is faster than 8-bit."""
        executor = MockExecutor(seed=42)
        recipe_4bit = ModelRecipe(bits=4, pruning_ratio=0.0)
        recipe_8bit = ModelRecipe(bits=8, pruning_ratio=0.0)

        result_4bit = executor.run_benchmark(recipe_4bit, profile)

        executor = MockExecutor(seed=42)  # Reset seed for fair comparison
        result_8bit = executor.run_benchmark(recipe_8bit, profile)

        assert result_4bit.actual_tps > result_8bit.actual_tps

    def test_pruning_increases_speed(self, profile: HardwareProfile) -> None:
        """Test that higher pruning ratio increases speed."""
        executor = MockExecutor(seed=42)
        recipe_no_prune = ModelRecipe(bits=8, pruning_ratio=0.0)
        recipe_pruned = ModelRecipe(bits=8, pruning_ratio=0.5)

        result_no_prune = executor.run_benchmark(recipe_no_prune, profile)

        executor = MockExecutor(seed=42)  # Reset seed
        result_pruned = executor.run_benchmark(recipe_pruned, profile)

        assert result_pruned.actual_tps > result_no_prune.actual_tps

    def test_result_is_always_successful(self, profile: HardwareProfile) -> None:
        """Test that mock executor always returns successful results."""
        executor = MockExecutor(seed=42)
        recipe = ModelRecipe()

        result = executor.run_benchmark(recipe, profile)

        assert result.was_successful is True
        assert result.error_log is None

    def test_vram_usage_varies_by_bits(self, profile: HardwareProfile) -> None:
        """Test that VRAM usage depends on quantization bits."""
        executor = MockExecutor(seed=42)

        result_8bit = executor.run_benchmark(ModelRecipe(bits=8), profile)
        result_4bit = executor.run_benchmark(ModelRecipe(bits=4), profile)

        assert result_8bit.actual_vram_usage > result_4bit.actual_vram_usage

    def test_accuracy_decreases_with_compression(
        self, profile: HardwareProfile
    ) -> None:
        """Test that more aggressive compression decreases accuracy."""
        executor = MockExecutor(seed=42)
        recipe_conservative = ModelRecipe(bits=8, pruning_ratio=0.0)
        result_conservative = executor.run_benchmark(recipe_conservative, profile)

        executor = MockExecutor(seed=42)  # Reset seed
        recipe_aggressive = ModelRecipe(bits=4, pruning_ratio=0.5)
        result_aggressive = executor.run_benchmark(recipe_aggressive, profile)

        assert result_conservative.accuracy_score > result_aggressive.accuracy_score
