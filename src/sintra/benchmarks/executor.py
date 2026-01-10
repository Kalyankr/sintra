import random

from ..agents.state import ExperimentResult, ModelRecipe


class MockExecutor:
    """Simulates hardware behavior for testing the Agent's logic loop."""

    def run_benchmark(self, recipe: ModelRecipe) -> ExperimentResult:
        # Logic: Lower bits = Higher Speed, but Lower Accuracy
        # Base speed 2.0 TPS, +3.0 TPS if we use 4-bit
        speed_boost = 3.0 if recipe.bits == 4 else 0.0
        prune_boost = recipe.pruning_ratio * 10

        mock_tps = 2.0 + speed_boost + prune_boost + random.uniform(-0.5, 0.5)

        # Logic: More pruning/lower bits = Lower Accuracy
        accuracy_penalty = (recipe.pruning_ratio * 0.5) + (
            0.2 if recipe.bits == 4 else 0
        )
        mock_accuracy = max(0.1, 0.9 - accuracy_penalty + random.uniform(-0.05, 0.05))

        return ExperimentResult(
            actual_tps=round(mock_tps, 2),
            actual_vram_usage=4.2 if recipe.bits == 8 else 2.1,
            accuracy_score=round(mock_accuracy, 2),
            was_successful=True,
        )
