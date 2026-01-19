"""Utility functions for the Sintra Agent.

Handles string formatting, telemetry parsing, and log cleaning.
"""

from typing import Any

from sintra.profiles.models import ExperimentResult, ModelRecipe

# Type alias for history entries
HistoryEntry = dict[str, ModelRecipe | ExperimentResult]


def format_history_for_llm(history: list[HistoryEntry]) -> str:
    """Narrates the experiment history for LLM reasoning.

    Formats the relationship between recipes and hardware performance
    in a human-readable narrative that the LLM can reason about.

    Args:
        history: List of experiment entries, each containing 'recipe' and 'metrics'.

    Returns:
        Formatted string summarizing all experiments.
    """
    if not history:
        return "No previous experiments recorded. This is the baseline run."

    formatted_logs = []
    for i, entry in enumerate(history):
        recipe = entry["recipe"]
        metrics = entry["metrics"]

        status = "SUCCESS" if metrics.was_successful else "FAILED"
        log = (
            f"--- Attempt #{i + 1} [{status}] ---\n"
            f"STRATEGY: {recipe.bits}-bit quantization via {recipe.method}\n"
            f"SURGERY: Pruned {recipe.pruning_ratio * 100}% of weights; "
            f"Dropped layers: {recipe.layers_to_drop}\n"
            f"PERFORMANCE: {metrics.actual_tps:.2f} TPS | {metrics.actual_vram_usage:.2f}GB VRAM\n"
            f"SCORE: Reasoning Accuracy = {metrics.accuracy_score:.4f}"
        )
        if metrics.error_log:
            log += f"\nERROR: {metrics.error_log}"

        formatted_logs.append(log)

    return "\n\n".join(formatted_logs)


def is_duplicate_recipe(
    recipe: ModelRecipe, history: list[HistoryEntry], tolerance: float = 0.05
) -> bool:
    """Check if a recipe is essentially a duplicate of one already tried.

    A recipe is considered duplicate if bits match AND pruning_ratio is within
    tolerance AND layers_to_drop are identical.

    Args:
        recipe: The proposed recipe to check.
        history: List of previous experiment entries.
        tolerance: How close pruning_ratio can be to count as duplicate (default 5%).

    Returns:
        True if recipe is a duplicate, False otherwise.
    """
    for entry in history:
        past_recipe = entry["recipe"]

        # Check bits match exactly
        if recipe.bits != past_recipe.bits:
            continue

        # Check pruning_ratio is within tolerance
        if abs(recipe.pruning_ratio - past_recipe.pruning_ratio) > tolerance:
            continue

        # Check layers_to_drop are identical
        if set(recipe.layers_to_drop) != set(past_recipe.layers_to_drop):
            continue

        # All conditions match - it's a duplicate
        return True

    return False


def get_untried_variations(history: list[HistoryEntry]) -> dict[str, Any]:
    """Suggest recipe variations that haven't been tried yet.

    Analyzes history to find gaps in the search space.

    Args:
        history: List of previous experiment entries.

    Returns:
        Dict with suggested bits, pruning_ratios, and layers_to_drop options.
    """
    tried_bits = {entry["recipe"].bits for entry in history}
    tried_pruning = {round(entry["recipe"].pruning_ratio, 2) for entry in history}

    all_bits = {2, 3, 4, 5, 6, 8}
    all_pruning = {0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5}

    return {
        "untried_bits": sorted(all_bits - tried_bits),
        "untried_pruning": sorted(all_pruning - tried_pruning),
        "tried_combinations": [
            f"{e['recipe'].bits}-bit, {e['recipe'].pruning_ratio:.0%} prune"
            for e in history
        ],
    }
