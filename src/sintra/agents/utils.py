"""Utility functions for the Sintra Agent.

Handles string formatting, telemetry parsing, and log cleaning.
"""

from typing import Any, Dict, List, Union

from sintra.profiles.models import ExperimentResult, ModelRecipe

# Type alias for history entries
HistoryEntry = Dict[str, Union[ModelRecipe, ExperimentResult]]


def format_history_for_llm(history: List[HistoryEntry]) -> str:
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
