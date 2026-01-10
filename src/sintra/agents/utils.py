"""
Utility functions for the Sintra Agent.
Handles string formatting, telemetry parsing, and log cleaning.
"""


def format_history_for_llm(history: list) -> str:
    """
    Narrates the experiment history so the LLM can reason about
    the relationship between recipes and hardware performance.
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
