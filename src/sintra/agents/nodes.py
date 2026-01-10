import json
from pathlib import Path

from sintra.agents.factory import get_architect_llm
from sintra.benchmarks.executor import MockExecutor  # StandaloneExecutor
from sintra.ui.console import log_transition

from .state import SintraState
from .utils import format_history_for_llm


def architect_node(state: SintraState) -> dict:
    """
    The Brain: Analyzes past performance and proposes the next
    compression strategy (The Recipe).
    """
    brain = get_architect_llm(state["llm_config"])
    profile = state["profile"]
    history_summary = format_history_for_llm(state["history"])

    log_transition(
        "Architect", f"Analyzing iteration {state['iteration']}...", "arch.node"
    )

    system_prompt = f"""
    You are the Sintra Architect. Your goal is to compress a large LLM for: {profile.name}.
    
    CONSTRAINTS:
    - VRAM Limit: {profile.constraints.vram_gb} GB
    - Target TPS: {profile.targets.min_tokens_per_second} 
    - Target Accuracy: {profile.targets.min_accuracy_score}
    
    STRICT RULES:
    1. Propose a ModelRecipe with 'bits', 'pruning_ratio', and 'layers_to_drop'.
    2. If TPS is too low: Decrease bits or increase pruning.
    3. If Accuracy is too low: Increase bits or decrease pruning.
    """

    new_recipe = brain.invoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"History:\n{history_summary}\nPropose next recipe.",
            },
        ]
    )

    return {"current_recipe": new_recipe, "iteration": state["iteration"] + 1}


def benchmarker_node(state: SintraState) -> dict:
    """
    The Lab: Executes the recipe and returns physical metrics.
    """
    recipe = state["current_recipe"]
    log_transition(
        "Lab",
        f"Executing Surgery: {recipe.bits}-bit | Prune: {recipe.pruning_ratio}",
        "lab.node",
    )

    # --- SWAP EXECUTORS HERE ---
    # executor = StandaloneExecutor() # Use this for real hardware
    executor = MockExecutor()  # Use this for testing logic
    # ---------------------------

    result = executor.run_benchmark(recipe)

    # Return as a list because state['history'] is Annotated with operator.add
    return {"history": [{"recipe": recipe, "metrics": result}]}


def critic_router(state: SintraState) -> str:
    """
    The Judge: Evaluates the latest benchmark against hardware targets.
    """
    if not state["history"]:
        return "continue"

    last_experiment = state["history"][-1]
    metrics = last_experiment["metrics"]
    profile = state["profile"]

    if not metrics.was_successful:
        log_transition("Critic", f"Crash detected: {metrics.error_log}", "status.fail")
        return "continue"

    # Objective Analysis
    met_tps = metrics.actual_tps >= profile.targets.min_tokens_per_second
    met_accuracy = metrics.accuracy_score >= profile.targets.min_accuracy_score
    under_vram = metrics.actual_vram_usage <= profile.constraints.vram_gb

    if met_tps and met_accuracy and under_vram:
        log_transition(
            "Critic", "TARGETS ACHIEVED. Optimization converged.", "status.success"
        )
        return "end"

    if state["iteration"] >= 10:
        log_transition(
            "Critic", "MAX ITERATIONS REACHED. Stopping search.", "status.fail"
        )
        return "end"

    log_transition("Critic", "Performance gaps detected. Retrying...", "critic.node")
    return "continue"


def reporter_node(state: SintraState) -> dict:
    """
    The Archivist: Saves the winning recipe to a JSON file using Pydantic v2 standards.
    """
    log_transition("Reporter", "Archiving the final 'Golden Recipe'...", "hw.profile")

    successes = [e for e in state["history"] if e["metrics"].was_successful]

    if not successes:
        log_transition("Reporter", "No successful recipes to archive.", "status.fail")
        return state

    final_entry = successes[-1]
    recipe = final_entry["recipe"]
    metrics = final_entry["metrics"]

    report_data = {
        "hardware_profile": state["profile"].name,
        "recipe": recipe.model_dump(),
        "performance": {
            "tps": metrics.actual_tps,
            "vram_gb": metrics.actual_vram_usage,
            "accuracy": metrics.accuracy_score,
        },
    }

    output_path = Path("optimized_recipe.json")
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=4)

    log_transition(
        "Reporter", f"SUCCESS: Recipe saved to {output_path}", "status.success"
    )
    return state
