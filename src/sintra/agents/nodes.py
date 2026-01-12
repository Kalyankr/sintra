# -*- coding: utf-8 -*-
"""Agent workflow nodes for the Sintra optimization loop."""

import json
from typing import Any, Dict, Literal

from sintra.agents.factory import get_architect_llm
from sintra.benchmarks.executor import MockExecutor, StandaloneExecutor
from sintra.profiles.models import ModelRecipe
from sintra.ui.console import log_transition

from .state import SintraState
from .utils import format_history_for_llm

# Configuration constants
MAX_ITERATIONS = 10
DEFAULT_OUTPUT_FILE = "optimized_recipe.json"

# Type alias for state updates
StateUpdate = Dict[str, Any]


def architect_node(state: SintraState) -> StateUpdate:
    """The Brain: Analyzes past performance and proposes the next compression strategy."""

    # DEBUG MODE: Skip the LLM and return a fixed recipe
    if state.get("use_debug"):
        log_transition("Architect", "DEBUG MODE: Bypassing LLM API", "arch.node")

        test_recipe = ModelRecipe(
            bits=2,
            pruning_ratio=0.1,
            layers_to_drop=[],
            method="GGUF",
        )

        return {
            "current_recipe": test_recipe,
            "iteration": state["iteration"] + 1,
            "is_converged": True,
        }
    log_transition(
        "Architect", f"Analyzing iteration {state['iteration']}...", "arch.node"
    )

    brain = get_architect_llm(state["llm_config"])
    profile = state["profile"]
    history_summary = format_history_for_llm(state["history"])

    history = state.get("history", [])

    # Create a string of previous attempts to shame the AI into changing
    past_attempts = ""
    for i, entry in enumerate(history):
        recipe = entry.get("recipe")
        metrics = entry.get("metrics")
        past_attempts += (
            f"- Attempt {i}: Recipe {recipe} failed with TPS: {metrics.actual_tps}\n"
        )

    system_prompt = f"""
    You are **Sintra**, an expert LLM Compression Architect.  
    Your mission is to design an optimal compression recipe for the model belonging to: {profile.name}.  
    You must balance speed, accuracy, and VRAM efficiency using quantization, pruning, and layer dropping.

    Your output MUST follow all rules below.

    ====================================================
    STRICT OUTPUT FORMAT (MANDATORY)
    ====================================================
    - Output **ONLY valid JSON**. No explanations, no comments.
    - JSON must contain exactly one object with the keys:
        - "bits": integer (e.g., 4, 8)
        - "pruning_ratio": decimal between 0.0 and 1.0 (e.g., 0.25)
        - "layers_to_drop": list of integers OR an empty list
    - Never output whole numbers like 20 or 50 for pruning. Only decimals.

    ====================================================
    CONSTRAINTS
    ====================================================
    - VRAM Limit: {profile.constraints.vram_gb} GB  
    - Target TPS (tokens/sec): {profile.targets.min_tokens_per_second}  
    - Target Accuracy Score: {profile.targets.min_accuracy_score}

    ====================================================
    OPTIMIZATION RULES
    ====================================================
    1. Always propose a **ModelRecipe** using quantization + pruning + layer dropping.
    2. If TPS is too low:
        -> decrease bits OR increase pruning_ratio OR increase layers_to_drop.
    3. If Accuracy is too low:
        -> increase bits OR decrease pruning_ratio OR reduce layers_to_drop.
    4. Always ensure the recipe fits within the VRAM limit.
    5. You may use any compression strategy, but the final recipe must obey all constraints.

    ====================================================
    PAST FAILURES (DO NOT REPEAT)
    ====================================================
    {past_attempts}

    - You MUST NOT repeat any failed recipe.
    - A recipe counts as "repeated" if **bits**, **pruning_ratio**, AND **layers_to_drop** all match a past failure.
    - If a recipe failed, you MUST change at least one of:
        -> bits  
        -> pruning_ratio  
        -> layers_to_drop

    ====================================================
    REASONING RULES (INTERNAL ONLY)
    ====================================================
    - Think step-by-step internally, but output ONLY the final JSON.
    - Internally evaluate:
        - VRAM feasibility  
        - Expected TPS  
        - Expected accuracy impact  
        - Differences from past failures  
    - Never reveal your reasoning or internal thoughts.

    ====================================================
    FINAL INSTRUCTION
    ====================================================
    Respond with ONLY the JSON object representing the new ModelRecipe.

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
    current_iter = state.get("iteration", 0)
    return {"current_recipe": new_recipe, "iteration": current_iter + 1}


def benchmarker_node(state: SintraState) -> StateUpdate:
    """The Lab: Executes the recipe and returns physical metrics."""
    recipe = state["current_recipe"]
    profile = state["profile"]
    log_transition(
        "Lab",
        f"Executing Surgery: {recipe.bits}-bit | Prune: {recipe.pruning_ratio}",
        "lab.node",
    )
    # Use Mock if in debug mode
    if state.get("use_debug"):
        executor = MockExecutor()
    else:
        executor = StandaloneExecutor()
    result = executor.run_benchmark(recipe, profile)

    # Return as a list because state['history'] is Annotated with operator.add
    return {"history": [{"recipe": recipe, "metrics": result}]}


def critic_node(state: SintraState) -> dict:
    history = state.get("history", [])
    if not history:
        return {"critic_feedback": "Initial attempt."}

    # last_run is a dict from your history list
    last_run = history[-1]
    metrics = last_run["metrics"]
    recipe = last_run["recipe"]
    targets = state["profile"].targets

    # Track the "Best So Far"
    current_best = state.get("best_recipe")
    is_better = False

    if metrics.was_successful:
        if not current_best:
            if metrics.accuracy_score >= targets.min_accuracy_score:
                is_better = True
        else:
            best_metrics = current_best["metrics"]
            if metrics.actual_tps > best_metrics.actual_tps:
                if metrics.accuracy_score >= targets.min_accuracy_score:
                    is_better = True

    # Advice Logic
    feedback = []
    if metrics.actual_tps < targets.min_tokens_per_second:
        feedback.append(f"SPEED FAIL: {metrics.actual_tps} TPS is below target.")
        feedback.append("ADVICE: Try 3-bit or 2-bit. Set pruning_ratio to 0.0.")

    # Anti-Loop Logic
    if len(history) > 1:
        prev_run = history[-2]
        if recipe == prev_run["recipe"]:
            feedback.append(
                "WARNING: You repeated the exact same recipe. Try something new!"
            )

    updates = {"critic_feedback": "\n".join(feedback)}
    if is_better:
        # Save the current successful run as the best_recipe
        updates["best_recipe"] = last_run

    return updates


def critic_router(state: SintraState) -> str:
    """
    The Judge: Evaluates the latest benchmark against hardware targets and
    Decides if we need another iteration or if we are done..
    """

    if state.get("use_debug") or state.get("is_converged"):
        return "reporter"

    if state.get("iteration", 0) >= MAX_ITERATIONS:
        log_transition(
            "Critic",
            f"GIVING UP: Max iterations ({MAX_ITERATIONS}) reached. Using best attempt.",
            "status.warn",
        )
        return "reporter"

    if not state["history"]:
        return "architect"

    last_experiment = state["history"][-1]
    metrics = last_experiment["metrics"]
    profile = state["profile"]

    if not metrics.was_successful:
        log_transition("Critic", f"Crash detected: {metrics.error_log}", "status.fail")
        return "architect"

    # Objective Analysis
    met_tps = metrics.actual_tps >= profile.targets.min_tokens_per_second
    met_accuracy = metrics.accuracy_score >= profile.targets.min_accuracy_score
    under_vram = metrics.actual_vram_usage <= profile.constraints.vram_gb

    if met_tps and met_accuracy and under_vram:
        log_transition(
            "Critic", "TARGETS ACHIEVED. Optimization converged.", "status.success"
        )
        return "reporter"

    log_transition("Critic", "Performance gaps detected. Retrying...", "critic.node")
    return "architect"


def reporter_node(state: SintraState) -> dict:
    """
    The Archivist: Saves the winning recipe to a JSON file using Pydantic v2 standards.
    """
    log_transition("Reporter", "Archiving the final 'Golden Recipe'...", "hw.profile")

    if not state["history"]:
        log_transition("Reporter", "Error: No history found to report.", "status.fail")
        return state

    last_entry = state["history"][-1]

    output = {
        "hardware_profile": state["profile"].name,
        "recipe": last_entry["recipe"].model_dump()
        if hasattr(last_entry["recipe"], "model_dump")
        else last_entry["recipe"],
        "performance": last_entry["metrics"].model_dump()
        if hasattr(last_entry["metrics"], "model_dump")
        else last_entry["metrics"],
    }

    with open("optimized_recipe.json", "w") as f:
        json.dump(output, f, indent=4)

    return state
