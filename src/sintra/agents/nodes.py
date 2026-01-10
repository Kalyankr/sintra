from ..agents.factory import get_architect_llm
from .state import SintraState
from .utils import format_history_for_llm


def architect_node(state: SintraState) -> dict:
    """
    The Brain: Analyzes past performance and proposes the next
    compression strategy (The Recipe).
    """

    # 'llm_config' would be passed in the state or as a global
    brain = get_architect_llm(state["llm_config"])

    # 'Memory' context
    profile = state["profile"]
    history_summary = format_history_for_llm(state["history"])

    # optimization context
    system_prompt = f"""
    You are the Sintra Architect. Your goal is to compress a large LLM for: {profile.name}.
    
    CONSTRAINTS:
    - VRAM Limit: {profile.constraints.vram_gb} GB of VRAM
    - Target TPS: {profile.targets.min_tokens_per_second} tokens per second
    - Target Accuracy: {profile.targets.min_accuracy_score} accuracy score
    
    Current Iteration: {state["iteration"]}
    
    Your task is to propose a ModelRecipe (bits, pruning_ratio, layers_to_drop).
    If the previous attempt had low TPS, increase pruning or decrease bits.
    If the previous attempt had low Accuracy, reduce pruning or increase bits.
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


def critic_router(state: SintraState) -> str:
    """
    The Judge: Decides whether to continue optimizing or stop.
    """
    if not state["history"]:
        return "continue"

    last_result = state["history"][-1]["metrics"]
    profile = state["profile"]

    # Success Criteria
    met_accuracy = last_result.accuracy_score >= profile.targets.min_accuracy_score
    met_latency = last_result.actual_tps >= profile.targets.min_tokens_per_second

    if met_accuracy and met_latency:
        print("Target Achieved! Converging...")
        return "end"

    if state["iteration"] >= 10:
        print("Max iterations reached without convergence.")
        return "end"

    return "continue"
