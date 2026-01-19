"""ReAct-style architect node with tool use capabilities.

This module implements a more agentic architect that can:
1. Use tools to research before proposing recipes
2. Follow the ReAct pattern (Reason → Act → Observe)
3. Make more informed decisions based on tool outputs
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from sintra.agents.factory import get_tool_enabled_llm
from sintra.agents.state import SintraState
from sintra.agents.tools import get_architect_tools
from sintra.agents.utils import (
    get_untried_variations,
    is_duplicate_recipe,
)
from sintra.profiles.models import ModelRecipe
from sintra.ui.console import log_transition

logger = logging.getLogger(__name__)

# Configuration
MAX_TOOL_ITERATIONS = 5  # Maximum tool calls before forcing a recipe


class ReActStep(BaseModel):
    """A single step in the ReAct reasoning chain."""

    thought: str = Field(description="The agent's reasoning about what to do next")
    action: str = Field(
        description="The action to take: 'tool_call' or 'propose_recipe'"
    )
    tool_name: str | None = Field(default=None, description="Name of tool to call")
    tool_input: dict[str, Any] | None = Field(
        default=None, description="Input for tool"
    )
    observation: str | None = Field(
        default=None, description="Result from tool call"
    )


class ArchitectReasoning(BaseModel):
    """Complete reasoning chain from the architect."""

    steps: list[ReActStep] = Field(
        default_factory=list, description="ReAct reasoning steps"
    )
    final_recipe: ModelRecipe | None = Field(
        default=None, description="Proposed recipe"
    )
    confidence: float = Field(default=0.5, description="Confidence in the recipe (0-1)")
    reasoning_summary: str = Field(default="", description="Summary of reasoning")


REACT_SYSTEM_PROMPT = """You are **Sintra**, an expert LLM Compression Architect using the ReAct pattern.

Your mission is to design an optimal compression recipe for edge AI deployment.
You have access to tools to help you make informed decisions.

## Available Tools

1. **search_similar_models** - Find existing quantized models on HuggingFace
   - Use when: Starting optimization or looking for reference implementations
   - Input: base_model (str), task (str), max_results (int)

2. **estimate_compression_impact** - Predict performance before running
   - Use when: Evaluating a potential recipe before proposing
   - Input: model_size_billions, target_bits, pruning_ratio, layers_to_drop, total_layers

3. **query_hardware_capabilities** - Understand target device limits
   - Use when: Unclear what the hardware can support
   - Input: device_name, available_memory_gb, has_gpu, gpu_type

4. **lookup_quantization_benchmarks** - Get known benchmark data
   - Use when: Wanting reference performance numbers
   - Input: model_family (str), bits (int)

## ReAct Pattern

Follow this pattern:
1. **Thought**: Reason about what information you need
2. **Action**: Either call a tool OR propose a final recipe
3. **Observation**: Process tool results (I'll provide this)
4. Repeat until you have enough information to propose a recipe

## Output Format

For tool calls, respond with JSON:
```json
{{"thought": "I need to check what bit widths work for this hardware", "action": "tool_call", "tool_name": "query_hardware_capabilities", "tool_input": {{"device_name": "...", "available_memory_gb": ...}}}}
```

For final recipe, respond with JSON:
```json
{{"thought": "Based on my research, 4-bit with 0.1 pruning balances speed and accuracy", "action": "propose_recipe", "recipe": {{"bits": 4, "pruning_ratio": 0.1, "layers_to_drop": [], "method": "GGUF"}}}}
```

## Constraints

- Target Hardware: {hardware_name}
- VRAM Limit: {vram_gb} GB
- Target TPS: {target_tps}
- Target Accuracy: {target_accuracy}
- Model: {model_id}

## Past Attempts (DO NOT REPEAT)
{past_attempts}

## Strategy Hints
{strategy_hints}

IMPORTANT: Make 1-3 tool calls to gather information, then propose a recipe.
Do not make more than {max_tools} tool calls.
"""


def react_architect_node(state: SintraState) -> dict[str, Any]:
    """ReAct-style architect node with tool use.

    This architect:
    1. Analyzes the situation
    2. Uses tools to gather information
    3. Proposes an informed recipe

    Args:
        state: Current workflow state

    Returns:
        State update with new recipe and reasoning chain
    """
    # DEBUG MODE: Skip the LLM
    if state.get("use_debug"):
        log_transition("Architect", "DEBUG MODE: Bypassing LLM API", "arch.node")
        return _debug_recipe(state)

    log_transition(
        "Architect",
        f"[ReAct] Analyzing iteration {state['iteration']}...",
        "arch.node",
    )

    # Get tools and LLM
    tools = get_architect_tools()
    tool_map = {tool.name: tool for tool in tools}
    llm = get_tool_enabled_llm(state["llm_config"], tools)

    # Build context
    profile = state["profile"]
    history = state.get("history", [])

    past_attempts = _format_past_attempts(history)
    strategy_hints = _get_strategy_hints(history)

    system_prompt = REACT_SYSTEM_PROMPT.format(
        hardware_name=profile.name,
        vram_gb=profile.constraints.vram_gb,
        target_tps=profile.targets.min_tokens_per_second,
        target_accuracy=profile.targets.min_accuracy_score,
        model_id=state["target_model_id"],
        past_attempts=past_attempts,
        strategy_hints=strategy_hints,
        max_tools=MAX_TOOL_ITERATIONS,
    )

    # ReAct loop
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content="Analyze the situation and propose an optimal compression recipe."
        ),
    ]

    reasoning_steps: list[ReActStep] = []
    tool_iterations = 0
    final_recipe = None

    while tool_iterations < MAX_TOOL_ITERATIONS:
        try:
            response = llm.invoke(messages)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return _fallback_recipe(state, str(e))

        # Check if LLM wants to use tools
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                log_transition(
                    "Architect",
                    f"[Tool] Calling {tool_name}...",
                    "arch.node",
                )

                # Execute tool
                if tool_name in tool_map:
                    try:
                        tool_result = tool_map[tool_name].invoke(tool_args)
                        observation = (
                            json.dumps(tool_result, indent=2)
                            if isinstance(tool_result, (dict, list))
                            else str(tool_result)
                        )
                    except Exception as e:
                        observation = f"Tool error: {e!s}"
                else:
                    observation = f"Unknown tool: {tool_name}"

                # Record step
                reasoning_steps.append(
                    ReActStep(
                        thought=f"Need to call {tool_name}",
                        action="tool_call",
                        tool_name=tool_name,
                        tool_input=tool_args,
                        observation=observation,
                    )
                )

                # Add tool result to messages
                messages.append(response)
                messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=tool_call["id"],
                    )
                )

                tool_iterations += 1
        else:
            # No tool calls - try to extract recipe from response
            final_recipe = _extract_recipe_from_response(response, state)
            if final_recipe:
                reasoning_steps.append(
                    ReActStep(
                        thought="Proposing final recipe based on research",
                        action="propose_recipe",
                    )
                )
                break

            # Ask for explicit recipe
            messages.append(response)
            messages.append(
                HumanMessage(
                    content="Please propose a final recipe in JSON format with keys: bits, pruning_ratio, layers_to_drop, method"
                )
            )
            tool_iterations += 1

    # If we exhausted iterations without a recipe, use fallback
    if final_recipe is None:
        log_transition(
            "Architect",
            "[ReAct] Max iterations reached, using fallback",
            "status.warn",
        )
        return _fallback_recipe(state, "Max tool iterations reached")

    # Check for duplicates
    if is_duplicate_recipe(final_recipe, history):
        log_transition(
            "Architect",
            "[ReAct] Duplicate detected, modifying recipe",
            "status.warn",
        )
        final_recipe = _modify_to_avoid_duplicate(final_recipe, history)

    # Build reasoning summary
    reasoning_summary = _build_reasoning_summary(reasoning_steps)

    log_transition(
        "Architect",
        f"[ReAct] Proposing: {final_recipe.bits}-bit, {final_recipe.pruning_ratio:.0%} pruning",
        "arch.node",
    )

    current_iter = state.get("iteration", 0)
    return {
        "current_recipe": final_recipe,
        "iteration": current_iter + 1,
        "reasoning_chain": reasoning_steps,
        "reasoning_summary": reasoning_summary,
    }


def _debug_recipe(state: SintraState) -> dict[str, Any]:
    """Return a debug recipe without LLM calls."""
    test_recipe = ModelRecipe(
        bits=4,
        pruning_ratio=0.1,
        layers_to_drop=[],
        method="GGUF",
    )
    return {
        "current_recipe": test_recipe,
        "iteration": state["iteration"] + 1,
        "is_converged": True,
    }


def _fallback_recipe(state: SintraState, error: str) -> dict[str, Any]:
    """Return a fallback recipe when LLM fails."""
    log_transition(
        "Architect",
        f"Using fallback recipe due to: {error}",
        "status.warn",
    )

    fallback = ModelRecipe(
        bits=4,
        pruning_ratio=0.1,
        layers_to_drop=[],
        method="GGUF",
    )

    return {
        "current_recipe": fallback,
        "iteration": state.get("iteration", 0) + 1,
        "reasoning_summary": f"Fallback recipe due to error: {error}",
    }


def _format_past_attempts(history: list[dict]) -> str:
    """Format past attempts for the prompt."""
    if not history:
        return "No previous attempts. This is the first iteration."

    lines = []
    for i, entry in enumerate(history):
        recipe = entry.get("recipe")
        metrics = entry.get("metrics")
        status = "✓" if metrics.was_successful else "✗"
        lines.append(
            f"{status} Attempt {i + 1}: bits={recipe.bits}, pruning={recipe.pruning_ratio:.2f}, "
            f"layers_dropped={recipe.layers_to_drop} → TPS={metrics.actual_tps:.1f}, "
            f"accuracy={metrics.accuracy_score:.2f}"
        )
    return "\n".join(lines)


def _get_strategy_hints(history: list[dict]) -> str:
    """Generate strategy hints based on history."""
    if not history:
        return "Start with a balanced approach: 4-bit quantization, minimal pruning."

    variations = get_untried_variations(history)
    hints = []

    if variations.get("untried_bits"):
        hints.append(f"Untried bit widths: {variations['untried_bits']}")
    if variations.get("untried_pruning"):
        hints.append(f"Untried pruning ratios: {variations['untried_pruning']}")

    # Analyze trends
    last_result = history[-1]["metrics"]
    if last_result.actual_tps < 20:
        hints.append("Speed is low - consider more aggressive quantization or pruning")
    if last_result.accuracy_score < 0.6:
        hints.append("Accuracy is low - consider less aggressive compression")

    return "\n".join(hints) if hints else "Continue exploring the search space."


def _extract_recipe_from_response(
    response: AIMessage, state: SintraState
) -> ModelRecipe | None:
    """Try to extract a recipe from the LLM response."""
    content = response.content

    # Try to parse JSON from content
    try:
        # Look for JSON block in the response
        import re

        json_match = re.search(r'\{[^{}]*"bits"[^{}]*\}', content, re.DOTALL)
        if json_match:
            recipe_data = json.loads(json_match.group())

            # Handle nested recipe
            if "recipe" in recipe_data:
                recipe_data = recipe_data["recipe"]

            return ModelRecipe(
                bits=recipe_data.get("bits", 4),
                pruning_ratio=recipe_data.get("pruning_ratio", 0.1),
                layers_to_drop=recipe_data.get("layers_to_drop", []),
                method=recipe_data.get("method", "GGUF"),
            )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug(f"Failed to parse recipe from response: {e}")

    return None


def _modify_to_avoid_duplicate(recipe: ModelRecipe, history: list[dict]) -> ModelRecipe:
    """Modify a recipe to make it unique."""
    bit_options = [2, 3, 4, 5, 6, 8]
    pruning_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Try different bit widths first
    for bits in bit_options:
        if bits != recipe.bits:
            candidate = ModelRecipe(
                bits=bits,
                pruning_ratio=recipe.pruning_ratio,
                layers_to_drop=recipe.layers_to_drop,
                method=recipe.method,
            )
            if not is_duplicate_recipe(candidate, history):
                return candidate

    # Try different pruning ratios
    for pruning in pruning_options:
        if abs(pruning - recipe.pruning_ratio) > 0.05:
            candidate = ModelRecipe(
                bits=recipe.bits,
                pruning_ratio=pruning,
                layers_to_drop=recipe.layers_to_drop,
                method=recipe.method,
            )
            if not is_duplicate_recipe(candidate, history):
                return candidate

    # Last resort: change method
    return ModelRecipe(
        bits=recipe.bits,
        pruning_ratio=recipe.pruning_ratio + 0.05,
        layers_to_drop=recipe.layers_to_drop,
        method=recipe.method,
    )


def _build_reasoning_summary(steps: list[ReActStep]) -> str:
    """Build a human-readable summary of the reasoning chain."""
    if not steps:
        return "No reasoning steps recorded."

    summary_parts = []
    for i, step in enumerate(steps, 1):
        if step.action == "tool_call":
            summary_parts.append(f"{i}. Called {step.tool_name}")
        elif step.action == "propose_recipe":
            summary_parts.append(f"{i}. Proposed final recipe")

    return " → ".join(summary_parts)
