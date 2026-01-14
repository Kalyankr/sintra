"""Planner agent for strategic optimization planning.

This module implements a planner that creates a search strategy before
optimization begins, deciding on approaches like:
- Start aggressive, then tune
- Binary search on bit width
- Try known-good recipes first
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from sintra.agents.factory import get_critic_llm
from sintra.agents.state import SintraState
from sintra.persistence import format_history_from_db
from sintra.ui.console import log_transition

logger = logging.getLogger(__name__)


class OptimizationStep(BaseModel):
    """A single step in the optimization plan."""

    step_number: int = Field(description="Order of this step (1-indexed)")
    strategy: str = Field(
        description="Strategy type: 'explore', 'exploit', 'binary_search', 'known_good'"
    )
    description: str = Field(description="Human-readable description of what to try")
    target_bits: Optional[int] = Field(default=None, description="Suggested bit width")
    target_pruning: Optional[float] = Field(
        default=None, description="Suggested pruning ratio"
    )
    rationale: str = Field(description="Why this step makes sense")


class OptimizationPlan(BaseModel):
    """Complete optimization plan created by the planner."""

    model_id: str = Field(description="Model being optimized")
    hardware_name: str = Field(description="Target hardware")
    overall_strategy: str = Field(
        description="High-level strategy: 'conservative', 'aggressive', 'balanced', 'adaptive'"
    )
    steps: List[OptimizationStep] = Field(
        default_factory=list, description="Ordered list of optimization steps"
    )
    max_iterations: int = Field(
        default=10, description="Maximum iterations before stopping"
    )
    early_stop_threshold: float = Field(
        default=0.95, description="Stop if we achieve this fraction of targets"
    )
    fallback_strategy: str = Field(
        default="Use best result so far",
        description="What to do if plan fails",
    )
    confidence: float = Field(default=0.5, description="Confidence in this plan (0-1)")


PLANNER_SYSTEM_PROMPT = """You are the Strategic Planner for an LLM compression optimization system.

Your job is to create a multi-step optimization plan BEFORE any compression is attempted.
Analyze the hardware constraints, targets, and any historical data to design an efficient search strategy.

## Hardware Profile
- Device: {hardware_name}
- Available Memory: {vram_gb} GB
- Has CUDA: {has_cuda}

## Optimization Targets
- Minimum TPS: {target_tps} tokens/second
- Minimum Accuracy: {target_accuracy}

## Model Information
- Model ID: {model_id}
- Estimated Size: ~{estimated_size} parameters

## Historical Data (Previous Runs)
{historical_data}

## Available Strategies

1. **Conservative** - Start with gentle compression, increase if needed
   - Best for: Unknown models, accuracy-critical applications
   - Start: 8-bit, 0% pruning → reduce bits if TPS too low

2. **Aggressive** - Start with heavy compression, back off if accuracy suffers
   - Best for: Speed-critical edge devices, well-known models
   - Start: 4-bit, 20% pruning → increase bits if accuracy too low

3. **Binary Search** - Efficiently find optimal bit width
   - Best for: When historical data suggests bit width is key factor
   - Start: 4-bit → check → go to 3 or 6 based on result

4. **Known Good** - Start from previously successful recipes
   - Best for: When we have historical data for similar hardware
   - Start with best recipe from history, then fine-tune

5. **Adaptive** - Mix strategies based on feedback
   - Best for: Complex optimization landscapes
   - Start conservative, switch to aggressive if far from targets

## Output Format

Respond with a JSON optimization plan:
```json
{{
    "model_id": "{model_id}",
    "hardware_name": "{hardware_name}",
    "overall_strategy": "conservative|aggressive|balanced|adaptive",
    "steps": [
        {{
            "step_number": 1,
            "strategy": "explore|exploit|binary_search|known_good",
            "description": "What to try",
            "target_bits": 4,
            "target_pruning": 0.1,
            "rationale": "Why this makes sense"
        }}
    ],
    "max_iterations": 10,
    "early_stop_threshold": 0.95,
    "fallback_strategy": "What to do if plan fails",
    "confidence": 0.8
}}
```

Create a plan with 3-5 steps that efficiently explores the search space.
"""


def planner_node(state: SintraState) -> Dict[str, Any]:
    """Create an optimization plan before starting the search.

    This node:
    1. Analyzes hardware constraints and targets
    2. Checks historical data for insights
    3. Creates a strategic plan for the architect to follow

    Args:
        state: Current workflow state

    Returns:
        State update with optimization plan
    """
    # Skip planning in debug mode
    if state.get("use_debug"):
        log_transition("Planner", "DEBUG MODE: Using default plan", "arch.node")
        return {"optimization_plan": _create_default_plan(state)}

    log_transition("Planner", "Creating optimization strategy...", "arch.node")

    profile = state["profile"]
    model_id = state["target_model_id"]

    # Get historical data
    historical_data = format_history_from_db(
        model_id,
        profile.name,
        limit=5,
    )

    # Estimate model size from ID (rough heuristic)
    estimated_size = _estimate_model_size(model_id)

    # Try LLM-based planning first
    try:
        plan = _create_llm_plan(state, historical_data, estimated_size)
    except Exception as e:
        logger.warning(f"LLM planning failed: {e}, using rule-based plan")
        plan = _create_rule_based_plan(state, historical_data, estimated_size)

    log_transition(
        "Planner",
        f"Strategy: {plan.overall_strategy} ({len(plan.steps)} steps)",
        "arch.node",
    )

    return {"optimization_plan": plan}


def _create_llm_plan(
    state: SintraState,
    historical_data: str,
    estimated_size: str,
) -> OptimizationPlan:
    """Create a plan using LLM reasoning."""
    profile = state["profile"]

    prompt = PLANNER_SYSTEM_PROMPT.format(
        hardware_name=profile.name,
        vram_gb=profile.constraints.vram_gb,
        has_cuda=profile.constraints.has_cuda,
        target_tps=profile.targets.min_tokens_per_second,
        target_accuracy=profile.targets.min_accuracy_score,
        model_id=state["target_model_id"],
        estimated_size=estimated_size,
        historical_data=historical_data or "No historical data available.",
    )

    llm = get_critic_llm(state["llm_config"])
    structured_llm = llm.with_structured_output(OptimizationPlan)

    plan = structured_llm.invoke(
        [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "Create an optimization plan for this hardware and model.",
            },
        ]
    )

    return plan


def _create_rule_based_plan(
    state: SintraState,
    historical_data: str,
    estimated_size: str,
) -> OptimizationPlan:
    """Create a plan using rule-based logic (fallback)."""
    profile = state["profile"]
    model_id = state["target_model_id"]

    steps = []

    # Determine overall strategy based on constraints
    vram_gb = profile.constraints.vram_gb
    target_tps = profile.targets.min_tokens_per_second

    if vram_gb < 4:
        # Very constrained - be aggressive
        overall_strategy = "aggressive"
        steps = [
            OptimizationStep(
                step_number=1,
                strategy="explore",
                description="Start with 4-bit quantization for maximum compression",
                target_bits=4,
                target_pruning=0.0,
                rationale="Limited VRAM requires aggressive quantization",
            ),
            OptimizationStep(
                step_number=2,
                strategy="explore",
                description="Try 3-bit if 4-bit is too slow",
                target_bits=3,
                target_pruning=0.1,
                rationale="Further reduce if speed target not met",
            ),
            OptimizationStep(
                step_number=3,
                strategy="exploit",
                description="Fine-tune pruning to balance speed/accuracy",
                target_bits=4,
                target_pruning=0.2,
                rationale="Pruning can boost speed without changing bit width",
            ),
        ]
    elif target_tps > 50:
        # High speed requirement - aggressive
        overall_strategy = "aggressive"
        steps = [
            OptimizationStep(
                step_number=1,
                strategy="explore",
                description="Start with aggressive 3-bit quantization",
                target_bits=3,
                target_pruning=0.1,
                rationale="High TPS target requires aggressive compression",
            ),
            OptimizationStep(
                step_number=2,
                strategy="binary_search",
                description="Try 4-bit if accuracy is too low",
                target_bits=4,
                target_pruning=0.1,
                rationale="Binary search between 3-bit and 4-bit",
            ),
            OptimizationStep(
                step_number=3,
                strategy="exploit",
                description="Add pruning if still below TPS target",
                target_bits=4,
                target_pruning=0.3,
                rationale="Pruning can significantly boost speed",
            ),
        ]
    else:
        # Balanced approach
        overall_strategy = "balanced"
        steps = [
            OptimizationStep(
                step_number=1,
                strategy="explore",
                description="Start with balanced 4-bit quantization",
                target_bits=4,
                target_pruning=0.0,
                rationale="4-bit is a good starting point for most cases",
            ),
            OptimizationStep(
                step_number=2,
                strategy="binary_search",
                description="Adjust bits based on first result",
                target_bits=5,
                target_pruning=0.1,
                rationale="Increase bits if accuracy low, decrease if speed low",
            ),
            OptimizationStep(
                step_number=3,
                strategy="exploit",
                description="Fine-tune pruning ratio",
                target_bits=4,
                target_pruning=0.15,
                rationale="Small pruning adjustments for optimal balance",
            ),
            OptimizationStep(
                step_number=4,
                strategy="exploit",
                description="Try layer dropping if close to targets",
                target_bits=4,
                target_pruning=0.1,
                rationale="Layer dropping can provide final speed boost",
            ),
        ]

    return OptimizationPlan(
        model_id=model_id,
        hardware_name=profile.name,
        overall_strategy=overall_strategy,
        steps=steps,
        max_iterations=10,
        early_stop_threshold=0.95,
        fallback_strategy="Use best result from history",
        confidence=0.7,
    )


def _create_default_plan(state: SintraState) -> OptimizationPlan:
    """Create a simple default plan for debug mode."""
    return OptimizationPlan(
        model_id=state["target_model_id"],
        hardware_name=state["profile"].name,
        overall_strategy="balanced",
        steps=[
            OptimizationStep(
                step_number=1,
                strategy="explore",
                description="Debug: Single 4-bit attempt",
                target_bits=4,
                target_pruning=0.1,
                rationale="Debug mode - minimal exploration",
            ),
        ],
        max_iterations=1,
        early_stop_threshold=1.0,
        fallback_strategy="Accept any result",
        confidence=1.0,
    )


def _estimate_model_size(model_id: str) -> str:
    """Estimate model size from its ID (rough heuristic)."""
    model_lower = model_id.lower()

    # Check for size indicators in name (order matters - check larger first)
    if any(x in model_lower for x in ["70b", "72b", "65b"]):
        return "70B"
    elif any(x in model_lower for x in ["34b", "33b", "32b"]):
        return "34B"
    elif any(x in model_lower for x in ["13b", "14b"]):
        return "13B"
    elif any(x in model_lower for x in ["8b", "7b", "6b"]):
        return "7B"
    elif any(x in model_lower for x in ["3b", "2.7b"]):
        # Check this is not a 1.xb model mismatching
        if any(x in model_lower for x in ["1.3b", "1.1b"]):
            return "1B"
        return "3B"
    elif any(x in model_lower for x in ["1b", "1.1b", "1.3b"]):
        return "1B"
    elif any(x in model_lower for x in ["tiny", "small", "mini"]):
        return "1B"
    else:
        return "Unknown"


def get_plan_guidance(
    plan: OptimizationPlan, iteration: int
) -> Optional[OptimizationStep]:
    """Get the appropriate step from the plan for the current iteration.

    Args:
        plan: The optimization plan
        iteration: Current iteration number (1-indexed)

    Returns:
        The step to follow, or None if past plan length
    """
    if iteration <= len(plan.steps):
        return plan.steps[iteration - 1]
    return None
