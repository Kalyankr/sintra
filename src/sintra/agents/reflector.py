"""Self-reflection node for analyzing failures and improving strategy.

This module implements a reflector agent that:
1. Analyzes why previous attempts failed
2. Identifies patterns in failures
3. Provides strategic guidance to the architect
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from sintra.agents.factory import get_critic_llm
from sintra.agents.state import SintraState
from sintra.ui.console import log_transition

logger = logging.getLogger(__name__)


class FailureAnalysis(BaseModel):
    """Analysis of why a compression attempt failed."""

    failure_type: str = Field(
        description="Type of failure: 'speed', 'accuracy', 'memory', 'crash'"
    )
    root_cause: str = Field(description="Identified root cause of the failure")
    severity: str = Field(description="low, medium, or high")


class StrategyAdjustment(BaseModel):
    """Recommended adjustment to the optimization strategy."""

    parameter: str = Field(
        description="Which parameter to adjust: bits, pruning, layers"
    )
    direction: str = Field(description="increase, decrease, or maintain")
    magnitude: str = Field(description="small, medium, or large change")
    reasoning: str = Field(description="Why this adjustment is recommended")


class Reflection(BaseModel):
    """Complete reflection output from the reflector node."""

    iteration_analyzed: int = Field(description="Which iteration was analyzed")
    failures: list[FailureAnalysis] = Field(
        default_factory=list, description="List of identified failures"
    )
    patterns: list[str] = Field(
        default_factory=list, description="Patterns observed across attempts"
    )
    adjustments: list[StrategyAdjustment] = Field(
        default_factory=list, description="Recommended strategy adjustments"
    )
    confidence: float = Field(
        default=0.5, description="Confidence in the analysis (0-1)"
    )
    summary: str = Field(default="", description="Human-readable summary")


REFLECTOR_SYSTEM_PROMPT = """You are a Self-Reflection Agent analyzing compression optimization attempts.

Your job is to:
1. Analyze WHY the last attempt(s) failed to meet targets
2. Identify PATTERNS across multiple attempts
3. Recommend STRATEGIC ADJUSTMENTS for the next attempt

## Current Targets
- Target TPS: {target_tps}
- Target Accuracy: {target_accuracy}
- VRAM Limit: {vram_gb} GB

## Last {num_attempts} Attempts
{attempts_summary}

## Analysis Guidelines

### Failure Types
- **speed**: TPS below target (need more aggressive compression)
- **accuracy**: Accuracy below target (need less aggressive compression)  
- **memory**: VRAM exceeded limit
- **crash**: Model failed to load/run

### Pattern Recognition
Look for:
- Are we oscillating between speed and accuracy failures?
- Is a particular parameter consistently at its limit?
- Are we repeating similar configurations?

### Strategy Adjustments
- If speed failures dominate → recommend more quantization or pruning
- If accuracy failures dominate → recommend backing off compression
- If oscillating → recommend finding middle ground
- If stuck in local minimum → recommend exploring different direction

## Output Format

Provide your analysis as JSON:
```json
{{
  "iteration_analyzed": <int>,
  "failures": [
    {{"failure_type": "speed|accuracy|memory|crash", "root_cause": "...", "severity": "low|medium|high"}}
  ],
  "patterns": ["pattern 1", "pattern 2"],
  "adjustments": [
    {{"parameter": "bits|pruning|layers", "direction": "increase|decrease|maintain", "magnitude": "small|medium|large", "reasoning": "..."}}
  ],
  "confidence": <0.0-1.0>,
  "summary": "One paragraph summary of analysis and recommendations"
}}
```
"""


def reflector_node(state: SintraState) -> dict[str, Any]:
    """Self-reflection node that analyzes failures and updates strategy.

    This node:
    1. Examines recent history for failures
    2. Identifies patterns and root causes
    3. Provides strategic guidance for next iteration

    Args:
        state: Current workflow state

    Returns:
        State update with reflection analysis
    """
    history = state.get("history", [])

    # Skip reflection if no history or in debug mode
    if not history or state.get("use_debug"):
        return {"reflection": None}

    log_transition("Reflector", "Analyzing recent attempts...", "critic.node")

    # Analyze last 3 attempts (or fewer if not available)
    recent_history = history[-3:]
    profile = state["profile"]

    # Check if last attempt was successful
    last_metrics = recent_history[-1]["metrics"]
    if _all_targets_met(last_metrics, profile):
        log_transition(
            "Reflector",
            "Last attempt successful, no reflection needed",
            "status.success",
        )
        return {"reflection": None}

    # Perform analysis
    reflection = _analyze_history(recent_history, profile, state)

    log_transition(
        "Reflector",
        f"Analysis: {len(reflection.failures)} failures, {len(reflection.adjustments)} recommendations",
        "critic.node",
    )

    return {
        "reflection": reflection,
        "strategy_adjustments": reflection.adjustments,
    }


def reflector_node_llm(state: SintraState) -> dict[str, Any]:
    """LLM-powered reflector node for deeper analysis.

    Uses an LLM to provide more nuanced failure analysis and
    strategic recommendations.

    Args:
        state: Current workflow state

    Returns:
        State update with reflection analysis
    """
    history = state.get("history", [])

    if not history or state.get("use_debug"):
        return {"reflection": None}

    # Check if last attempt was successful
    profile = state["profile"]
    last_metrics = history[-1]["metrics"]
    if _all_targets_met(last_metrics, profile):
        return {"reflection": None}

    log_transition("Reflector", "[LLM] Deep analysis of failures...", "critic.node")

    # Prepare context for LLM
    recent_history = history[-3:]
    attempts_summary = _format_attempts_for_llm(recent_history, profile)

    system_prompt = REFLECTOR_SYSTEM_PROMPT.format(
        target_tps=profile.targets.min_tokens_per_second,
        target_accuracy=profile.targets.min_accuracy_score,
        vram_gb=profile.constraints.vram_gb,
        num_attempts=len(recent_history),
        attempts_summary=attempts_summary,
    )

    try:
        llm = get_critic_llm(state["llm_config"])
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Analyze the attempts and provide recommendations.",
                },
            ]
        )

        reflection = _parse_llm_reflection(response.content, len(history))

    except Exception as e:
        logger.warning(f"LLM reflection failed: {e}, using rule-based analysis")
        reflection = _analyze_history(recent_history, profile, state)

    return {
        "reflection": reflection,
        "strategy_adjustments": reflection.adjustments,
    }


def _all_targets_met(metrics, profile) -> bool:
    """Check if all optimization targets are met."""
    if not metrics.was_successful:
        return False
    return (
        metrics.actual_tps >= profile.targets.min_tokens_per_second
        and metrics.accuracy_score >= profile.targets.min_accuracy_score
        and metrics.actual_vram_usage <= profile.constraints.vram_gb
    )


def _analyze_history(history: list[dict], profile, state: SintraState) -> Reflection:
    """Rule-based analysis of recent history."""
    failures = []
    patterns = []
    adjustments = []

    targets = profile.targets
    constraints = profile.constraints

    # Analyze each attempt
    speed_failures = 0
    accuracy_failures = 0

    for entry in history:
        metrics = entry["metrics"]
        recipe = entry["recipe"]

        if not metrics.was_successful:
            failures.append(
                FailureAnalysis(
                    failure_type="crash",
                    root_cause=metrics.error_log or "Unknown error",
                    severity="high",
                )
            )
            continue

        # Check speed
        if metrics.actual_tps < targets.min_tokens_per_second:
            speed_failures += 1
            gap = targets.min_tokens_per_second - metrics.actual_tps
            severity = "high" if gap > 10 else "medium" if gap > 5 else "low"
            failures.append(
                FailureAnalysis(
                    failure_type="speed",
                    root_cause=f"TPS {metrics.actual_tps:.1f} below target {targets.min_tokens_per_second}",
                    severity=severity,
                )
            )

        # Check accuracy
        if metrics.accuracy_score < targets.min_accuracy_score:
            accuracy_failures += 1
            gap = targets.min_accuracy_score - metrics.accuracy_score
            severity = "high" if gap > 0.15 else "medium" if gap > 0.05 else "low"
            failures.append(
                FailureAnalysis(
                    failure_type="accuracy",
                    root_cause=f"Accuracy {metrics.accuracy_score:.2f} below target {targets.min_accuracy_score}",
                    severity=severity,
                )
            )

        # Check memory
        if metrics.actual_vram_usage > constraints.vram_gb:
            failures.append(
                FailureAnalysis(
                    failure_type="memory",
                    root_cause=f"VRAM {metrics.actual_vram_usage:.1f}GB exceeds limit {constraints.vram_gb}GB",
                    severity="high",
                )
            )

    # Identify patterns
    if speed_failures > 0 and accuracy_failures > 0:
        patterns.append("Oscillating between speed and accuracy failures")
    elif speed_failures > 1:
        patterns.append("Consistent speed failures - model too slow")
    elif accuracy_failures > 1:
        patterns.append("Consistent accuracy failures - too aggressive compression")

    # Check for repeated recipes
    recipes = [entry["recipe"] for entry in history]
    if len(recipes) >= 2 and recipes[-1].bits == recipes[-2].bits:
        patterns.append("Same bit width tried multiple times")

    # Generate adjustments
    if speed_failures > accuracy_failures:
        adjustments.append(
            StrategyAdjustment(
                parameter="bits",
                direction="decrease",
                magnitude="medium" if speed_failures > 1 else "small",
                reasoning="Speed failures dominate - need more aggressive quantization",
            )
        )
        adjustments.append(
            StrategyAdjustment(
                parameter="pruning",
                direction="increase",
                magnitude="small",
                reasoning="Adding pruning can boost speed with less accuracy impact than quantization",
            )
        )
    elif accuracy_failures > speed_failures:
        adjustments.append(
            StrategyAdjustment(
                parameter="bits",
                direction="increase",
                magnitude="medium" if accuracy_failures > 1 else "small",
                reasoning="Accuracy failures dominate - need gentler quantization",
            )
        )
        adjustments.append(
            StrategyAdjustment(
                parameter="pruning",
                direction="decrease",
                magnitude="small",
                reasoning="Reduce pruning to preserve model quality",
            )
        )
    elif speed_failures > 0 and accuracy_failures > 0:
        adjustments.append(
            StrategyAdjustment(
                parameter="layers",
                direction="increase",
                magnitude="small",
                reasoning="Try layer dropping as alternative to quantization/pruning trade-off",
            )
        )

    # Build summary
    summary_parts = []
    if failures:
        summary_parts.append(f"Found {len(failures)} failure(s) in recent attempts.")
    if patterns:
        summary_parts.append(f"Patterns: {', '.join(patterns)}.")
    if adjustments:
        summary_parts.append(
            f"Recommend: {adjustments[0].direction} {adjustments[0].parameter} ({adjustments[0].reasoning})"
        )

    return Reflection(
        iteration_analyzed=len(history),
        failures=failures,
        patterns=patterns,
        adjustments=adjustments,
        confidence=0.7 if len(history) >= 3 else 0.5,
        summary=" ".join(summary_parts)
        if summary_parts
        else "Insufficient data for analysis.",
    )


def _format_attempts_for_llm(history: list[dict], profile) -> str:
    """Format attempt history for LLM consumption."""
    lines = []
    targets = profile.targets

    for i, entry in enumerate(history, 1):
        metrics = entry["metrics"]
        recipe = entry["recipe"]

        status = "✓" if metrics.was_successful else "✗"

        # Calculate gaps
        tps_gap = metrics.actual_tps - targets.min_tokens_per_second
        acc_gap = metrics.accuracy_score - targets.min_accuracy_score

        line = (
            f"Attempt {i} [{status}]:\n"
            f"  Recipe: {recipe.bits}-bit, {recipe.pruning_ratio:.0%} pruning, "
            f"{len(recipe.layers_to_drop)} layers dropped\n"
            f"  Results: TPS={metrics.actual_tps:.1f} ({'+' if tps_gap >= 0 else ''}{tps_gap:.1f}), "
            f"Accuracy={metrics.accuracy_score:.2f} ({'+' if acc_gap >= 0 else ''}{acc_gap:.2f})"
        )

        if not metrics.was_successful:
            line += f"\n  Error: {metrics.error_log}"

        lines.append(line)

    return "\n\n".join(lines)


def _parse_llm_reflection(content: str, iteration: int) -> Reflection:
    """Parse LLM response into Reflection object."""
    import json
    import re

    try:
        # Find JSON in response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            data = json.loads(json_match.group())

            return Reflection(
                iteration_analyzed=data.get("iteration_analyzed", iteration),
                failures=[FailureAnalysis(**f) for f in data.get("failures", [])],
                patterns=data.get("patterns", []),
                adjustments=[
                    StrategyAdjustment(**a) for a in data.get("adjustments", [])
                ],
                confidence=data.get("confidence", 0.5),
                summary=data.get("summary", ""),
            )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse LLM reflection: {e}")

    # Fallback
    return Reflection(
        iteration_analyzed=iteration,
        summary="LLM analysis could not be parsed.",
        confidence=0.3,
    )
