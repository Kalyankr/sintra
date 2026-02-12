"""Multi-agent collaboration with specialized expert agents.

This module implements specialized expert agents that can be consulted
by the architect for domain-specific advice on:
- Quantization strategies
- Pruning techniques
- Integration of multiple compression methods

The coordinator aggregates expert opinions into a unified recommendation.
"""

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from sintra.agents.factory import get_critic_llm
from sintra.agents.state import SintraState
from sintra.agents.tools import (
    get_model_architecture,
)
from sintra.persistence import format_history_from_db
from sintra.ui.console import log_transition

logger = logging.getLogger(__name__)


# ============================================================================
# Expert Data Models
# ============================================================================


class ExpertOpinion(BaseModel):
    """A single expert's recommendation."""

    expert_name: str = Field(description="Name of the expert agent")
    domain: str = Field(
        description="Expert domain: 'quantization', 'pruning', 'integration'"
    )
    recommendation: str = Field(description="The expert's recommendation")
    suggested_bits: int | None = Field(default=None, description="Suggested bit width")
    suggested_pruning: float | None = Field(
        default=None, description="Suggested pruning ratio"
    )
    suggested_layers_to_drop: list[int] | None = Field(
        default=None, description="Suggested layers to drop"
    )
    confidence: float = Field(
        default=0.5, description="Confidence in this recommendation (0-1)"
    )
    reasoning: str = Field(default="", description="Detailed reasoning")
    risk_assessment: str = Field(
        default="medium", description="Risk level: 'low', 'medium', 'high'"
    )


class ExpertConsensus(BaseModel):
    """Aggregated consensus from all expert agents."""

    opinions: list[ExpertOpinion] = Field(
        default_factory=list, description="Individual expert opinions"
    )
    consensus_bits: int = Field(default=4, description="Agreed bit width")
    consensus_pruning: float = Field(default=0.0, description="Agreed pruning ratio")
    consensus_layers_to_drop: list[int] = Field(
        default_factory=list, description="Agreed layers to drop"
    )
    agreement_level: float = Field(
        default=0.5, description="How much experts agree (0-1)"
    )
    summary: str = Field(default="", description="Summary of expert discussion")
    strategy_notes: str = Field(default="", description="Additional strategic guidance")


# ============================================================================
# Expert System Prompts
# ============================================================================


QUANTIZATION_EXPERT_PROMPT = """You are a **Quantization Expert** specializing in LLM weight quantization.

You deeply understand:
- GGUF quantization types (Q2_K, Q3_K_S, Q3_K_M, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0)
- Tradeoffs between bit width and quality for different model architectures
- How quantization interacts with model size and attention head count
- BitsAndBytes NF4 vs FP4 vs INT8 quantization
- ONNX INT8 dynamic vs static quantization

## Context
- Model: {model_id} (~{estimated_params})
- Hardware: {hardware_name} ({vram_gb}GB memory)
- Target TPS: {target_tps}
- Target Accuracy: {target_accuracy}
- Backend: {backend}

## Previous Attempts
{history}

## Your Task
Recommend the BEST quantization bit width for this scenario.
Consider:
1. Model architecture sensitivity to quantization
2. Hardware memory constraints
3. Speed vs quality tradeoff
4. What has and hasn't worked before

Respond with JSON:
```json
{{
    "suggested_bits": <int>,
    "confidence": <0-1>,
    "reasoning": "detailed explanation",
    "risk_assessment": "low|medium|high",
    "alternative": <int or null>
}}
```"""


PRUNING_EXPERT_PROMPT = """You are a **Pruning Expert** specializing in neural network pruning.

You deeply understand:
- Structured vs unstructured pruning tradeoffs
- Which layers in transformers are safe to prune
- How pruning ratio affects different model sizes
- The interaction between pruning and quantization
- Layer importance and sensitivity analysis

## Context
- Model: {model_id} (~{estimated_params}, {num_layers} layers)
- Hardware: {hardware_name} ({vram_gb}GB memory)
- Target TPS: {target_tps}
- Target Accuracy: {target_accuracy}
- Planned Quantization: {planned_bits}-bit

## Previous Attempts
{history}

## Your Task
Recommend the optimal pruning configuration:
1. Pruning ratio (0.0 to 0.5)
2. Which layers to drop (if any)
3. How pruning should complement the quantization plan

Respond with JSON:
```json
{{
    "suggested_pruning": <float>,
    "suggested_layers_to_drop": [<int>, ...],
    "confidence": <0-1>,
    "reasoning": "detailed explanation",
    "risk_assessment": "low|medium|high",
    "max_safe_pruning": <float>
}}
```"""


INTEGRATION_EXPERT_PROMPT = """You are an **Integration Expert** specializing in combining compression methods.

You deeply understand:
- How quantization and pruning interact (synergies and conflicts)
- Optimal ordering of compression operations
- Edge device deployment constraints
- Production readiness assessment
- When to stop optimizing

## Context
- Model: {model_id} (~{estimated_params})
- Hardware: {hardware_name} ({vram_gb}GB memory)
- Target TPS: {target_tps}
- Target Accuracy: {target_accuracy}
- Quantization Expert suggests: {quant_bits}-bit (confidence: {quant_confidence})
- Pruning Expert suggests: {prune_ratio:.0%} pruning, drop layers {prune_layers} (confidence: {prune_confidence})

## Previous Attempts
{history}

## Your Task
Evaluate the combined compression strategy and recommend final settings.
Consider:
1. Do the quantization and pruning recommendations complement each other?
2. Is the combined compression too aggressive or not aggressive enough?
3. What's the risk of this configuration failing?
4. Should we adjust either recommendation?

Respond with JSON:
```json
{{
    "final_bits": <int>,
    "final_pruning": <float>,
    "final_layers_to_drop": [<int>, ...],
    "confidence": <0-1>,
    "reasoning": "detailed explanation",
    "risk_assessment": "low|medium|high",
    "production_ready": <bool>,
    "suggestion": "any additional advice"
}}
```"""


# ============================================================================
# Expert Agent Functions
# ============================================================================


def _get_model_info(model_id: str) -> dict[str, Any]:
    """Get model architecture info for experts."""
    try:
        info = get_model_architecture.invoke({"model_id": model_id})
        return info
    except Exception:
        return {
            "num_layers": 32,
            "num_parameters_billions": 7,
            "estimated": True,
        }


def consult_quantization_expert(
    state: SintraState,
    use_llm: bool = True,
) -> ExpertOpinion:
    """Consult the quantization expert for bit width recommendation.

    Args:
        state: Current workflow state
        use_llm: Whether to use LLM or rule-based logic

    Returns:
        ExpertOpinion with quantization recommendation
    """
    profile = state["profile"]
    history = state.get("history", [])

    if use_llm and not state.get("use_debug"):
        try:
            return _llm_quantization_expert(state)
        except Exception as e:
            logger.warning(f"LLM quantization expert failed: {e}")

    # Rule-based fallback
    vram_gb = profile.constraints.vram_gb
    target_tps = profile.targets.min_tokens_per_second
    target_accuracy = profile.targets.min_accuracy_score

    # Analyze history for what works
    successful_bits = []
    failed_bits = []
    for entry in history:
        metrics = entry["metrics"]
        recipe = entry["recipe"]
        if metrics.was_successful and metrics.accuracy_score >= target_accuracy:
            successful_bits.append(recipe.bits)
        elif not metrics.was_successful or metrics.accuracy_score < target_accuracy:
            failed_bits.append(recipe.bits)

    # Determine recommendation
    if successful_bits:
        # Use best successful bit width
        bits = min(successful_bits)  # Most compressed that still works
        confidence = 0.8
        reasoning = (
            f"Based on {len(successful_bits)} successful runs, {bits}-bit is reliable"
        )
    elif vram_gb < 4:
        bits = 4
        confidence = 0.7
        reasoning = "Limited VRAM requires 4-bit quantization"
    elif target_tps > 40:
        bits = 3
        confidence = 0.6
        reasoning = "High TPS target suggests aggressive quantization"
    elif target_accuracy > 0.85:
        bits = 5
        confidence = 0.7
        reasoning = "High accuracy target favors conservative quantization"
    else:
        bits = 4
        confidence = 0.7
        reasoning = "4-bit quantization offers good balance for this hardware"

    # Avoid bits that failed before
    if bits in failed_bits and bits < 8:
        bits = min(bits + 1, 8)
        reasoning += f" (adjusted up from {bits - 1} due to previous failures)"
        confidence *= 0.8

    risk = "low" if bits >= 5 else "medium" if bits >= 3 else "high"

    return ExpertOpinion(
        expert_name="Quantization Expert",
        domain="quantization",
        recommendation=f"Use {bits}-bit quantization",
        suggested_bits=bits,
        confidence=confidence,
        reasoning=reasoning,
        risk_assessment=risk,
    )


def _llm_quantization_expert(state: SintraState) -> ExpertOpinion:
    """LLM-powered quantization expert."""
    profile = state["profile"]
    model_id = state["target_model_id"]
    model_info = _get_model_info(model_id)

    history_str = format_history_from_db(model_id, profile.name, limit=5)

    prompt = QUANTIZATION_EXPERT_PROMPT.format(
        model_id=model_id,
        estimated_params=f"{model_info.get('num_parameters_billions', '?')}B",
        hardware_name=profile.name,
        vram_gb=profile.constraints.vram_gb,
        target_tps=profile.targets.min_tokens_per_second,
        target_accuracy=profile.targets.min_accuracy_score,
        backend=state.get("backend", "gguf"),
        history=history_str or "No previous attempts.",
    )

    llm = get_critic_llm(state["llm_config"])
    response = llm.invoke(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Provide your quantization recommendation."},
        ]
    )

    data = _parse_json_response(response.content)

    return ExpertOpinion(
        expert_name="Quantization Expert",
        domain="quantization",
        recommendation=f"Use {data.get('suggested_bits', 4)}-bit quantization",
        suggested_bits=data.get("suggested_bits", 4),
        confidence=data.get("confidence", 0.5),
        reasoning=data.get("reasoning", ""),
        risk_assessment=data.get("risk_assessment", "medium"),
    )


def consult_pruning_expert(
    state: SintraState,
    planned_bits: int = 4,
    use_llm: bool = True,
) -> ExpertOpinion:
    """Consult the pruning expert for pruning & layer-drop recommendations.

    Args:
        state: Current workflow state
        planned_bits: Bit width recommended by quantization expert
        use_llm: Whether to use LLM or rule-based logic

    Returns:
        ExpertOpinion with pruning recommendation
    """
    profile = state["profile"]
    history = state.get("history", [])

    if use_llm and not state.get("use_debug"):
        try:
            return _llm_pruning_expert(state, planned_bits)
        except Exception as e:
            logger.warning(f"LLM pruning expert failed: {e}")

    # Rule-based fallback
    target_tps = profile.targets.min_tokens_per_second
    target_accuracy = profile.targets.min_accuracy_score

    # Analyze history
    speed_gap = 0
    accuracy_gap = 0
    for entry in history[-3:]:
        metrics = entry["metrics"]
        if metrics.was_successful:
            speed_gap = max(speed_gap, target_tps - metrics.actual_tps)
            accuracy_gap = max(accuracy_gap, target_accuracy - metrics.accuracy_score)

    # Determine pruning recommendation
    pruning_ratio = 0.0
    layers_to_drop: list[int] = []
    confidence = 0.6

    if speed_gap > 10:
        # Need significant speed boost
        pruning_ratio = 0.2
        confidence = 0.6
        reasoning = "Significant TPS gap requires moderate pruning"
    elif speed_gap > 5:
        pruning_ratio = 0.1
        reasoning = "Moderate TPS gap, light pruning should help"
    elif accuracy_gap > 0.1:
        # Accuracy is suffering, don't prune much
        pruning_ratio = 0.0
        reasoning = "Accuracy is already low, avoid additional pruning"
        confidence = 0.8
    else:
        pruning_ratio = 0.05
        reasoning = "Light pruning for small speed improvement without accuracy loss"

    # Consider layer dropping for aggressive cases
    model_info = _get_model_info(state["target_model_id"])
    num_layers = model_info.get("num_layers", 32)

    if speed_gap > 15 and planned_bits <= 4:
        # Very aggressive case: drop some middle layers
        mid = num_layers // 2
        layers_to_drop = list(range(mid - 1, mid + 2))  # Drop 3 middle layers
        reasoning += f" + dropping {len(layers_to_drop)} middle layers for extra speed"
        confidence *= 0.85

    risk = (
        "low" if pruning_ratio <= 0.1 else "medium" if pruning_ratio <= 0.25 else "high"
    )

    return ExpertOpinion(
        expert_name="Pruning Expert",
        domain="pruning",
        recommendation=f"{pruning_ratio:.0%} pruning, drop {len(layers_to_drop)} layers",
        suggested_pruning=pruning_ratio,
        suggested_layers_to_drop=layers_to_drop,
        confidence=confidence,
        reasoning=reasoning,
        risk_assessment=risk,
    )


def _llm_pruning_expert(state: SintraState, planned_bits: int) -> ExpertOpinion:
    """LLM-powered pruning expert."""
    profile = state["profile"]
    model_id = state["target_model_id"]
    model_info = _get_model_info(model_id)

    history_str = format_history_from_db(model_id, profile.name, limit=5)

    prompt = PRUNING_EXPERT_PROMPT.format(
        model_id=model_id,
        estimated_params=f"{model_info.get('num_parameters_billions', '?')}B",
        num_layers=model_info.get("num_layers", 32),
        hardware_name=profile.name,
        vram_gb=profile.constraints.vram_gb,
        target_tps=profile.targets.min_tokens_per_second,
        target_accuracy=profile.targets.min_accuracy_score,
        planned_bits=planned_bits,
        history=history_str or "No previous attempts.",
    )

    llm = get_critic_llm(state["llm_config"])
    response = llm.invoke(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Provide your pruning recommendation."},
        ]
    )

    data = _parse_json_response(response.content)

    return ExpertOpinion(
        expert_name="Pruning Expert",
        domain="pruning",
        recommendation=f"{data.get('suggested_pruning', 0.0):.0%} pruning",
        suggested_pruning=data.get("suggested_pruning", 0.0),
        suggested_layers_to_drop=data.get("suggested_layers_to_drop", []),
        confidence=data.get("confidence", 0.5),
        reasoning=data.get("reasoning", ""),
        risk_assessment=data.get("risk_assessment", "medium"),
    )


def consult_integration_expert(
    state: SintraState,
    quant_opinion: ExpertOpinion,
    prune_opinion: ExpertOpinion,
    use_llm: bool = True,
) -> ExpertOpinion:
    """Consult the integration expert to combine recommendations.

    Args:
        state: Current workflow state
        quant_opinion: Quantization expert's recommendation
        prune_opinion: Pruning expert's recommendation
        use_llm: Whether to use LLM or rule-based logic

    Returns:
        ExpertOpinion with integrated recommendation
    """
    if use_llm and not state.get("use_debug"):
        try:
            return _llm_integration_expert(state, quant_opinion, prune_opinion)
        except Exception as e:
            logger.warning(f"LLM integration expert failed: {e}")

    # Rule-based integration
    bits = quant_opinion.suggested_bits or 4
    pruning = prune_opinion.suggested_pruning or 0.0
    layers = prune_opinion.suggested_layers_to_drop or []

    # Check for conflicts
    total_compression = (8 - bits) * 0.1 + pruning + len(layers) * 0.05
    target_accuracy = state["profile"].targets.min_accuracy_score

    if total_compression > 0.5 and target_accuracy > 0.7:
        # Too aggressive for accuracy target
        if pruning > 0.1:
            pruning = max(0.0, pruning - 0.1)
        elif len(layers) > 0:
            layers = layers[: max(0, len(layers) - 1)]
        else:
            bits = min(bits + 1, 8)
        reasoning = "Reduced compression aggressiveness to protect accuracy"
        risk = "medium"
    elif total_compression < 0.2:
        reasoning = "Conservative settings should be safe"
        risk = "low"
    else:
        reasoning = "Balanced compression combining quantization and pruning"
        risk = "medium"

    confidence = min(quant_opinion.confidence, prune_opinion.confidence) * 0.9

    return ExpertOpinion(
        expert_name="Integration Expert",
        domain="integration",
        recommendation=f"Final: {bits}-bit, {pruning:.0%} pruning, {len(layers)} layers dropped",
        suggested_bits=bits,
        suggested_pruning=pruning,
        suggested_layers_to_drop=layers,
        confidence=confidence,
        reasoning=reasoning,
        risk_assessment=risk,
    )


def _llm_integration_expert(
    state: SintraState,
    quant_opinion: ExpertOpinion,
    prune_opinion: ExpertOpinion,
) -> ExpertOpinion:
    """LLM-powered integration expert."""
    profile = state["profile"]
    model_id = state["target_model_id"]
    model_info = _get_model_info(model_id)

    history_str = format_history_from_db(model_id, profile.name, limit=5)

    prompt = INTEGRATION_EXPERT_PROMPT.format(
        model_id=model_id,
        estimated_params=f"{model_info.get('num_parameters_billions', '?')}B",
        hardware_name=profile.name,
        vram_gb=profile.constraints.vram_gb,
        target_tps=profile.targets.min_tokens_per_second,
        target_accuracy=profile.targets.min_accuracy_score,
        quant_bits=quant_opinion.suggested_bits or 4,
        quant_confidence=quant_opinion.confidence,
        prune_ratio=prune_opinion.suggested_pruning or 0.0,
        prune_layers=prune_opinion.suggested_layers_to_drop or [],
        prune_confidence=prune_opinion.confidence,
        history=history_str or "No previous attempts.",
    )

    llm = get_critic_llm(state["llm_config"])
    response = llm.invoke(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Provide your integration assessment."},
        ]
    )

    data = _parse_json_response(response.content)

    return ExpertOpinion(
        expert_name="Integration Expert",
        domain="integration",
        recommendation=f"Final: {data.get('final_bits', 4)}-bit, "
        f"{data.get('final_pruning', 0.0):.0%} pruning",
        suggested_bits=data.get("final_bits", 4),
        suggested_pruning=data.get("final_pruning", 0.0),
        suggested_layers_to_drop=data.get("final_layers_to_drop", []),
        confidence=data.get("confidence", 0.5),
        reasoning=data.get("reasoning", ""),
        risk_assessment=data.get("risk_assessment", "medium"),
    )


# ============================================================================
# Coordinator: Multi-Agent Collaboration Node
# ============================================================================


def expert_collaboration_node(state: SintraState) -> dict[str, Any]:
    """Coordinate multiple expert agents to produce a consensus recommendation.

    This node:
    1. Consults quantization expert
    2. Consults pruning expert (informed by quant recommendation)
    3. Consults integration expert (informed by both)
    4. Aggregates into consensus

    Args:
        state: Current workflow state

    Returns:
        State update with expert consensus
    """
    if state.get("use_debug"):
        log_transition("Experts", "DEBUG: Skipping expert consultation", "arch.node")
        return {"expert_consensus": None}

    log_transition(
        "Experts",
        "Consulting specialist agents...",
        "arch.node",
    )

    use_llm = not state.get("use_debug") and not state.get("use_mock")

    # Step 1: Quantization Expert
    log_transition("Experts", "→ Quantization Expert analyzing...", "status.dim")
    quant_opinion = consult_quantization_expert(state, use_llm=use_llm)
    log_transition(
        "Experts",
        f"  Quant: {quant_opinion.suggested_bits}-bit "
        f"(confidence: {quant_opinion.confidence:.0%})",
        "status.dim",
    )

    # Step 2: Pruning Expert (informed by quant recommendation)
    log_transition("Experts", "→ Pruning Expert analyzing...", "status.dim")
    prune_opinion = consult_pruning_expert(
        state,
        planned_bits=quant_opinion.suggested_bits or 4,
        use_llm=use_llm,
    )
    log_transition(
        "Experts",
        f"  Prune: {prune_opinion.suggested_pruning:.0%} pruning "
        f"(confidence: {prune_opinion.confidence:.0%})",
        "status.dim",
    )

    # Step 3: Integration Expert (informed by both)
    log_transition("Experts", "→ Integration Expert evaluating...", "status.dim")
    integ_opinion = consult_integration_expert(
        state, quant_opinion, prune_opinion, use_llm=use_llm
    )
    log_transition(
        "Experts",
        f"  Final: {integ_opinion.suggested_bits}-bit, "
        f"{integ_opinion.suggested_pruning:.0%} pruning "
        f"(confidence: {integ_opinion.confidence:.0%})",
        "status.dim",
    )

    # Build consensus
    opinions = [quant_opinion, prune_opinion, integ_opinion]

    # Use integration expert's recommendation as final
    consensus = ExpertConsensus(
        opinions=opinions,
        consensus_bits=integ_opinion.suggested_bits or 4,
        consensus_pruning=integ_opinion.suggested_pruning or 0.0,
        consensus_layers_to_drop=integ_opinion.suggested_layers_to_drop or [],
        agreement_level=_calculate_agreement(opinions),
        summary=_build_consensus_summary(opinions),
        strategy_notes=integ_opinion.reasoning,
    )

    log_transition(
        "Experts",
        f"Consensus: {consensus.consensus_bits}-bit, "
        f"{consensus.consensus_pruning:.0%} pruning "
        f"(agreement: {consensus.agreement_level:.0%})",
        "arch.node",
    )

    return {"expert_consensus": consensus}


def _calculate_agreement(opinions: list[ExpertOpinion]) -> float:
    """Calculate how much experts agree with each other."""
    if len(opinions) < 2:
        return 1.0

    bits_values = [o.suggested_bits for o in opinions if o.suggested_bits is not None]
    pruning_values = [
        o.suggested_pruning for o in opinions if o.suggested_pruning is not None
    ]

    agreement = 1.0

    # Check bits agreement
    if bits_values and len(set(bits_values)) > 1:
        spread = max(bits_values) - min(bits_values)
        agreement -= spread * 0.1  # Penalize disagreement

    # Check pruning agreement
    if pruning_values and len(pruning_values) > 1:
        spread = max(pruning_values) - min(pruning_values)
        agreement -= spread * 0.5

    return max(0.0, min(1.0, agreement))


def _build_consensus_summary(opinions: list[ExpertOpinion]) -> str:
    """Build a human-readable summary of expert opinions."""
    parts = []
    for op in opinions:
        parts.append(
            f"{op.expert_name}: {op.recommendation} "
            f"(conf={op.confidence:.0%}, risk={op.risk_assessment})"
        )
    return " | ".join(parts)


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse JSON from an LLM response, handling markdown code blocks."""
    try:
        # Try direct parse
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code blocks or raw
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from expert response")
    return {}
