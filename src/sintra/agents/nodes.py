"""Agent workflow nodes for the Sintra optimization loop."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

from sintra.agents.factory import get_architect_llm
from sintra.benchmarks.executor import MockExecutor, StandaloneExecutor
from sintra.persistence import format_history_from_db, get_history_db
from sintra.profiles.models import ModelRecipe
from sintra.ui.console import log_transition

from .state import SintraState
from .utils import format_history_for_llm, get_untried_variations, is_duplicate_recipe

logger = logging.getLogger(__name__)

# Configuration constants
MAX_ITERATIONS = 10
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "optimized_recipe.json"

# Type alias for state updates
StateUpdate = dict[str, Any]


class LLMConnectionError(Exception):
    """Raised when the LLM service is unavailable or connection fails."""

    pass


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

    # Get historical context from database (across all runs)
    db_history = format_history_from_db(
        state["target_model_id"],
        profile.name,
        limit=10,
    )

    history = state.get("history", [])

    # Build past attempts summary (single pass \u2014 reused in both prompt and user message)
    past_attempts = format_history_for_llm(history)

    # Get suggestions for untried combinations
    variations = get_untried_variations(history) if history else {}
    untried_hint = ""
    if variations:
        if variations.get("untried_bits"):
            untried_hint += f"\n- Untried bit widths: {variations['untried_bits']}"
        if variations.get("untried_pruning"):
            untried_hint += (
                f"\n- Untried pruning ratios: {variations['untried_pruning']}"
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
        - "bits": integer (2, 3, 4, 5, 6, or 8)
        - "pruning_ratio": decimal between 0.0 and 1.0 (e.g., 0.25)
        - "layers_to_drop": list of layer indices (0-indexed) OR an empty list
    - Never output whole numbers like 20 or 50 for pruning. Only decimals.

    ====================================================
    CONSTRAINTS
    ====================================================
    - VRAM Limit: {profile.constraints.vram_gb} GB
    - Target TPS (tokens/sec): {profile.targets.min_tokens_per_second}
    - Target Accuracy Score: {profile.targets.min_accuracy_score}

    ====================================================
    COMPRESSION TECHNIQUES (Now Fully Implemented!)
    ====================================================
    
    **1. QUANTIZATION (bits):**
    - Controls numerical precision of weights
    - Lower bits = smaller model, faster inference, lower accuracy
    - Supported: 2, 3, 4, 5, 6, 8 bits
    - Impact: ~12-15% size reduction per bit level
    - Recommended: Start with 4-bit (Q4_K_M), good balance
    
    **2. STRUCTURED PRUNING (pruning_ratio):**
    - Zeros out smallest-magnitude weights in attention/MLP layers
    - Range: 0.0 (no pruning) to 0.5 (aggressive)
    - Impact on TPS: +5-15% per 0.1 pruning ratio
    - Impact on accuracy: -2-5% per 0.1 pruning ratio
    - Guidelines:
        - 0.0-0.1: Safe, minimal quality loss
        - 0.1-0.2: Noticeable speedup, slight quality loss
        - 0.2-0.3: Significant speedup, moderate quality loss
        - >0.3: Aggressive, may hurt accuracy significantly
    
    **3. LAYER DROPPING (layers_to_drop):**
    - Removes entire transformer layers from the model
    - Most aggressive compression - removes ~1/N of model per layer
    - Impact: Each dropped layer reduces params by ~1/total_layers
    - Guidelines:
        - Prefer dropping middle layers (less critical than first/last)
        - Dropping 10-20% of layers: Minimal quality loss
        - Dropping 20-40%: Noticeable degradation
        - Dropping >40%: Significant quality loss
    - Example for 32-layer model: layers_to_drop=[10, 11, 12, 20, 21, 22]

    ====================================================
    OPTIMIZATION STRATEGY
    ====================================================
    1. Start conservative: bits=4, pruning_ratio=0.0, layers_to_drop=[]
    2. If TPS too low:
        - First: Reduce bits (4→3→2)
        - Second: Add pruning (0.1→0.2→0.3)
        - Third: Drop layers (start with middle layers)
    3. If accuracy too low:
        - Increase bits (4→5→6→8)
        - Reduce pruning ratio
        - Remove fewer layers
    4. Combine techniques for fine-grained control

    ====================================================
    PAST ATTEMPTS THIS RUN (NEVER REPEAT THESE)
    ====================================================
    {past_attempts}
    {untried_hint}

    ====================================================
    HISTORICAL INSIGHTS (FROM PREVIOUS RUNS)
    ====================================================
    {db_history}

    CRITICAL: You MUST propose a DIFFERENT recipe than all past attempts.
    - Change AT LEAST ONE of: bits, pruning_ratio, or layers_to_drop
    - If bits=4 and pruning=0.2 was tried, try bits=3 or pruning=0.3
    - Prefer untried combinations listed above
    - Small changes (e.g., pruning 0.20 → 0.21) count as duplicates!

    ====================================================
    REASONING RULES (INTERNAL ONLY)
    ====================================================
    - Think step-by-step internally, but output ONLY the final JSON.
    - Internally evaluate:
        - VRAM feasibility
        - Expected TPS (consider all three techniques)
        - Expected accuracy impact (cumulative from all techniques)
        - Differences from past failures
    - Never reveal your reasoning or internal thoughts.

    ====================================================
    FINAL INSTRUCTION
    ====================================================
    Respond with ONLY the JSON object representing the new ModelRecipe.

    """

    try:
        new_recipe = brain.invoke(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"History:\n{past_attempts}\nPropose next recipe.",
                },
            ]
        )
    except Exception as e:
        # Check for connection-related errors
        error_str = str(e).lower()
        if any(
            term in error_str
            for term in ["connection", "refused", "timeout", "unreachable", "connect"]
        ):
            provider = state["llm_config"].provider.value
            raise LLMConnectionError(
                f"Cannot connect to {provider} LLM service. Original error: {e}"
            ) from e
        # Check for API key errors
        if any(
            term in error_str
            for term in ["api_key", "api key", "authentication", "unauthorized", "401"]
        ):
            provider = state["llm_config"].provider.value
            raise LLMConnectionError(
                f"Authentication failed for {provider}. Check your API key. "
                f"Original error: {e}"
            ) from e
        # Check for rate limiting
        if any(
            term in error_str
            for term in ["rate limit", "rate_limit", "429", "too many requests"]
        ):
            raise LLMConnectionError(
                f"Rate limited by LLM provider. Please wait and try again. "
                f"Original error: {e}"
            ) from e
        # Re-raise other exceptions as-is
        raise

    # Validate we got a proper recipe back
    if new_recipe is None:
        log_transition(
            "Architect",
            "LLM returned empty response, using fallback recipe",
            "status.warn",
        )
        new_recipe = ModelRecipe(
            bits=4, pruning_ratio=0.1, layers_to_drop=[], method="GGUF"
        )

    # DUPLICATE DETECTION: Prevent running the same experiment twice
    max_retries = 3
    retry_count = 0

    while is_duplicate_recipe(new_recipe, history) and retry_count < max_retries:
        retry_count += 1
        log_transition(
            "Architect",
            f"Duplicate recipe detected! Auto-modifying (attempt {retry_count}/{max_retries})",
            "status.warn",
        )

        # Programmatically modify the recipe to make it unique
        # Strategy: cycle through bits, then pruning ratio
        bit_options = [2, 3, 4, 5, 6, 8]
        current_bit_idx = (
            bit_options.index(new_recipe.bits) if new_recipe.bits in bit_options else 2
        )

        # Try next bit width first
        new_bits = bit_options[(current_bit_idx + retry_count) % len(bit_options)]

        # Also adjust pruning ratio
        new_pruning = round(min(0.5, new_recipe.pruning_ratio + 0.1 * retry_count), 2)

        new_recipe = ModelRecipe(
            bits=new_bits,
            pruning_ratio=new_pruning,
            layers_to_drop=new_recipe.layers_to_drop,
            method=new_recipe.method,
        )
        log_transition(
            "Architect",
            f"Modified to: {new_recipe.bits}-bit, {new_recipe.pruning_ratio:.0%} pruning",
            "arch.node",
        )

    if retry_count > 0 and is_duplicate_recipe(new_recipe, history):
        log_transition(
            "Architect",
            "Could not find unique recipe after retries. Forcing convergence.",
            "status.warn",
        )
        return {
            "current_recipe": new_recipe,
            "iteration": state.get("iteration", 0) + 1,
            "is_converged": True,
            "critic_feedback": "Search space exhausted - no new recipes to try.",
        }

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
    # Use Mock if in debug mode or explicitly requested
    if state.get("use_debug") or state.get("use_mock"):
        executor = MockExecutor()
    else:
        executor = StandaloneExecutor()
    result = executor.run_benchmark(recipe, profile)

    # Save experiment to database for cross-run learning
    run_id = state.get("run_id")
    if run_id:
        try:
            db = get_history_db()
            db.save_experiment(
                run_id=run_id,
                model_id=state["target_model_id"],
                hardware_name=profile.name,
                recipe=recipe,
                result=result,
                backend=state.get("backend", "gguf"),
            )
        except Exception as e:
            log_transition(
                "Lab",
                f"Warning: Failed to save experiment to database: {e}",
                "status.warn",
            )

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
            if (
                metrics.actual_tps > best_metrics.actual_tps
                and metrics.accuracy_score >= targets.min_accuracy_score
            ):
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


# ============================================================================
# LLM-Based Routing (Alternative to rule-based critic_router)
# ============================================================================


from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    """LLM's decision about whether to continue or stop optimization."""

    decision: Literal["continue", "stop"] = Field(
        description="Whether to continue optimization or stop"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in this decision (0-1)"
    )
    reasoning: str = Field(
        description="Brief explanation of why this decision was made"
    )
    suggestion: str = Field(
        default="", description="Optional suggestion for the architect if continuing"
    )


CRITIC_ROUTER_PROMPT = """You are the Critic agent in an LLM compression optimization system.

Your job is to decide whether to:
1. **CONTINUE** optimization (route to architect) - if targets aren't met or we can do better
2. **STOP** optimization (route to reporter) - if targets are met or further improvement is unlikely

## Current Targets
- Minimum TPS: {target_tps}
- Minimum Accuracy: {target_accuracy}
- Maximum VRAM: {vram_limit} GB

## Current Iteration: {iteration} / {max_iterations}

## Latest Result
- Recipe: {recipe_bits}-bit, {recipe_pruning:.0%} pruning, {layers_dropped} layers dropped
- Actual TPS: {actual_tps} (target: {target_tps})
- Actual Accuracy: {actual_accuracy} (target: {target_accuracy})
- Actual VRAM: {actual_vram} GB (limit: {vram_limit} GB)
- Status: {status}

## Recent History
{history_summary}

## Decision Guidelines
- If ALL targets are met → STOP (we succeeded!)
- If we're at max iterations → STOP (accept best result)
- If we've tried many similar recipes without improvement → STOP (likely stuck)
- If there's clear room for improvement → CONTINUE
- If the last attempt crashed → CONTINUE (try different approach)

Respond with JSON:
```json
{{
    "decision": "continue" or "stop",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "suggestion": "Hint for architect if continuing"
}}
```
"""


def critic_router_llm(state: SintraState) -> str:
    """
    LLM-based routing decision for more nuanced optimization control.

    Uses an LLM to decide whether to continue optimization or stop,
    allowing for more sophisticated reasoning about trade-offs.

    Falls back to rule-based routing if LLM fails.
    """
    from sintra.agents.factory import get_critic_llm

    # Quick exits that don't need LLM
    if state.get("use_debug") or state.get("is_converged"):
        return "reporter"

    if state.get("iteration", 0) >= MAX_ITERATIONS:
        log_transition(
            "Critic",
            f"[LLM] Max iterations ({MAX_ITERATIONS}) reached.",
            "status.warn",
        )
        return "reporter"

    if not state["history"]:
        return "architect"

    # Prepare context for LLM
    last_experiment = state["history"][-1]
    metrics = last_experiment["metrics"]
    recipe = last_experiment["recipe"]
    profile = state["profile"]

    # Format history summary
    history_lines = []
    for i, entry in enumerate(state["history"][-5:], 1):  # Last 5 attempts
        m = entry["metrics"]
        r = entry["recipe"]
        status = "✓" if m.was_successful else "✗"
        history_lines.append(
            f"  {i}. [{status}] {r.bits}-bit, {r.pruning_ratio:.0%} prune → "
            f"TPS={m.actual_tps:.1f}, Acc={m.accuracy_score:.2f}"
        )

    prompt = CRITIC_ROUTER_PROMPT.format(
        target_tps=profile.targets.min_tokens_per_second,
        target_accuracy=profile.targets.min_accuracy_score,
        vram_limit=profile.constraints.vram_gb,
        iteration=state.get("iteration", 0),
        max_iterations=MAX_ITERATIONS,
        recipe_bits=recipe.bits,
        recipe_pruning=recipe.pruning_ratio,
        layers_dropped=len(recipe.layers_to_drop),
        actual_tps=metrics.actual_tps,
        actual_accuracy=metrics.accuracy_score,
        actual_vram=metrics.actual_vram_usage,
        status="SUCCESS" if metrics.was_successful else f"FAILED: {metrics.error_log}",
        history_summary="\n".join(history_lines)
        if history_lines
        else "No previous attempts",
    )

    try:
        log_transition(
            "Critic", "[LLM] Evaluating optimization progress...", "critic.node"
        )

        llm = get_critic_llm(state["llm_config"])
        structured_llm = llm.with_structured_output(RoutingDecision)

        decision = structured_llm.invoke(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Should we continue optimization or stop?"},
            ]
        )

        if decision.decision == "stop":
            log_transition(
                "Critic",
                f"[LLM] STOP ({decision.confidence:.0%} confident): {decision.reasoning}",
                "status.success" if decision.confidence > 0.7 else "status.warn",
            )
            return "reporter"
        else:
            log_transition(
                "Critic",
                f"[LLM] CONTINUE ({decision.confidence:.0%} confident): {decision.reasoning}",
                "critic.node",
            )
            if decision.suggestion:
                log_transition(
                    "Critic", f"[LLM] Suggestion: {decision.suggestion}", "critic.node"
                )
            return "architect"

    except Exception as e:
        log_transition(
            "Critic",
            f"[LLM] Routing failed: {e}. Falling back to rules.",
            "status.warn",
        )
        # Fallback to rule-based routing
        return critic_router(state)


def reporter_node(state: SintraState) -> dict:
    """
    The Archivist: Saves the winning recipe to a JSON file using Pydantic v2 standards.
    Also displays baseline accuracy comparison if available.
    """
    log_transition("Reporter", "Archiving the final 'Golden Recipe'...", "hw.profile")

    if not state["history"]:
        log_transition("Reporter", "Error: No history found to report.", "status.fail")
        return state

    last_entry = state["history"][-1]

    # Get recipe and metrics
    recipe_data = (
        last_entry["recipe"].model_dump()
        if hasattr(last_entry["recipe"], "model_dump")
        else last_entry["recipe"]
    )
    metrics_data = (
        last_entry["metrics"].model_dump()
        if hasattr(last_entry["metrics"], "model_dump")
        else last_entry["metrics"]
    )

    output = {
        "hardware_profile": state["profile"].name,
        "recipe": recipe_data,
        "performance": metrics_data,
    }

    # Display baseline accuracy comparison if available
    from sintra.ui.console import console

    if metrics_data.get("accuracy_retention") is not None:
        retention = metrics_data["accuracy_retention"] * 100
        accuracy_loss = metrics_data.get("accuracy_loss", 0) * 100

        console.print("\n[bold cyan]📊 Baseline Accuracy Comparison[/bold cyan]")
        console.print(f"  Optimized Accuracy: {metrics_data['accuracy_score']:.2%}")
        console.print(f"  Accuracy Retention: [green]{retention:.1f}%[/green]")
        if accuracy_loss > 0:
            console.print(f"  Accuracy Loss:      [yellow]{accuracy_loss:.1f}%[/yellow]")
        console.print()

        output["baseline_comparison"] = {
            "retention_rate": metrics_data["accuracy_retention"],
            "accuracy_loss": metrics_data.get("accuracy_loss", 0),
        }

    try:
        output_dir = Path(os.environ.get("SINTRA_OUTPUT_DIR", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "optimized_recipe.json"

        with open(output_file, "w") as f:
            json.dump(output, f, indent=4)
        log_transition(
            "Reporter", f"Recipe saved to {output_file}", "status.success"
        )

        # Handle Ollama export if requested
        ollama_model_name = os.environ.get("SINTRA_EXPORT_OLLAMA")
        if ollama_model_name:
            _export_to_ollama(output_dir, ollama_model_name)

    except OSError as e:
        log_transition(
            "Reporter",
            f"Warning: Could not save to file: {e}. Printing to console instead.",
            "status.warn",
        )
        # Fallback: print to console
        console.print_json(data=output)

    return state


def _export_to_ollama(output_dir: Path, model_name: str) -> None:
    """Export the optimized model to Ollama.

    Args:
        output_dir: Directory containing the optimized GGUF model
        model_name: Name for the Ollama model
    """
    from sintra.ui.console import console

    try:
        from sintra.compression.ollama_exporter import (
            OllamaExportError,
            OllamaExporter,
        )

        # Find the GGUF model in output or cache directory
        gguf_files = list(output_dir.glob("*.gguf"))

        # Also check the cache directory
        cache_dir = Path.home() / ".cache" / "sintra" / "quantized"
        if cache_dir.exists():
            gguf_files.extend(cache_dir.glob("*.gguf"))

        if not gguf_files:
            log_transition(
                "Reporter",
                "No GGUF model found for Ollama export",
                "status.warn",
            )
            return

        # Use the most recently modified GGUF file
        gguf_path = max(gguf_files, key=lambda p: p.stat().st_mtime)

        console.print(f"\n[bold cyan]🦙 Exporting to Ollama...[/bold cyan]")
        console.print(f"  Model: {gguf_path.name}")
        console.print(f"  Name:  {model_name}")

        exporter = OllamaExporter()
        system_prompt = os.environ.get("SINTRA_OLLAMA_SYSTEM_PROMPT")

        result = exporter.export(
            gguf_path,
            model_name,
            system_prompt=system_prompt,
            force=True,
        )

        if result.success:
            console.print(f"\n[green]✓ {result.message}[/green]")
            console.print(f"  Run: [cyan]ollama run {model_name}[/cyan]")
        else:
            console.print(f"\n[red]✗ Ollama export failed: {result.message}[/red]")

    except ImportError:
        log_transition(
            "Reporter",
            "Ollama exporter not available",
            "status.warn",
        )
    except OllamaExportError as e:
        console.print(f"\n[red]✗ Ollama export error: {e}[/red]")
    except Exception as e:
        logger.warning(f"Ollama export failed: {e}")
        console.print(f"\n[yellow]Warning: Ollama export failed: {e}[/yellow]")
