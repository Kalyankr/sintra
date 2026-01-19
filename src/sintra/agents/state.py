import operator
from typing import Annotated, Any, TypedDict

from sintra.profiles.models import (
    ExperimentResult,
    HardwareProfile,
    LLMConfig,
    ModelRecipe,
)


class SintraState(TypedDict):
    """The dynamic state passed between Architect, Lab, and Critic."""

    # Configuration
    profile: HardwareProfile
    llm_config: LLMConfig

    # Target model configuration
    target_model_id: str

    # Backend configuration
    backend: str

    # Run identification (for persistence)
    run_id: str

    # Current Work-in-Progress
    current_recipe: ModelRecipe | None

    # History (Annotated with operator.add, results append automatically)
    history: Annotated[
        list[dict[str, ModelRecipe | ExperimentResult]], operator.add
    ]
    # critic feedback
    critic_feedback: str
    # best recipe (dict with 'recipe' and 'metrics' keys)
    best_recipe: dict[str, ModelRecipe | ExperimentResult] | None

    # Progress Tracking
    iteration: int
    is_converged: bool

    # Execution modes
    use_debug: bool
    use_mock: bool

    # Agentic features (ReAct pattern)
    use_react: bool  # Whether to use ReAct-style architect
    reasoning_chain: list[Any] | None  # ReAct reasoning steps
    reasoning_summary: str | None  # Human-readable reasoning summary

    # Self-reflection
    reflection: Any | None  # Reflection analysis from reflector node
    strategy_adjustments: list[Any] | None  # Recommended strategy changes

    # Planner
    optimization_plan: Any | None  # OptimizationPlan from planner node
