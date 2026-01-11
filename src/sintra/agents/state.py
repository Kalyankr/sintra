import operator
from typing import Annotated, Dict, List, Optional, TypedDict, Union

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

    # Current Work-in-Progress
    current_recipe: Optional[ModelRecipe]

    # History (Annotated with operator.add, results append automatically)
    history: Annotated[
        List[Dict[str, Union[ModelRecipe, ExperimentResult]]], operator.add
    ]
    # critic feedback
    critic_feedback: str
    # best recipe
    best_recipe: Optional[ModelRecipe]

    # Progress Tracking
    iteration: int
    is_converged: bool

    # debug mode
    use_debug: bool
