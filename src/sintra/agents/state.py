import operator
from typing import Annotated, Dict, List, Optional, TypedDict, Union

from pydantic import BaseModel, Field

from ..profiles.models import HardwareProfile


class ModelRecipe(BaseModel):
    """The 'Surgery' instructions proposed by the Architect."""

    bits: int = Field(4, ge=2, le=8, description="Bitwidth to quantize to.")
    pruning_ratio: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of layers to prune.",
    )
    layer_to_drop: List[int] = Field(
        default_factory=list,
        description="Indices of layers to prune.",
    )
    method: str = Field(
        "GGUF",  # or GPTQ/AWQ
        description="The type of surgery to perform.",
    )


class ExperimentResult(BaseModel):
    """The real-world telemetry from the Docker benchmark."""

    actual_tps: float
    actual_vram_usage: float
    accuracy_score: float
    was_sucessful: bool
    error_log: Optional[str] = None


class DistillerState(TypedDict):
    """The state of the optimization loop."""

    profile: HardwareProfile
    current_recipe: ModelRecipe

    # recursive history, append new results to this list
    history: Annotated[
        List[Dict[str, Union[ModelRecipe, ExperimentResult]]], operator.add
    ]

    # Metadata
    iteration: int
    is_converged: bool
