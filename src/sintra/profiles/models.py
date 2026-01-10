from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# LLM Architect Config
class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.GOOGLE
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.2


# Hardware & Target Definitions
class Constraints(BaseModel):
    vram_gb: float


class Targets(BaseModel):
    min_tokens_per_second: float
    min_accuracy_score: float


class HardwareProfile(BaseModel):
    name: str
    constraints: Constraints
    targets: Targets


# Surgery & Results
class ModelRecipe(BaseModel):
    """The 'Surgery' instructions proposed by the Architect."""

    bits: int = Field(4, ge=2, le=8)
    pruning_ratio: float = Field(0.0, ge=0.0, le=1.0)
    layers_to_drop: List[int] = Field(
        default_factory=list, description="Indices of layers to prune."
    )
    method: str = Field("GGUF")


class ExperimentResult(BaseModel):
    actual_tps: float
    actual_vram_usage: float
    accuracy_score: float
    was_successful: bool
    error_log: Optional[str] = None
