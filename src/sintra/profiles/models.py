from enum import Enum

from pydantic import BaseModel, Field, field_validator


# LLM Architect Config
class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.OLLAMA
    model_name: str = "qwen3:8b"
    temperature: float = 0.8


# Hardware & Target Definitions
class Constraints(BaseModel):
    vram_gb: float
    cpu_arch: str | None = None
    has_cuda: bool = False


class Targets(BaseModel):
    min_tokens_per_second: float
    min_accuracy_score: float
    max_latency_ms: float | None = None


class HardwareProfile(BaseModel):
    name: str
    constraints: Constraints
    targets: Targets
    supported_quantizations: list[str] | None = None


# Surgery & Results
class ModelRecipe(BaseModel):
    """The 'Surgery' instructions proposed by the Architect."""

    bits: int = Field(4, ge=2, le=8)
    pruning_ratio: float = Field(0.0, ge=0.0, le=1.0)
    layers_to_drop: list[int] = Field(
        default_factory=list, description="Indices of layers to prune."
    )
    method: str = Field("GGUF")

    @field_validator("pruning_ratio", mode="before")
    @classmethod
    def scale_pruning_ratio(cls, v):
        # If the AI sends 50.0 instead of 0.5, fix it automatically
        if isinstance(v, (int, float)) and v > 1.0:
            return v / 100.0
        return v


class ExperimentResult(BaseModel):
    actual_tps: float
    actual_vram_usage: float
    accuracy_score: float
    was_successful: bool
    error_log: str | None = None
    # Baseline comparison fields (optional, may not be present in older data)
    accuracy_retention: float | None = None  # e.g., 0.95 = 95% of original accuracy
    accuracy_loss: float | None = None  # e.g., 0.05 = 5% loss vs original
