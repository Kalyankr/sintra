from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class HardwareConstraints(BaseModel):
    vram_gb: float = Field(..., description="Maximum available VRAM in Gigabytes")
    cpu_arch: str = Field(..., pattern="^(x86_64|arm64)$")
    has_cuda: bool = False
    max_power_draw_watts: Optional[float] = None


class TargetMetrics(BaseModel):
    min_tokens_per_second: float = Field(default=5.0)
    max_latency_ms: float = Field(default=500.0)
    min_accuracy_score: float = Field(
        0.7, description="Baseline MMLU or reasoning score"
    )


class HardwareProfile(BaseModel):
    name: str
    constraints: HardwareConstraints
    targets: TargetMetrics
    supported_quantizations: List[str] = ["GGUF", "GPTQ", "AWQ"]


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"  # For local-hosted architecting
    GOOGLE = "google"


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4o"
    temperature: float = 0.1
    api_key_env_var: str = "OPENAI_API_KEY"
