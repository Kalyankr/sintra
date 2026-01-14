"""Tools for the Sintra Architect agent.

These tools enable the architect to research and make informed decisions
about compression strategies before proposing recipes.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Input/Output Schemas
# ============================================================================


class ModelArchitecture(BaseModel):
    """Actual model architecture information from HuggingFace."""
    
    model_id: str = Field(description="The HuggingFace model ID")
    num_layers: int = Field(description="Number of transformer layers")
    hidden_size: int = Field(description="Hidden dimension size")
    num_attention_heads: int = Field(description="Number of attention heads")
    vocab_size: int = Field(description="Vocabulary size")
    num_parameters: float = Field(description="Total parameters in billions")
    architecture_type: str = Field(description="Model architecture (e.g., LlamaForCausalLM)")
    max_position_embeddings: int = Field(description="Maximum context length")


class ModelSearchResult(BaseModel):
    """Result from searching HuggingFace for similar models."""
    
    model_id: str = Field(description="The HuggingFace model ID")
    downloads: int = Field(description="Number of downloads")
    likes: int = Field(description="Number of likes")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    quantization_info: Optional[str] = Field(
        default=None, description="Known quantization variants"
    )


class CompressionEstimate(BaseModel):
    """Estimated impact of a compression configuration."""
    
    estimated_size_gb: float = Field(description="Estimated model size in GB")
    estimated_tps_range: tuple[float, float] = Field(
        description="Expected TPS range (min, max)"
    )
    estimated_accuracy_loss: float = Field(
        description="Expected accuracy loss (0.0 to 1.0)"
    )
    confidence: float = Field(
        description="Confidence in this estimate (0.0 to 1.0)"
    )
    reasoning: str = Field(description="Explanation of the estimate")


class HardwareCapability(BaseModel):
    """Hardware capability information."""
    
    device_name: str = Field(description="Name of the device")
    available_vram_gb: float = Field(description="Available VRAM/RAM in GB")
    supports_4bit: bool = Field(description="Whether 4-bit quantization is supported")
    supports_8bit: bool = Field(description="Whether 8-bit quantization is supported")
    recommended_bits: List[int] = Field(description="Recommended bit widths")
    max_model_params_billions: float = Field(
        description="Maximum model parameters in billions"
    )


# ============================================================================
# Tools
# ============================================================================


@tool
def get_model_architecture(model_id: str) -> Dict[str, Any]:
    """Fetch the actual architecture of a model from HuggingFace.
    
    ALWAYS use this tool before proposing layer dropping or pruning
    to know the exact number of layers, hidden size, and other details.
    This prevents suggesting invalid configurations like dropping layer 33
    when the model only has 32 layers.
    
    Args:
        model_id: The HuggingFace model ID (e.g., "meta-llama/Meta-Llama-3-8B")
        
    Returns:
        Model architecture details including layer count, hidden size, etc.
    """
    try:
        from huggingface_hub import hf_hub_download, HfApi
        import json
        
        api = HfApi()
        
        # Try to get config.json from the model
        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                repo_type="model",
            )
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not download config.json: {e}")
            # Fall back to model info
            model_info = api.model_info(model_id)
            return _estimate_architecture_from_name(model_id, model_info)
        
        # Extract architecture details
        num_layers = (
            config.get("num_hidden_layers") or 
            config.get("n_layer") or 
            config.get("num_layers") or
            32  # default
        )
        hidden_size = (
            config.get("hidden_size") or 
            config.get("n_embd") or
            config.get("d_model") or
            4096
        )
        num_heads = (
            config.get("num_attention_heads") or
            config.get("n_head") or
            32
        )
        vocab_size = config.get("vocab_size", 32000)
        max_pos = (
            config.get("max_position_embeddings") or
            config.get("n_positions") or
            config.get("max_seq_len") or
            4096
        )
        arch_type = config.get("architectures", ["Unknown"])[0] if config.get("architectures") else config.get("model_type", "Unknown")
        
        # Estimate parameters
        num_params = _estimate_parameters(num_layers, hidden_size, vocab_size, num_heads)
        
        return {
            "success": True,
            "model_id": model_id,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "vocab_size": vocab_size,
            "num_parameters_billions": round(num_params / 1e9, 2),
            "architecture_type": arch_type,
            "max_position_embeddings": max_pos,
            "safe_layers_to_drop": list(range(1, min(4, num_layers // 4))),  # Safe early layers
            "layer_drop_limit": num_layers // 4,  # Max 25% layer drop recommended
            "note": f"Model has {num_layers} layers. Do NOT drop more than {num_layers // 4} layers.",
        }
        
    except ImportError:
        logger.warning("huggingface_hub not installed, using estimates")
        return _estimate_architecture_from_name(model_id, None)
    except Exception as e:
        logger.warning(f"Failed to fetch model architecture: {e}")
        return _estimate_architecture_from_name(model_id, None)


def _estimate_parameters(num_layers: int, hidden_size: int, vocab_size: int, num_heads: int) -> float:
    """Estimate total parameters based on architecture."""
    # Embedding parameters
    embed_params = vocab_size * hidden_size * 2  # input + output embeddings
    
    # Per-layer parameters (attention + MLP)
    head_dim = hidden_size // num_heads
    attn_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
    mlp_params = 3 * hidden_size * (4 * hidden_size)  # up, gate, down (for LLaMA-style)
    layer_params = attn_params + mlp_params
    
    total = embed_params + (num_layers * layer_params)
    return total


def _estimate_architecture_from_name(model_id: str, model_info: Any) -> Dict[str, Any]:
    """Estimate architecture based on model name patterns."""
    name_lower = model_id.lower()
    
    # Common model configurations
    configs = {
        "70b": {"layers": 80, "hidden": 8192, "heads": 64, "params": 70},
        "65b": {"layers": 80, "hidden": 8192, "heads": 64, "params": 65},
        "34b": {"layers": 48, "hidden": 8192, "heads": 64, "params": 34},
        "33b": {"layers": 60, "hidden": 6656, "heads": 52, "params": 33},
        "13b": {"layers": 40, "hidden": 5120, "heads": 40, "params": 13},
        "8b": {"layers": 32, "hidden": 4096, "heads": 32, "params": 8},
        "7b": {"layers": 32, "hidden": 4096, "heads": 32, "params": 7},
        "3b": {"layers": 26, "hidden": 3200, "heads": 32, "params": 3},
        "2b": {"layers": 24, "hidden": 2560, "heads": 32, "params": 2},
        "1b": {"layers": 16, "hidden": 2048, "heads": 16, "params": 1},
    }
    
    # Find matching config
    matched = None
    for size_key, config in configs.items():
        if size_key in name_lower:
            matched = config
            break
    
    if not matched:
        matched = configs["7b"]  # Default to 7B
    
    return {
        "success": False,
        "estimated": True,
        "model_id": model_id,
        "num_layers": matched["layers"],
        "hidden_size": matched["hidden"],
        "num_attention_heads": matched["heads"],
        "vocab_size": 32000,
        "num_parameters_billions": matched["params"],
        "architecture_type": "estimated",
        "max_position_embeddings": 4096,
        "safe_layers_to_drop": list(range(1, min(4, matched["layers"] // 4))),
        "layer_drop_limit": matched["layers"] // 4,
        "note": f"ESTIMATED: Could not fetch actual config. Model likely has ~{matched['layers']} layers.",
        "warning": "Install huggingface_hub for accurate architecture info",
    }


@tool
def search_similar_models(
    base_model: str,
    task: str = "text-generation",
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """Search HuggingFace for similar or quantized versions of a model.
    
    Use this tool to find existing quantized versions of a model or
    similar models that might work better for the target hardware.
    
    Args:
        base_model: The base model ID (e.g., "meta-llama/Llama-3.2-1B")
        task: The task type (e.g., "text-generation")
        max_results: Maximum number of results to return
        
    Returns:
        List of similar models with their metadata
    """
    # Extract model family from the base model
    model_family = base_model.split("/")[-1].lower()
    
    # Common quantized model patterns
    quantized_patterns = [
        ("GGUF", ["Q4_K_M", "Q5_K_M", "Q8_0"]),
        ("AWQ", ["awq", "4bit"]),
        ("GPTQ", ["gptq", "4bit"]),
    ]
    
    results = []
    
    # Simulate search results based on common patterns
    # In production, this would call the HuggingFace API
    if "llama" in model_family:
        results.extend([
            {
                "model_id": "TheBloke/Llama-2-7B-GGUF",
                "downloads": 500000,
                "likes": 1200,
                "tags": ["gguf", "llama", "quantized"],
                "quantization_info": "Q4_K_M, Q5_K_M, Q8_0 available",
            },
            {
                "model_id": "unsloth/Llama-3.2-1B",
                "downloads": 250000,
                "likes": 800,
                "tags": ["llama", "small", "efficient"],
                "quantization_info": "Base model, optimized for fine-tuning",
            },
        ])
    elif "mistral" in model_family:
        results.extend([
            {
                "model_id": "TheBloke/Mistral-7B-GGUF",
                "downloads": 400000,
                "likes": 950,
                "tags": ["gguf", "mistral", "quantized"],
                "quantization_info": "Q4_K_M recommended for edge",
            },
        ])
    elif "phi" in model_family:
        results.extend([
            {
                "model_id": "microsoft/phi-2-gguf",
                "downloads": 300000,
                "likes": 700,
                "tags": ["gguf", "phi", "small"],
                "quantization_info": "2.7B params, excellent for edge",
            },
        ])
    
    # Add generic results
    results.append({
        "model_id": f"{base_model}-optimized",
        "downloads": 10000,
        "likes": 50,
        "tags": ["optimized", task],
        "quantization_info": "Community optimized version",
    })
    
    return results[:max_results]


@tool
def estimate_compression_impact(
    model_size_billions: float,
    target_bits: int,
    pruning_ratio: float,
    layers_to_drop: int = 0,
    total_layers: int = 32,
) -> Dict[str, Any]:
    """Estimate the impact of compression settings on model performance.
    
    Use this tool before proposing a recipe to predict whether it will
    meet the hardware constraints and performance targets.
    
    Args:
        model_size_billions: Original model size in billions of parameters
        target_bits: Target quantization bits (2, 3, 4, 5, 6, or 8)
        pruning_ratio: Fraction of weights to prune (0.0 to 0.5)
        layers_to_drop: Number of layers to remove
        total_layers: Total number of layers in the model
        
    Returns:
        Estimated performance metrics and recommendations
    """
    # Base calculations
    base_size_gb = model_size_billions * 2  # FP16 baseline
    
    # Quantization impact
    bit_multiplier = target_bits / 16.0
    quantized_size = base_size_gb * bit_multiplier
    
    # Pruning impact (structured pruning reduces size)
    pruned_size = quantized_size * (1 - pruning_ratio * 0.3)
    
    # Layer dropping impact
    layer_reduction = layers_to_drop / total_layers
    final_size = pruned_size * (1 - layer_reduction)
    
    # TPS estimation (inverse relationship with size, rough heuristic)
    base_tps = 10.0  # Baseline TPS for FP16
    tps_boost_from_quant = (16 - target_bits) * 1.5
    tps_boost_from_pruning = pruning_ratio * 15
    tps_boost_from_layers = layer_reduction * 20
    
    estimated_tps = base_tps + tps_boost_from_quant + tps_boost_from_pruning + tps_boost_from_layers
    tps_variance = estimated_tps * 0.2
    
    # Accuracy estimation
    accuracy_loss_from_quant = max(0, (8 - target_bits) * 0.02)
    accuracy_loss_from_pruning = pruning_ratio * 0.15
    accuracy_loss_from_layers = layer_reduction * 0.25
    
    total_accuracy_loss = min(0.5, accuracy_loss_from_quant + accuracy_loss_from_pruning + accuracy_loss_from_layers)
    
    # Confidence based on how aggressive the compression is
    aggressiveness = (accuracy_loss_from_quant + accuracy_loss_from_pruning + accuracy_loss_from_layers) / 0.5
    confidence = max(0.3, 1.0 - aggressiveness * 0.5)
    
    # Generate reasoning
    reasoning_parts = []
    if target_bits <= 4:
        reasoning_parts.append(f"{target_bits}-bit quantization provides good compression with moderate quality loss")
    if pruning_ratio > 0.2:
        reasoning_parts.append(f"{pruning_ratio:.0%} pruning is aggressive, expect noticeable accuracy impact")
    if layers_to_drop > 0:
        reasoning_parts.append(f"Dropping {layers_to_drop} layers removes {layer_reduction:.0%} of model capacity")
    
    return {
        "estimated_size_gb": round(final_size, 2),
        "estimated_tps_range": (round(estimated_tps - tps_variance, 1), round(estimated_tps + tps_variance, 1)),
        "estimated_accuracy_loss": round(total_accuracy_loss, 3),
        "confidence": round(confidence, 2),
        "reasoning": " | ".join(reasoning_parts) if reasoning_parts else "Standard compression settings",
        "recommendations": {
            "safe_for_production": total_accuracy_loss < 0.1,
            "good_for_edge": final_size < 4.0 and estimated_tps > 20,
            "needs_evaluation": total_accuracy_loss > 0.15,
        },
    }


@tool
def query_hardware_capabilities(
    device_name: str,
    available_memory_gb: float,
    has_gpu: bool = False,
    gpu_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Query what compression configurations are supported by the target hardware.
    
    Use this tool to understand what quantization methods and model sizes
    can run on the target device.
    
    Args:
        device_name: Name of the device (e.g., "Raspberry Pi 5", "Mac Mini M4")
        available_memory_gb: Available RAM/VRAM in GB
        has_gpu: Whether the device has a GPU
        gpu_type: Type of GPU if available (e.g., "CUDA", "Metal", "None")
        
    Returns:
        Hardware capabilities and recommendations
    """
    # Determine supported quantization based on hardware
    supported_bits = []
    recommended_bits = []
    
    if available_memory_gb >= 16:
        supported_bits = [2, 3, 4, 5, 6, 8]
        recommended_bits = [4, 5]
        max_params = 7.0
    elif available_memory_gb >= 8:
        supported_bits = [2, 3, 4, 5, 6]
        recommended_bits = [4]
        max_params = 3.0
    elif available_memory_gb >= 4:
        supported_bits = [2, 3, 4]
        recommended_bits = [3, 4]
        max_params = 1.5
    else:
        supported_bits = [2, 3]
        recommended_bits = [2, 3]
        max_params = 0.5
    
    # GPU-specific recommendations
    supports_4bit = True
    supports_8bit = available_memory_gb >= 8
    
    if has_gpu:
        if gpu_type == "CUDA":
            # CUDA supports all quantization methods
            supported_bits = [2, 3, 4, 5, 6, 8]
            max_params *= 1.5  # GPU offloading helps
        elif gpu_type == "Metal":
            # Metal (Apple Silicon) is efficient
            supported_bits = [2, 3, 4, 5, 6, 8]
            max_params *= 1.3
    
    return {
        "device_name": device_name,
        "available_vram_gb": available_memory_gb,
        "supports_4bit": supports_4bit,
        "supports_8bit": supports_8bit,
        "supported_bits": supported_bits,
        "recommended_bits": recommended_bits,
        "max_model_params_billions": round(max_params, 1),
        "recommendations": {
            "backend": "gguf" if not has_gpu or gpu_type != "CUDA" else "bnb",
            "start_with_bits": recommended_bits[0] if recommended_bits else 4,
            "max_pruning_ratio": 0.3 if available_memory_gb < 8 else 0.2,
        },
    }


@tool
def lookup_quantization_benchmarks(
    model_family: str,
    bits: int,
) -> Dict[str, Any]:
    """Look up known benchmark results for a model family at a specific bit width.
    
    Use this tool to see what performance others have achieved with
    similar compression settings.
    
    Args:
        model_family: Model family (e.g., "llama", "mistral", "phi")
        bits: Quantization bit width
        
    Returns:
        Known benchmark results and recommendations
    """
    # Simulated benchmark database
    # In production, this would query a real database or API
    benchmarks = {
        "llama": {
            2: {"tps_range": (25, 40), "accuracy_drop": 0.15, "quality": "low"},
            3: {"tps_range": (20, 35), "accuracy_drop": 0.08, "quality": "medium"},
            4: {"tps_range": (15, 30), "accuracy_drop": 0.04, "quality": "good"},
            5: {"tps_range": (12, 25), "accuracy_drop": 0.02, "quality": "very_good"},
            6: {"tps_range": (10, 20), "accuracy_drop": 0.01, "quality": "excellent"},
            8: {"tps_range": (8, 15), "accuracy_drop": 0.005, "quality": "near_original"},
        },
        "mistral": {
            2: {"tps_range": (28, 45), "accuracy_drop": 0.12, "quality": "medium"},
            3: {"tps_range": (22, 38), "accuracy_drop": 0.06, "quality": "good"},
            4: {"tps_range": (18, 32), "accuracy_drop": 0.03, "quality": "very_good"},
            5: {"tps_range": (14, 26), "accuracy_drop": 0.015, "quality": "excellent"},
            6: {"tps_range": (11, 21), "accuracy_drop": 0.008, "quality": "excellent"},
            8: {"tps_range": (9, 16), "accuracy_drop": 0.003, "quality": "near_original"},
        },
        "phi": {
            2: {"tps_range": (35, 55), "accuracy_drop": 0.10, "quality": "good"},
            3: {"tps_range": (30, 48), "accuracy_drop": 0.05, "quality": "very_good"},
            4: {"tps_range": (25, 42), "accuracy_drop": 0.025, "quality": "excellent"},
            5: {"tps_range": (20, 35), "accuracy_drop": 0.012, "quality": "excellent"},
            6: {"tps_range": (16, 28), "accuracy_drop": 0.006, "quality": "near_original"},
            8: {"tps_range": (12, 22), "accuracy_drop": 0.002, "quality": "near_original"},
        },
    }
    
    # Normalize model family
    family_lower = model_family.lower()
    matched_family = None
    for known_family in benchmarks:
        if known_family in family_lower:
            matched_family = known_family
            break
    
    if not matched_family:
        return {
            "found": False,
            "message": f"No benchmark data for {model_family}. Using generic estimates.",
            "generic_estimate": {
                "tps_range": (10 + (8 - bits) * 3, 20 + (8 - bits) * 5),
                "accuracy_drop": max(0, (8 - bits) * 0.02),
                "quality": "unknown",
            },
        }
    
    if bits not in benchmarks[matched_family]:
        return {
            "found": False,
            "message": f"No data for {bits}-bit quantization of {matched_family}",
        }
    
    data = benchmarks[matched_family][bits]
    return {
        "found": True,
        "model_family": matched_family,
        "bits": bits,
        "benchmark_results": data,
        "recommendation": f"{bits}-bit {matched_family} typically achieves {data['quality']} quality "
                         f"with TPS in range {data['tps_range']} and ~{data['accuracy_drop']:.1%} accuracy drop",
    }


# ============================================================================
# Tool Collection
# ============================================================================


def get_architect_tools() -> List:
    """Get all tools available to the architect agent."""
    return [
        get_model_architecture,  # ALWAYS call first to know layer count
        search_similar_models,
        estimate_compression_impact,
        query_hardware_capabilities,
        lookup_quantization_benchmarks,
    ]
