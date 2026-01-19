"""BitsAndBytes quantization backend.

This module provides GPU-accelerated quantization using the bitsandbytes library,
which is the industry standard for NF4 and INT8 quantization with transformers.

Key features:
- NF4 (Normal Float 4-bit): Best quality 4-bit quantization
- INT8 (LLM.int8()): Dynamic 8-bit quantization with outlier handling
- Double quantization: Further compress quantization constants
- Native transformers integration

Usage:
    from sintra.compression.bnb_quantizer import BitsAndBytesQuantizer

    quantizer = BitsAndBytesQuantizer()
    model_path = quantizer.quantize("meta-llama/Llama-3.2-1B", bits=4)
"""

import json
import sys
from enum import Enum
from pathlib import Path

import torch


class BnBQuantType(str, Enum):
    """BitsAndBytes quantization types."""

    NF4 = "nf4"  # Normal Float 4-bit (best quality)
    FP4 = "fp4"  # Float Point 4-bit
    INT8 = "int8"  # LLM.int8() dynamic quantization


class BitsAndBytesError(Exception):
    """Error during bitsandbytes quantization."""

    pass


def _check_bnb_available() -> bool:
    """Check if bitsandbytes is available."""
    try:
        import accelerate  # noqa: F401
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


def _check_cuda_available() -> bool:
    """Check if CUDA is available for bitsandbytes."""
    return torch.cuda.is_available()


class BitsAndBytesQuantizer:
    """Quantize models using bitsandbytes (NF4/INT8).

    BitsAndBytes provides GPU-accelerated quantization that integrates
    directly with the transformers library. It's the standard choice for
    running large models on consumer GPUs.

    Attributes:
        cache_dir: Directory for cached quantized models.
        compute_dtype: Dtype for computation (bfloat16 recommended).

    Note:
        Requires GPU with CUDA support. For CPU-only systems, use
        GGUFQuantizer instead.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the quantizer.

        Args:
            cache_dir: Directory for cached models.
            compute_dtype: Dtype for computations (default: bfloat16).

        Raises:
            BitsAndBytesError: If bitsandbytes is not installed.
        """
        if not _check_bnb_available():
            raise BitsAndBytesError(
                "bitsandbytes not installed. Install with: pip install sintra[bnb]"
            )

        self.cache_dir = cache_dir or Path.home() / ".cache" / "sintra" / "bnb"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compute_dtype = compute_dtype

    def quantize(
        self,
        model_id_or_path: str,
        bits: int = 4,
        quant_type: BnBQuantType = BnBQuantType.NF4,
        use_double_quant: bool = True,
        output_name: str | None = None,
    ) -> Path:
        """Quantize a model using bitsandbytes.

        This loads a model with quantization applied via transformers'
        BitsAndBytesConfig, then saves it for later use.

        Args:
            model_id_or_path: HuggingFace model ID or local path.
            bits: Quantization bits (4 or 8).
            quant_type: Quantization type (nf4, fp4, int8).
            use_double_quant: Use nested quantization for 4-bit.
            output_name: Custom name for output directory.

        Returns:
            Path to the quantized model directory.

        Raises:
            BitsAndBytesError: If quantization fails.
            ValueError: If invalid bits specified.
        """
        if bits not in (4, 8):
            raise ValueError(f"BitsAndBytes supports 4 or 8 bits, got {bits}")

        if bits == 8 and quant_type != BnBQuantType.INT8:
            quant_type = BnBQuantType.INT8

        if not _check_cuda_available():
            raise BitsAndBytesError(
                "CUDA not available. BitsAndBytes requires GPU. "
                "Use GGUFQuantizer for CPU-only systems."
            )

        # Import here to avoid loading if not needed
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        sys.stderr.write(
            f"BnB: Loading {model_id_or_path} with {bits}-bit {quant_type.value}...\n"
        )

        # Configure quantization
        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type.value,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=use_double_quant,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        try:
            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=self.compute_dtype,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

            # Generate output path
            if output_name is None:
                model_name = (
                    Path(model_id_or_path).name
                    if "/" not in model_id_or_path
                    else model_id_or_path.split("/")[-1]
                )
                quant_suffix = f"{bits}bit-{quant_type.value}"
                if use_double_quant and bits == 4:
                    quant_suffix += "-dq"
                output_name = f"{model_name}-bnb-{quant_suffix}"

            output_path = self.cache_dir / output_name

            sys.stderr.write(f"BnB: Saving to {output_path}...\n")

            # Save quantized model
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            # Save quantization metadata
            metadata = {
                "source_model": model_id_or_path,
                "bits": bits,
                "quant_type": quant_type.value,
                "double_quant": use_double_quant,
                "compute_dtype": str(self.compute_dtype),
                "backend": "bitsandbytes",
            }
            with open(output_path / "quantization_config.json", "w") as f:
                json.dump(metadata, f, indent=2)

            sys.stderr.write("BnB: Quantization complete!\n")
            return output_path

        except Exception as e:
            raise BitsAndBytesError(f"Quantization failed: {e}") from e

    def load_quantized(
        self,
        model_path: Path,
    ) -> tuple:
        """Load a previously quantized model.

        Args:
            model_path: Path to quantized model directory.

        Returns:
            Tuple of (model, tokenizer).

        Raises:
            BitsAndBytesError: If loading fails.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer
        except Exception as e:
            raise BitsAndBytesError(f"Failed to load model: {e}") from e

    def get_cached_models(self) -> list[Path]:
        """List all cached quantized models.

        Returns:
            List of paths to cached model directories.
        """
        if not self.cache_dir.exists():
            return []

        return [
            p
            for p in self.cache_dir.iterdir()
            if p.is_dir() and (p / "config.json").exists()
        ]

    def estimate_memory(
        self,
        model_id: str,
        bits: int = 4,
    ) -> dict:
        """Estimate memory requirements for quantized model.

        Args:
            model_id: HuggingFace model ID.
            bits: Target quantization bits.

        Returns:
            Dict with memory estimates in GB.
        """
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(model_id)

            # Estimate parameter count
            hidden_size = getattr(config, "hidden_size", 4096)
            num_layers = getattr(config, "num_hidden_layers", 32)
            vocab_size = getattr(config, "vocab_size", 32000)
            intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)

            # Rough parameter estimate
            params_per_layer = (
                4 * hidden_size * hidden_size  # attention
                + 3 * hidden_size * intermediate_size  # MLP
            )
            total_params = (
                vocab_size * hidden_size  # embeddings
                + num_layers * params_per_layer
                + hidden_size  # final norm
            )

            # Memory estimates
            fp16_memory = total_params * 2 / (1024**3)  # 2 bytes per param
            quantized_memory = total_params * bits / 8 / (1024**3)

            return {
                "estimated_params_b": total_params / 1e9,
                "fp16_memory_gb": round(fp16_memory, 2),
                "quantized_memory_gb": round(quantized_memory, 2),
                "memory_saved_gb": round(fp16_memory - quantized_memory, 2),
                "compression_ratio": round(fp16_memory / quantized_memory, 1),
            }
        except Exception:
            return {"error": "Could not estimate memory"}


def is_bnb_available() -> bool:
    """Check if bitsandbytes backend is available.

    Returns:
        True if bitsandbytes and CUDA are available.
    """
    return _check_bnb_available() and _check_cuda_available()
