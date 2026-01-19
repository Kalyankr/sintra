"""Optimum ONNX export and optimization backend.

This module provides ONNX export and optimization using the Hugging Face Optimum
library, enabling deployment on various hardware accelerators via ONNX Runtime.

Key features:
- Export transformers models to ONNX format
- Graph optimization (fusion, constant folding)
- ONNX Runtime quantization (INT8 dynamic/static)
- Multi-platform inference (CPU, GPU, TensorRT)

Usage:
    from sintra.compression.onnx_optimizer import ONNXOptimizer

    optimizer = ONNXOptimizer()
    onnx_path = optimizer.export("meta-llama/Llama-3.2-1B")
    optimized_path = optimizer.optimize(onnx_path)
"""

import json
import sys
from enum import Enum
from pathlib import Path


class OptimizationLevel(str, Enum):
    """ONNX Runtime optimization levels."""

    DISABLE = "disable"  # No optimization
    BASIC = "basic"  # Basic optimizations
    EXTENDED = "extended"  # More optimizations
    ALL = "all"  # All optimizations including hardware-specific


class QuantizationMode(str, Enum):
    """ONNX quantization modes."""

    DYNAMIC = "dynamic"  # Dynamic quantization (no calibration)
    STATIC = "static"  # Static quantization (requires calibration)


class ONNXOptimizerError(Exception):
    """Error during ONNX optimization."""

    pass


def _check_optimum_available() -> bool:
    """Check if optimum is available."""
    try:
        import onnx  # noqa: F401
        import optimum  # noqa: F401

        return True
    except ImportError:
        return False


def _check_onnxruntime_available() -> bool:
    """Check if ONNX Runtime is available."""
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


class ONNXOptimizer:
    """Export and optimize models using ONNX and Optimum.

    Optimum provides a seamless way to export transformers models to ONNX
    format and apply various optimizations for efficient inference.

    Attributes:
        cache_dir: Directory for cached ONNX models.
        optimization_level: Default optimization level.

    Note:
        ONNX models can run on any platform with ONNX Runtime, including
        CPU, GPU, and specialized accelerators.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        optimization_level: OptimizationLevel = OptimizationLevel.ALL,
    ):
        """Initialize the optimizer.

        Args:
            cache_dir: Directory for cached models.
            optimization_level: Default optimization level.

        Raises:
            ONNXOptimizerError: If optimum is not installed.
        """
        if not _check_optimum_available():
            raise ONNXOptimizerError(
                "optimum not installed. Install with: pip install sintra[onnx]"
            )

        self.cache_dir = cache_dir or Path.home() / ".cache" / "sintra" / "onnx"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.optimization_level = optimization_level

    def export(
        self,
        model_id_or_path: str,
        task: str = "text-generation",
        output_name: str | None = None,
        opset: int = 17,
    ) -> Path:
        """Export a model to ONNX format.

        Uses Optimum's export functionality to convert a transformers
        model to ONNX format with proper handling of dynamic axes.

        Args:
            model_id_or_path: HuggingFace model ID or local path.
            task: Model task (text-generation, text-classification, etc.).
            output_name: Custom name for output directory.
            opset: ONNX opset version (default: 17).

        Returns:
            Path to the exported ONNX model directory.

        Raises:
            ONNXOptimizerError: If export fails.
        """
        from optimum.exporters.onnx import main_export

        sys.stderr.write(f"ONNX: Exporting {model_id_or_path} (task={task})...\n")

        # Generate output path
        if output_name is None:
            model_name = (
                Path(model_id_or_path).name
                if "/" not in model_id_or_path
                else model_id_or_path.split("/")[-1]
            )
            output_name = f"{model_name}-onnx"

        output_path = self.cache_dir / output_name

        try:
            main_export(
                model_name_or_path=model_id_or_path,
                output=output_path,
                task=task,
                opset=opset,
            )

            # Save export metadata
            metadata = {
                "source_model": model_id_or_path,
                "task": task,
                "opset": opset,
                "backend": "optimum-onnx",
            }
            with open(output_path / "export_config.json", "w") as f:
                json.dump(metadata, f, indent=2)

            sys.stderr.write(f"ONNX: Export complete! Saved to {output_path}\n")
            return output_path

        except Exception as e:
            raise ONNXOptimizerError(f"Export failed: {e}") from e

    def optimize(
        self,
        model_path: Path,
        optimization_level: OptimizationLevel | None = None,
        output_name: str | None = None,
    ) -> Path:
        """Apply graph optimizations to an ONNX model.

        Applies optimizations like operator fusion, constant folding,
        and redundant node elimination.

        Args:
            model_path: Path to ONNX model directory.
            optimization_level: Optimization level to apply.
            output_name: Custom name for optimized model.

        Returns:
            Path to the optimized ONNX model directory.

        Raises:
            ONNXOptimizerError: If optimization fails.
        """
        from optimum.onnxruntime import ORTOptimizer
        from optimum.onnxruntime.configuration import AutoOptimizationConfig

        level = optimization_level or self.optimization_level
        sys.stderr.write(f"ONNX: Optimizing with level={level.value}...\n")

        # Generate output path
        if output_name is None:
            output_name = f"{model_path.name}-optimized"

        output_path = self.cache_dir / output_name

        try:
            optimizer = ORTOptimizer.from_pretrained(model_path)

            # Create optimization config based on level
            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level=level.value,
            )

            optimizer.optimize(
                save_dir=output_path,
                optimization_config=optimization_config,
            )

            sys.stderr.write(f"ONNX: Optimization complete! Saved to {output_path}\n")
            return output_path

        except Exception as e:
            raise ONNXOptimizerError(f"Optimization failed: {e}") from e

    def quantize(
        self,
        model_path: Path,
        mode: QuantizationMode = QuantizationMode.DYNAMIC,
        output_name: str | None = None,
    ) -> Path:
        """Apply INT8 quantization to an ONNX model.

        Quantizes the model weights and activations to INT8 for faster
        inference with minimal accuracy loss.

        Args:
            model_path: Path to ONNX model directory.
            mode: Quantization mode (dynamic or static).
            output_name: Custom name for quantized model.

        Returns:
            Path to the quantized ONNX model directory.

        Raises:
            ONNXOptimizerError: If quantization fails.
        """
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        sys.stderr.write(f"ONNX: Quantizing with mode={mode.value}...\n")

        # Generate output path
        if output_name is None:
            output_name = f"{model_path.name}-int8-{mode.value}"

        output_path = self.cache_dir / output_name

        try:
            quantizer = ORTQuantizer.from_pretrained(model_path)

            # Create quantization config
            if mode == QuantizationMode.DYNAMIC:
                quant_config = AutoQuantizationConfig.avx512_vnni(
                    is_static=False,
                    per_channel=False,
                )
            else:
                # Static quantization requires calibration dataset
                quant_config = AutoQuantizationConfig.avx512_vnni(
                    is_static=True,
                    per_channel=True,
                )

            quantizer.quantize(
                save_dir=output_path,
                quantization_config=quant_config,
            )

            sys.stderr.write(f"ONNX: Quantization complete! Saved to {output_path}\n")
            return output_path

        except Exception as e:
            raise ONNXOptimizerError(f"Quantization failed: {e}") from e

    def export_and_optimize(
        self,
        model_id_or_path: str,
        task: str = "text-generation",
        apply_quantization: bool = False,
        output_name: str | None = None,
    ) -> Path:
        """Export, optimize, and optionally quantize in one step.

        Convenience method that chains export → optimize → quantize.

        Args:
            model_id_or_path: HuggingFace model ID or local path.
            task: Model task.
            apply_quantization: Whether to apply INT8 quantization.
            output_name: Custom name for final output.

        Returns:
            Path to the final optimized model directory.
        """
        # Export
        exported_path = self.export(model_id_or_path, task=task)

        # Optimize
        optimized_path = self.optimize(exported_path)

        # Optionally quantize
        if apply_quantization:
            final_path = self.quantize(optimized_path)
        else:
            final_path = optimized_path

        # Rename to final output name if specified
        if output_name:
            target_path = self.cache_dir / output_name
            if target_path.exists():
                import shutil

                shutil.rmtree(target_path)
            final_path.rename(target_path)
            final_path = target_path

        return final_path

    def load_for_inference(
        self,
        model_path: Path,
        provider: str = "CPUExecutionProvider",
    ):
        """Load an ONNX model for inference.

        Args:
            model_path: Path to ONNX model directory.
            provider: ONNX Runtime execution provider.

        Returns:
            ORTModel ready for inference.

        Raises:
            ONNXOptimizerError: If loading fails.
        """
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        try:
            model = ORTModelForCausalLM.from_pretrained(
                model_path,
                provider=provider,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer
        except Exception as e:
            raise ONNXOptimizerError(f"Failed to load model: {e}") from e

    def get_cached_models(self) -> list[Path]:
        """List all cached ONNX models.

        Returns:
            List of paths to cached model directories.
        """
        if not self.cache_dir.exists():
            return []

        return [
            p for p in self.cache_dir.iterdir() if p.is_dir() and any(p.glob("*.onnx"))
        ]

    def benchmark(
        self,
        model_path: Path,
        prompt: str = "Hello, how are you?",
        num_runs: int = 10,
    ) -> dict:
        """Benchmark an ONNX model's inference speed.

        Args:
            model_path: Path to ONNX model.
            prompt: Test prompt.
            num_runs: Number of inference runs.

        Returns:
            Dict with timing statistics.
        """
        import time

        try:
            model, tokenizer = self.load_for_inference(model_path)
            inputs = tokenizer(prompt, return_tensors="pt")

            # Warmup
            _ = model.generate(**inputs, max_new_tokens=10)

            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model.generate(**inputs, max_new_tokens=50)
                times.append(time.perf_counter() - start)

            return {
                "mean_latency_ms": round(sum(times) / len(times) * 1000, 2),
                "min_latency_ms": round(min(times) * 1000, 2),
                "max_latency_ms": round(max(times) * 1000, 2),
                "num_runs": num_runs,
            }
        except Exception as e:
            return {"error": str(e)}


def is_onnx_available() -> bool:
    """Check if ONNX backend is available.

    Returns:
        True if optimum and onnxruntime are available.
    """
    return _check_optimum_available() and _check_onnxruntime_available()
