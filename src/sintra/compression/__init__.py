"""Model compression pipeline for Sintra.

This module provides the core compression functionality:
- Downloading models from HuggingFace
- Converting to GGUF format (llama.cpp)
- Quantizing to various bit depths
- Pruning and layer dropping
- Evaluating model accuracy

Advanced backends (optional):
- BitsAndBytes: pip install sintra[bnb]
- Optimum/ONNX: pip install sintra[onnx]
"""

import contextlib

from .downloader import ModelDownloader, download_model
from .evaluator import (
    AccuracyComparison,
    AccuracyEvaluator,
    compare_accuracy,
    evaluate_perplexity,
    evaluate_with_baseline,
)
from .pruner import (
    LayerDropper,
    PruningError,
    StructuredPruner,
    apply_compression,
    drop_layers,
    prune_model,
)
from .quantizer import (
    LLAMA_CPP_INSTALL_INSTRUCTIONS,
    GGUFQuantizer,
    QuantizationError,
    QuantizationType,
    check_llama_cpp_available,
    quantize_model,
    quantize_with_compression,
)

# Optional backends - import only if available
with contextlib.suppress(ImportError):
    from .bnb_quantizer import (  # noqa: F401
        BitsAndBytesError,
        BitsAndBytesQuantizer,
        BnBQuantType,
        is_bnb_available,
    )

with contextlib.suppress(ImportError):
    from .onnx_optimizer import (  # noqa: F401
        ONNXOptimizer,
        ONNXOptimizerError,
        OptimizationLevel,
        QuantizationMode,
        is_onnx_available,
    )

# Ollama exporter (always available)
from .ollama_exporter import (
    OllamaExporter,
    OllamaExportError,
    OllamaExportResult,
    export_to_ollama,
    is_ollama_available,
)

__all__ = [
    "LLAMA_CPP_INSTALL_INSTRUCTIONS",
    "AccuracyComparison",
    "AccuracyEvaluator",
    "GGUFQuantizer",
    "LayerDropper",
    "ModelDownloader",
    "OllamaExportError",
    "OllamaExportResult",
    "OllamaExporter",
    "PruningError",
    "QuantizationError",
    "QuantizationType",
    "StructuredPruner",
    "apply_compression",
    "check_llama_cpp_available",
    "compare_accuracy",
    "download_model",
    "drop_layers",
    "evaluate_perplexity",
    "evaluate_with_baseline",
    "export_to_ollama",
    "is_ollama_available",
    "prune_model",
    "quantize_model",
    "quantize_with_compression",
]
