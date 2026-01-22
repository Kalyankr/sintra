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
try:
    from .bnb_quantizer import (
        BitsAndBytesError,
        BitsAndBytesQuantizer,
        BnBQuantType,
        is_bnb_available,
    )
except ImportError:
    pass

try:
    from .onnx_optimizer import (
        ONNXOptimizer,
        ONNXOptimizerError,
        OptimizationLevel,
        QuantizationMode,
        is_onnx_available,
    )
except ImportError:
    pass

# Ollama exporter (always available)
from .ollama_exporter import (
    OllamaExporter,
    OllamaExportError,
    OllamaExportResult,
    export_to_ollama,
    is_ollama_available,
)

__all__ = [
    # Downloader
    "ModelDownloader",
    "download_model",
    # GGUF Quantizer (llama.cpp)
    "GGUFQuantizer",
    "QuantizationType",
    "quantize_model",
    "quantize_with_compression",
    # Pruner
    "LayerDropper",
    "StructuredPruner",
    "PruningError",
    "drop_layers",
    "prune_model",
    "apply_compression",
    # Evaluator
    "AccuracyEvaluator",
    "AccuracyComparison",
    "evaluate_perplexity",
    "compare_accuracy",
    "evaluate_with_baseline",
    # Ollama Exporter
    "OllamaExporter",
    "OllamaExportError",
    "OllamaExportResult",
    "export_to_ollama",
    "is_ollama_available",
]
