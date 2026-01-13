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
from .quantizer import (
    GGUFQuantizer,
    QuantizationType,
    quantize_model,
    quantize_with_compression,
)
from .pruner import (
    LayerDropper,
    StructuredPruner,
    PruningError,
    drop_layers,
    prune_model,
    apply_compression,
)
from .evaluator import AccuracyEvaluator, evaluate_perplexity

# Optional backends - import only if available
try:
    from .bnb_quantizer import (
        BitsAndBytesQuantizer,
        BnBQuantType,
        BitsAndBytesError,
        is_bnb_available,
    )
except ImportError:
    pass

try:
    from .onnx_optimizer import (
        ONNXOptimizer,
        OptimizationLevel,
        QuantizationMode,
        ONNXOptimizerError,
        is_onnx_available,
    )
except ImportError:
    pass

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
    "evaluate_perplexity",
]
