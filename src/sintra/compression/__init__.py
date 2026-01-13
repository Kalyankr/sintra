"""Model compression pipeline for Sintra.

This module provides the core compression functionality:
- Downloading models from HuggingFace
- Converting to GGUF format
- Quantizing to various bit depths
- Pruning and layer dropping
- Evaluating model accuracy
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

__all__ = [
    # Downloader
    "ModelDownloader",
    "download_model",
    # Quantizer
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
