"""Model compression pipeline for Sintra.

This module provides the core compression functionality:
- Downloading models from HuggingFace
- Converting to GGUF format
- Quantizing to various bit depths
- Evaluating model accuracy
"""

from .downloader import ModelDownloader, download_model
from .quantizer import GGUFQuantizer, QuantizationType, quantize_model
from .evaluator import AccuracyEvaluator, evaluate_perplexity

__all__ = [
    "ModelDownloader",
    "download_model",
    "GGUFQuantizer",
    "QuantizationType",
    "quantize_model",
    "AccuracyEvaluator",
    "evaluate_perplexity",
]
