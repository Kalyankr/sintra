"""Tests for the compression module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sintra.compression.downloader import (
    DEFAULT_CACHE_DIR,
    DownloadError,
    ModelDownloader,
)
from sintra.compression.evaluator import (
    AccuracyEvaluator,
    EvaluationError,
)
from sintra.compression.quantizer import (
    BITS_TO_QUANT,
    GGUFQuantizer,
    QuantizationError,
    QuantizationType,
)


class TestModelDownloader:
    """Tests for ModelDownloader class."""

    def test_default_cache_dir(self) -> None:
        """Test default cache directory is set correctly."""
        downloader = ModelDownloader()
        assert downloader.cache_dir == DEFAULT_CACHE_DIR
        assert downloader.downloads_dir == DEFAULT_CACHE_DIR / "downloads"

    def test_custom_cache_dir(self, tmp_path: Path) -> None:
        """Test custom cache directory."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        assert downloader.cache_dir == tmp_path
        assert downloader.downloads_dir == tmp_path / "downloads"
        assert downloader.downloads_dir.exists()

    def test_is_complete_download_missing_dir(self, tmp_path: Path) -> None:
        """Test _is_complete_download returns False for missing directory."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        assert not downloader._is_complete_download(tmp_path / "nonexistent")

    def test_is_complete_download_missing_config(self, tmp_path: Path) -> None:
        """Test _is_complete_download returns False without config.json."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").touch()

        downloader = ModelDownloader(cache_dir=tmp_path)
        assert not downloader._is_complete_download(model_dir)

    def test_is_complete_download_missing_weights(self, tmp_path: Path) -> None:
        """Test _is_complete_download returns False without weight files."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").touch()

        downloader = ModelDownloader(cache_dir=tmp_path)
        assert not downloader._is_complete_download(model_dir)

    def test_is_complete_download_valid(self, tmp_path: Path) -> None:
        """Test _is_complete_download returns True for valid download."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").touch()
        (model_dir / "model.safetensors").touch()

        downloader = ModelDownloader(cache_dir=tmp_path)
        assert downloader._is_complete_download(model_dir)

    def test_list_cached_models_empty(self, tmp_path: Path) -> None:
        """Test list_cached_models with empty cache."""
        downloader = ModelDownloader(cache_dir=tmp_path)
        assert downloader.list_cached_models() == []


class TestGGUFQuantizer:
    """Tests for GGUFQuantizer class."""

    def test_default_cache_dirs(self) -> None:
        """Test default cache directories are created."""
        quantizer = GGUFQuantizer()
        assert quantizer.gguf_dir == DEFAULT_CACHE_DIR / "gguf"
        assert quantizer.quantized_dir == DEFAULT_CACHE_DIR / "quantized"

    def test_custom_cache_dir(self, tmp_path: Path) -> None:
        """Test custom cache directory."""
        quantizer = GGUFQuantizer(cache_dir=tmp_path)
        assert quantizer.gguf_dir == tmp_path / "gguf"
        assert quantizer.quantized_dir == tmp_path / "quantized"
        assert quantizer.gguf_dir.exists()
        assert quantizer.quantized_dir.exists()

    def test_bits_to_quant_mapping(self) -> None:
        """Test bits to quantization type mapping."""
        assert BITS_TO_QUANT[2] == QuantizationType.Q2_K
        assert BITS_TO_QUANT[4] == QuantizationType.Q4_K_M
        assert BITS_TO_QUANT[8] == QuantizationType.Q8_0

    def test_unsupported_bits_raises_error(self, tmp_path: Path) -> None:
        """Test unsupported bit depth raises error."""
        quantizer = GGUFQuantizer(cache_dir=tmp_path)

        with pytest.raises(QuantizationError, match="Unsupported bit depth"):
            quantizer.quantize(tmp_path / "model", bits=7)

    def test_get_cached_quantizations_empty(self, tmp_path: Path) -> None:
        """Test get_cached_quantizations with no cached models."""
        quantizer = GGUFQuantizer(cache_dir=tmp_path)
        assert quantizer.get_cached_quantizations("nonexistent") == []


class TestQuantizationType:
    """Tests for QuantizationType enum."""

    def test_all_quant_types_defined(self) -> None:
        """Test all expected quantization types are defined."""
        expected = [
            "Q2_K",
            "Q3_K_S",
            "Q3_K_M",
            "Q3_K_L",
            "Q4_0",
            "Q4_K_S",
            "Q4_K_M",
            "Q5_0",
            "Q5_K_S",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
            "F16",
            "F32",
        ]

        for quant in expected:
            assert hasattr(QuantizationType, quant)

    def test_quant_type_values(self) -> None:
        """Test quantization type string values."""
        assert QuantizationType.Q4_K_M.value == "Q4_K_M"
        assert QuantizationType.Q8_0.value == "Q8_0"


class TestAccuracyEvaluator:
    """Tests for AccuracyEvaluator class."""

    def test_default_eval_text(self) -> None:
        """Test default evaluation text is set."""
        evaluator = AccuracyEvaluator()
        assert evaluator.eval_text is not None
        assert len(evaluator.eval_text) > 100

    def test_custom_eval_text(self) -> None:
        """Test custom evaluation text."""
        custom_text = "This is a test."
        evaluator = AccuracyEvaluator(eval_text=custom_text)
        assert evaluator.eval_text == custom_text

    def test_perplexity_to_accuracy_low(self) -> None:
        """Test low perplexity gives high accuracy."""
        evaluator = AccuracyEvaluator()
        accuracy = evaluator._perplexity_to_accuracy(5.0)
        assert accuracy >= 0.9

    def test_perplexity_to_accuracy_high(self) -> None:
        """Test high perplexity gives low accuracy."""
        evaluator = AccuracyEvaluator()
        accuracy = evaluator._perplexity_to_accuracy(100.0)
        assert accuracy <= 0.5

    def test_perplexity_to_accuracy_bounds(self) -> None:
        """Test accuracy is bounded between 0.1 and 0.99."""
        evaluator = AccuracyEvaluator()

        # Very low perplexity
        acc_low = evaluator._perplexity_to_accuracy(1.0)
        assert 0.1 <= acc_low <= 0.99

        # Very high perplexity
        acc_high = evaluator._perplexity_to_accuracy(1000.0)
        assert 0.1 <= acc_high <= 0.99

    def test_evaluate_missing_model(self, tmp_path: Path) -> None:
        """Test evaluate raises error for missing model."""
        evaluator = AccuracyEvaluator()

        with pytest.raises(EvaluationError, match="Model not found"):
            evaluator.evaluate(tmp_path / "nonexistent.gguf")

    def test_evaluate_quick_missing_model(self, tmp_path: Path) -> None:
        """Test evaluate_quick raises error for missing model."""
        evaluator = AccuracyEvaluator()

        with pytest.raises(EvaluationError, match="Model not found"):
            evaluator.evaluate_quick(tmp_path / "nonexistent.gguf")
