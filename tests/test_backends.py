"""Tests for advanced quantization backends (bitsandbytes, ONNX/optimum)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestBitsAndBytesQuantizer:
    """Tests for BitsAndBytesQuantizer."""
    
    def test_bnb_quantizer_raises_without_deps(self):
        """Test that BnB raises appropriate error when deps not installed."""
        from sintra.compression.bnb_quantizer import is_bnb_available, BitsAndBytesError
        
        if not is_bnb_available():
            from sintra.compression.bnb_quantizer import BitsAndBytesQuantizer
            with pytest.raises(BitsAndBytesError):
                BitsAndBytesQuantizer()
    
    def test_bnb_quant_types_available(self):
        """Test that BnBQuantType enum values are available."""
        from sintra.compression.bnb_quantizer import BnBQuantType
        
        assert BnBQuantType.NF4 == "nf4"
        assert BnBQuantType.FP4 == "fp4"
        assert BnBQuantType.INT8 == "int8"


class TestONNXOptimizer:
    """Tests for ONNXOptimizer."""
    
    def test_onnx_optimizer_raises_without_deps(self):
        """Test that ONNX raises appropriate error when deps not installed."""
        from sintra.compression.onnx_optimizer import is_onnx_available, ONNXOptimizerError
        
        if not is_onnx_available():
            from sintra.compression.onnx_optimizer import ONNXOptimizer
            with pytest.raises(ONNXOptimizerError):
                ONNXOptimizer()
    
    def test_optimization_levels_available(self):
        """Test that OptimizationLevel enum values are available."""
        from sintra.compression.onnx_optimizer import OptimizationLevel
        
        assert OptimizationLevel.DISABLE == "disable"
        assert OptimizationLevel.BASIC == "basic"
        assert OptimizationLevel.EXTENDED == "extended"
        assert OptimizationLevel.ALL == "all"
    
    def test_quantization_modes_available(self):
        """Test that QuantizationMode enum values are available."""
        from sintra.compression.onnx_optimizer import QuantizationMode
        
        assert QuantizationMode.DYNAMIC == "dynamic"
        assert QuantizationMode.STATIC == "static"


class TestCompressionModuleExports:
    """Test that compression module exports core backends."""
    
    def test_core_exports_available(self):
        """Test core exports are in __all__."""
        from sintra import compression
        
        expected_exports = [
            # Core (always available)
            "ModelDownloader",
            "download_model",
            "GGUFQuantizer",
            "QuantizationType",
            "quantize_model",
            "quantize_with_compression",
            "LayerDropper",
            "StructuredPruner",
            "PruningError",
            "drop_layers",
            "prune_model",
            "apply_compression",
            "AccuracyEvaluator",
            "evaluate_perplexity",
        ]
        
        for export in expected_exports:
            assert export in compression.__all__, f"Missing export: {export}"
    
    def test_can_import_core_exports(self):
        """Test core exports can be imported."""
        from sintra.compression import (
            ModelDownloader,
            GGUFQuantizer,
            LayerDropper,
            StructuredPruner,
            AccuracyEvaluator,
        )
        
        # All core imports should succeed
        assert ModelDownloader is not None
        assert GGUFQuantizer is not None
        assert LayerDropper is not None


class TestBackendAvailability:
    """Test backend availability detection."""
    
    def test_bnb_availability_is_bool(self):
        """Test is_bnb_available returns boolean."""
        from sintra.compression.bnb_quantizer import is_bnb_available
        assert isinstance(is_bnb_available(), bool)
    
    def test_onnx_availability_is_bool(self):
        """Test is_onnx_available returns boolean."""
        from sintra.compression.onnx_optimizer import is_onnx_available
        assert isinstance(is_onnx_available(), bool)


class TestBnBQuantizerUnit:
    """Unit tests for BitsAndBytesQuantizer with mocked dependencies."""
    
    @pytest.fixture
    def mock_bnb_available(self):
        """Mock bitsandbytes as available."""
        with patch("sintra.compression.bnb_quantizer._check_bnb_available", return_value=True), \
             patch("sintra.compression.bnb_quantizer._check_cuda_available", return_value=True):
            yield
    
    def test_init_creates_cache_dir(self, mock_bnb_available, tmp_path):
        """Test that init creates cache directory."""
        from sintra.compression.bnb_quantizer import BitsAndBytesQuantizer
        
        cache_dir = tmp_path / "bnb_cache"
        quantizer = BitsAndBytesQuantizer(cache_dir=cache_dir)
        
        assert cache_dir.exists()
        assert quantizer.cache_dir == cache_dir
    
    def test_quantize_validates_bits(self, mock_bnb_available, tmp_path):
        """Test that quantize validates bit values."""
        from sintra.compression.bnb_quantizer import BitsAndBytesQuantizer
        
        quantizer = BitsAndBytesQuantizer(cache_dir=tmp_path)
        
        with pytest.raises(ValueError, match="4 or 8 bits"):
            quantizer.quantize("model_id", bits=2)
        
        with pytest.raises(ValueError, match="4 or 8 bits"):
            quantizer.quantize("model_id", bits=16)


class TestONNXOptimizerUnit:
    """Unit tests for ONNXOptimizer with mocked dependencies."""
    
    @pytest.fixture
    def mock_optimum_available(self):
        """Mock optimum as available."""
        with patch("sintra.compression.onnx_optimizer._check_optimum_available", return_value=True), \
             patch("sintra.compression.onnx_optimizer._check_onnxruntime_available", return_value=True):
            yield
    
    def test_init_creates_cache_dir(self, mock_optimum_available, tmp_path):
        """Test that init creates cache directory."""
        from sintra.compression.onnx_optimizer import ONNXOptimizer
        
        cache_dir = tmp_path / "onnx_cache"
        optimizer = ONNXOptimizer(cache_dir=cache_dir)
        
        assert cache_dir.exists()
        assert optimizer.cache_dir == cache_dir
    
    def test_get_cached_models_empty(self, mock_optimum_available, tmp_path):
        """Test get_cached_models returns empty list for new cache."""
        from sintra.compression.onnx_optimizer import ONNXOptimizer
        
        optimizer = ONNXOptimizer(cache_dir=tmp_path)
        assert optimizer.get_cached_models() == []


class TestCLIBackendFlag:
    """Test CLI --backend flag."""
    
    def test_backend_choices(self):
        """Test that CLI accepts valid backend choices."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", "sintra", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/kkr/dev/sintra",
        )
        
        assert "gguf" in result.stdout
        assert "bnb" in result.stdout
        assert "onnx" in result.stdout
