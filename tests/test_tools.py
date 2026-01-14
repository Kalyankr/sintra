"""Tests for the Sintra agent tools."""

import pytest

from sintra.agents.tools import (
    estimate_compression_impact,
    get_architect_tools,
    lookup_quantization_benchmarks,
    query_hardware_capabilities,
    search_similar_models,
)


class TestSearchSimilarModels:
    """Tests for the search_similar_models tool."""

    def test_returns_results_for_llama(self):
        """Should return results for Llama models."""
        results = search_similar_models.invoke({
            "base_model": "meta-llama/Llama-3.2-1B",
            "task": "text-generation",
            "max_results": 3,
        })
        
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all("model_id" in r for r in results)
        assert any("llama" in r["model_id"].lower() for r in results)

    def test_returns_results_for_mistral(self):
        """Should return results for Mistral models."""
        results = search_similar_models.invoke({
            "base_model": "mistralai/Mistral-7B",
            "max_results": 2,
        })
        
        assert isinstance(results, list)
        assert any("mistral" in r["model_id"].lower() for r in results)

    def test_returns_generic_for_unknown_model(self):
        """Should return generic results for unknown models."""
        results = search_similar_models.invoke({
            "base_model": "unknown/model",
            "max_results": 1,
        })
        
        assert isinstance(results, list)
        assert len(results) >= 1


class TestEstimateCompressionImpact:
    """Tests for the estimate_compression_impact tool."""

    def test_basic_estimation(self):
        """Should return valid estimation."""
        result = estimate_compression_impact.invoke({
            "model_size_billions": 7.0,
            "target_bits": 4,
            "pruning_ratio": 0.1,
            "layers_to_drop": 0,
            "total_layers": 32,
        })
        
        assert "estimated_size_gb" in result
        assert "estimated_tps_range" in result
        assert "estimated_accuracy_loss" in result
        assert "confidence" in result
        assert "reasoning" in result
        
        # Size should be smaller than FP16
        assert result["estimated_size_gb"] < 14.0  # 7B * 2 bytes

    def test_aggressive_compression_has_lower_confidence(self):
        """Aggressive settings should have lower confidence."""
        conservative = estimate_compression_impact.invoke({
            "model_size_billions": 7.0,
            "target_bits": 8,
            "pruning_ratio": 0.0,
        })
        
        aggressive = estimate_compression_impact.invoke({
            "model_size_billions": 7.0,
            "target_bits": 2,
            "pruning_ratio": 0.4,
            "layers_to_drop": 10,
            "total_layers": 32,
        })
        
        assert aggressive["confidence"] < conservative["confidence"]

    def test_layer_dropping_reduces_size(self):
        """Dropping layers should reduce estimated size."""
        without_drop = estimate_compression_impact.invoke({
            "model_size_billions": 7.0,
            "target_bits": 4,
            "pruning_ratio": 0.0,
            "layers_to_drop": 0,
        })
        
        with_drop = estimate_compression_impact.invoke({
            "model_size_billions": 7.0,
            "target_bits": 4,
            "pruning_ratio": 0.0,
            "layers_to_drop": 8,
            "total_layers": 32,
        })
        
        assert with_drop["estimated_size_gb"] < without_drop["estimated_size_gb"]


class TestQueryHardwareCapabilities:
    """Tests for the query_hardware_capabilities tool."""

    def test_low_memory_device(self):
        """Should recommend aggressive quantization for low memory."""
        result = query_hardware_capabilities.invoke({
            "device_name": "Raspberry Pi 5",
            "available_memory_gb": 4.0,
            "has_gpu": False,
        })
        
        assert result["device_name"] == "Raspberry Pi 5"
        assert result["available_vram_gb"] == 4.0
        assert 2 in result["supported_bits"]
        assert 3 in result["supported_bits"]
        assert result["max_model_params_billions"] <= 2.0

    def test_high_memory_device(self):
        """Should support more options for high memory."""
        result = query_hardware_capabilities.invoke({
            "device_name": "Mac Studio M2 Ultra",
            "available_memory_gb": 64.0,
            "has_gpu": True,
            "gpu_type": "Metal",
        })
        
        assert 8 in result["supported_bits"]
        assert result["max_model_params_billions"] > 5.0

    def test_cuda_device(self):
        """CUDA devices should support all quantization."""
        result = query_hardware_capabilities.invoke({
            "device_name": "RTX 4090",
            "available_memory_gb": 24.0,
            "has_gpu": True,
            "gpu_type": "CUDA",
        })
        
        assert 2 in result["supported_bits"]
        assert 8 in result["supported_bits"]
        assert result["recommendations"]["backend"] == "bnb"


class TestLookupQuantizationBenchmarks:
    """Tests for the lookup_quantization_benchmarks tool."""

    def test_known_model_family(self):
        """Should return data for known model families."""
        result = lookup_quantization_benchmarks.invoke({
            "model_family": "llama",
            "bits": 4,
        })
        
        assert result["found"] is True
        assert "benchmark_results" in result
        assert result["model_family"] == "llama"

    def test_unknown_model_family(self):
        """Should return generic estimate for unknown families."""
        result = lookup_quantization_benchmarks.invoke({
            "model_family": "unknown_model",
            "bits": 4,
        })
        
        assert result["found"] is False
        assert "generic_estimate" in result or "message" in result

    def test_different_bit_widths(self):
        """Lower bits should have higher TPS but more accuracy loss."""
        result_4bit = lookup_quantization_benchmarks.invoke({
            "model_family": "mistral",
            "bits": 4,
        })
        
        result_8bit = lookup_quantization_benchmarks.invoke({
            "model_family": "mistral",
            "bits": 8,
        })
        
        # 4-bit should be faster but less accurate
        assert result_4bit["benchmark_results"]["tps_range"][0] > result_8bit["benchmark_results"]["tps_range"][0]
        assert result_4bit["benchmark_results"]["accuracy_drop"] > result_8bit["benchmark_results"]["accuracy_drop"]


class TestGetArchitectTools:
    """Tests for the get_architect_tools function."""

    def test_returns_all_tools(self):
        """Should return all architect tools."""
        tools = get_architect_tools()
        
        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        
        assert "search_similar_models" in tool_names
        assert "estimate_compression_impact" in tool_names
        assert "query_hardware_capabilities" in tool_names
        assert "lookup_quantization_benchmarks" in tool_names

    def test_tools_are_callable(self):
        """All tools should be callable."""
        tools = get_architect_tools()
        
        for tool in tools:
            assert callable(tool.invoke)
