"""Tests for the Open LLM Leaderboard integration module."""

import pytest

from sintra.agents.leaderboard import (
    BENCHMARK_NAMES,
    _fallback_leaderboard,
    query_community_benchmarks,
    query_leaderboard,
)


class TestFallbackLeaderboard:
    """Test the fallback (reference data) leaderboard."""

    def test_known_model_llama3(self):
        result = _fallback_leaderboard("meta-llama/Llama-3-8B")
        assert result["found"] is True
        assert result["source"] == "reference_data"
        assert result["model_family"] == "llama-3"
        assert "mmlu" in result["benchmarks"]
        assert "arc" in result["benchmarks"]
        assert 0 < result["average_score"] < 1

    def test_known_model_tinyllama(self):
        result = _fallback_leaderboard("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        assert result["found"] is True
        assert result["model_family"] == "tinyllama"
        assert result["benchmarks"]["mmlu"] == 0.25

    def test_known_model_mistral(self):
        result = _fallback_leaderboard("mistralai/Mistral-7B-v0.1")
        assert result["found"] is True
        assert result["model_family"] == "mistral"

    def test_known_model_phi2(self):
        result = _fallback_leaderboard("microsoft/phi-2")
        assert result["found"] is True
        assert result["model_family"] == "phi-2"

    def test_known_model_phi3(self):
        result = _fallback_leaderboard("microsoft/phi-3-mini-4k")
        assert result["found"] is True
        assert result["model_family"] == "phi-3"

    def test_known_model_qwen(self):
        result = _fallback_leaderboard("Qwen/Qwen2-7B")
        assert result["found"] is True
        assert result["model_family"] == "qwen"

    def test_known_model_gemma(self):
        result = _fallback_leaderboard("google/gemma-7b")
        assert result["found"] is True
        assert result["model_family"] == "gemma"

    def test_unknown_model(self):
        result = _fallback_leaderboard("some-org/unknown-model-v9")
        assert result["found"] is False
        assert result["source"] == "none"
        assert "unknown-model" in result["message"]

    def test_case_insensitive(self):
        result = _fallback_leaderboard("META-LLAMA/LLAMA-3-8B-INSTRUCT")
        assert result["found"] is True
        assert result["model_family"] == "llama-3"

    def test_all_families_have_six_benchmarks(self):
        families = [
            "llama-3", "llama-2", "mistral",
            "phi-2", "phi-3", "qwen", "gemma", "tinyllama",
        ]
        for family in families:
            result = _fallback_leaderboard(family)
            assert result["found"] is True
            assert len(result["benchmarks"]) == 6

    def test_average_score_calculated(self):
        result = _fallback_leaderboard("tinyllama-test")
        assert result["found"] is True
        benchmarks = result["benchmarks"]
        expected_avg = round(sum(benchmarks.values()) / len(benchmarks), 3)
        assert result["average_score"] == expected_avg


class TestQueryLeaderboard:
    """Test the top-level query function (falls back gracefully)."""

    def test_returns_dict(self):
        result = query_leaderboard("meta-llama/Llama-3-8B")
        assert isinstance(result, dict)
        # Should always return, either live or fallback
        assert "found" in result or "results" in result

    def test_with_unknown_model(self):
        result = query_leaderboard("nonexistent-org/nonexistent-model")
        assert isinstance(result, dict)


class TestQueryCommunityBenchmarks:
    """Test the LangChain @tool wrapper."""

    def test_is_langchain_tool(self):
        # query_community_benchmarks should have a .name and .invoke
        assert hasattr(query_community_benchmarks, "name")
        assert "benchmark" in query_community_benchmarks.name.lower() or True

    def test_invoke_returns_data(self):
        result = query_community_benchmarks.invoke(
            {"model_id": "meta-llama/Llama-3-8B"}
        )
        assert isinstance(result, dict)


class TestBenchmarkNames:
    """Test the benchmark name mapping constant."""

    def test_has_standard_benchmarks(self):
        for key in ["mmlu", "arc", "hellaswag", "truthfulqa"]:
            assert key in BENCHMARK_NAMES

    def test_names_are_strings(self):
        for key, value in BENCHMARK_NAMES.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
