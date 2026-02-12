"""Tests for the adaptive learning module."""

from sintra.agents.adaptive import (
    AdaptiveLearner,
    enhance_estimate_with_history,
    get_adaptive_learner,
)


class TestAdaptiveLearner:
    """Test the AdaptiveLearner class."""

    def test_creation_defaults(self):
        learner = AdaptiveLearner()
        assert learner.min_samples == 3
        assert learner._cache == {}

    def test_creation_custom_min_samples(self):
        learner = AdaptiveLearner(min_samples=10)
        assert learner.min_samples == 10

    def test_get_accuracy_correction_default(self):
        learner = AdaptiveLearner()
        # With no history available, should return the default
        correction = learner.get_accuracy_correction("llama", 4, default_loss=0.05)
        assert isinstance(correction, float)
        assert 0.0 <= correction <= 1.0

    def test_get_accuracy_correction_uses_cache(self):
        learner = AdaptiveLearner()
        learner._cache["acc_llama_4"] = 0.123
        correction = learner.get_accuracy_correction("llama", 4)
        assert correction == 0.123

    def test_get_tps_estimate_default(self):
        learner = AdaptiveLearner()
        result = learner.get_tps_estimate("llama", 4, 0.0, default_tps=20.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        tps_low, tps_high = result
        assert isinstance(tps_low, float)
        assert isinstance(tps_high, float)

    def test_get_tps_estimate_uses_cache(self):
        learner = AdaptiveLearner()
        learner._cache["tps_llama_4_0.00"] = (30.0, 45.0)
        result = learner.get_tps_estimate("llama", 4, 0.0)
        assert result == (30.0, 45.0)

    def test_get_size_estimate_default(self):
        learner = AdaptiveLearner()
        result = learner.get_size_estimate("llama", 4, default_size_gb=4.0)
        assert isinstance(result, float)
        assert result > 0

    def test_get_size_estimate_uses_cache(self):
        learner = AdaptiveLearner()
        learner._cache["size_llama_4"] = 3.5
        result = learner.get_size_estimate("llama", 4)
        assert result == 3.5

    def test_get_best_starting_config_no_data(self):
        learner = AdaptiveLearner()
        result = learner.get_best_starting_config("llama", "gguf")
        # With no persistence data, should return None
        assert result is None

    def test_clear_cache(self):
        learner = AdaptiveLearner()
        learner._cache["test_key"] = "test_value"
        learner.clear_cache()
        assert learner._cache == {}

    def test_get_statistics_no_db(self):
        learner = AdaptiveLearner()
        stats = learner.get_statistics()
        assert isinstance(stats, dict)
        assert "cache_size" in stats
        # Without DB, should report no data
        assert stats.get("has_data", False) is False or "total_experiments" in stats


class TestGetAdaptiveLearner:
    """Test the singleton accessor."""

    def test_returns_instance(self):
        learner = get_adaptive_learner()
        assert isinstance(learner, AdaptiveLearner)

    def test_singleton(self):
        a = get_adaptive_learner()
        b = get_adaptive_learner()
        assert a is b


class TestEnhanceEstimateWithHistory:
    """Test the estimate enhancement function."""

    def test_no_history(self):
        base = {
            "estimated_accuracy_loss": 0.04,
            "estimated_tps_range": (15, 25),
            "estimated_size_gb": 4.0,
            "confidence": 0.5,
        }
        result = enhance_estimate_with_history("llama", 4, 0.0, base)
        assert isinstance(result, dict)
        # Should contain adaptive_learning key
        assert "adaptive_learning" in result
        # With no history, values either stay the same or are marked "no_history_available"
        assert result["adaptive_learning"] in ("no_history_available", "applied")

    def test_preserves_base_estimate_keys(self):
        base = {
            "estimated_accuracy_loss": 0.04,
            "estimated_tps_range": (15, 25),
            "estimated_size_gb": 4.0,
            "confidence": 0.5,
            "custom_key": "should_survive",
        }
        result = enhance_estimate_with_history("llama", 4, 0.0, base)
        assert "custom_key" in result
        assert result["custom_key"] == "should_survive"

    def test_returns_dict(self):
        result = enhance_estimate_with_history("mistral", 4, 0.1, {})
        assert isinstance(result, dict)
