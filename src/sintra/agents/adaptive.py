"""Adaptive learning module for improving compression estimates.

Analyzes historical experiment data from the SQLite persistence layer
to build better heuristics for predicting compression impact. Over time,
as more experiments are run, the estimates become more accurate.
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class AdaptiveLearner:
    """Learns from historical experiments to improve compression estimates.

    This class queries the persistence layer for past experiments and
    builds calibrated prediction models for:
    - Accuracy loss at different bit widths
    - TPS gains from quantization and pruning
    - Memory footprint predictions

    Example:
        >>> learner = AdaptiveLearner()
        >>> correction = learner.get_accuracy_correction("llama", 4)
        >>> estimated_accuracy = base_accuracy * (1 - correction)
    """

    def __init__(self, min_samples: int = 3):
        """Initialize the adaptive learner.

        Args:
            min_samples: Minimum samples needed before overriding defaults
        """
        self.min_samples = min_samples
        self._cache: dict[str, Any] = {}

    def get_accuracy_correction(
        self,
        model_family: str,
        bits: int,
        default_loss: float = 0.04,
    ) -> float:
        """Get a calibrated accuracy loss estimate from history.

        Args:
            model_family: Model family name (e.g., "llama", "mistral")
            bits: Target quantization bits
            default_loss: Default accuracy loss if no history

        Returns:
            Estimated accuracy loss (0.0 to 1.0)
        """
        cache_key = f"acc_{model_family}_{bits}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            from sintra.persistence import get_history_db

            db = get_history_db()
            experiments = db.find_similar_experiments(
                model_id=f"%{model_family}%",
                successful_only=True,
                limit=50,
            )

            # Filter for matching bit width
            matching = [
                e for e in experiments
                if e.recipe.bits == bits and e.result.was_successful
            ]

            if len(matching) < self.min_samples:
                return default_loss

            # Calculate average accuracy (lower accuracy = higher loss)
            accuracies = [e.result.accuracy_score for e in matching]
            avg_accuracy = sum(accuracies) / len(accuracies)

            # Estimate loss from baseline (assume 0.9 baseline)
            estimated_loss = max(0.0, 0.9 - avg_accuracy)

            self._cache[cache_key] = estimated_loss
            return estimated_loss

        except Exception as e:
            logger.debug(f"Adaptive accuracy lookup failed: {e}")
            return default_loss

    def get_tps_estimate(
        self,
        model_family: str,
        bits: int,
        pruning_ratio: float = 0.0,
        default_tps: float = 15.0,
    ) -> tuple[float, float]:
        """Get a calibrated TPS estimate from history.

        Args:
            model_family: Model family name
            bits: Target quantization bits
            pruning_ratio: Pruning ratio (0.0 to 1.0)
            default_tps: Default TPS if no history

        Returns:
            Tuple of (min_tps, max_tps) estimated range
        """
        cache_key = f"tps_{model_family}_{bits}_{pruning_ratio:.2f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            from sintra.persistence import get_history_db

            db = get_history_db()
            experiments = db.find_similar_experiments(
                model_id=f"%{model_family}%",
                successful_only=True,
                limit=50,
            )

            # Filter for similar configurations
            matching = [
                e for e in experiments
                if e.recipe.bits == bits
                and abs(e.recipe.pruning_ratio - pruning_ratio) < 0.1
                and e.result.was_successful
                and e.result.actual_tps > 0
            ]

            if len(matching) < self.min_samples:
                variance = default_tps * 0.2
                return (default_tps - variance, default_tps + variance)

            tps_values = [e.result.actual_tps for e in matching]
            avg_tps = sum(tps_values) / len(tps_values)
            # Standard deviation for range
            if len(tps_values) > 1:
                variance = math.sqrt(
                    sum((t - avg_tps) ** 2 for t in tps_values) / (len(tps_values) - 1)
                )
            else:
                variance = avg_tps * 0.15

            result = (
                round(max(0, avg_tps - variance), 1),
                round(avg_tps + variance, 1),
            )
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.debug(f"Adaptive TPS lookup failed: {e}")
            variance = default_tps * 0.2
            return (default_tps - variance, default_tps + variance)

    def get_size_estimate(
        self,
        model_family: str,
        bits: int,
        default_size_gb: float = 4.0,
    ) -> float:
        """Get a calibrated model size estimate from history.

        Args:
            model_family: Model family name
            bits: Target quantization bits
            default_size_gb: Default size estimate if no history

        Returns:
            Estimated model size in GB
        """
        cache_key = f"size_{model_family}_{bits}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            from sintra.persistence import get_history_db

            db = get_history_db()
            experiments = db.find_similar_experiments(
                model_id=f"%{model_family}%",
                successful_only=True,
                limit=50,
            )

            matching = [
                e for e in experiments
                if e.recipe.bits == bits
                and e.result.was_successful
                and e.result.actual_vram_usage > 0
            ]

            if len(matching) < self.min_samples:
                return default_size_gb

            sizes = [e.result.actual_vram_usage for e in matching]
            avg_size = sum(sizes) / len(sizes)

            self._cache[cache_key] = round(avg_size, 2)
            return round(avg_size, 2)

        except Exception as e:
            logger.debug(f"Adaptive size lookup failed: {e}")
            return default_size_gb

    def get_best_starting_config(
        self,
        model_family: str,
        hardware_name: str,
    ) -> dict[str, Any] | None:
        """Get the best known starting configuration from history.

        Finds the most successful configuration for similar models
        on similar hardware.

        Args:
            model_family: Model family name
            hardware_name: Target hardware profile name

        Returns:
            Best known config dict or None if no history
        """
        try:
            from sintra.persistence import get_history_db

            db = get_history_db()

            # Try exact model+hardware match first
            best = db.get_best_recipe_for_hardware(
                model_id=f"%{model_family}%",
                hardware_name=hardware_name,
            )

            if best:
                recipe, result = best
                return {
                    "bits": recipe.bits,
                    "pruning_ratio": recipe.pruning_ratio,
                    "layers_to_drop": recipe.layers_to_drop,
                    "expected_tps": result.actual_tps,
                    "expected_accuracy": result.accuracy_score,
                    "source": "history_exact_match",
                    "confidence": 0.9,
                }

            # Fall back to model match only
            experiments = db.find_similar_experiments(
                model_id=f"%{model_family}%",
                successful_only=True,
                limit=10,
            )

            if experiments:
                # Pick the one with best accuracy
                best_exp = max(experiments, key=lambda e: e.result.accuracy_score)
                return {
                    "bits": best_exp.recipe.bits,
                    "pruning_ratio": best_exp.recipe.pruning_ratio,
                    "layers_to_drop": best_exp.recipe.layers_to_drop,
                    "expected_tps": best_exp.result.actual_tps,
                    "expected_accuracy": best_exp.result.accuracy_score,
                    "source": "history_model_match",
                    "confidence": 0.7,
                }

            return None

        except Exception as e:
            logger.debug(f"Best config lookup failed: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the internal prediction cache."""
        self._cache.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the adaptive learning data.

        Returns:
            Dict with stats about available history data
        """
        try:
            from sintra.persistence import get_history_db

            db = get_history_db()
            stats = db.get_statistics()
            return {
                "has_data": stats["total_experiments"] > 0,
                "total_experiments": stats["total_experiments"],
                "successful_experiments": stats["successful_experiments"],
                "success_rate": stats["success_rate"],
                "unique_models": stats["unique_models"],
                "cache_size": len(self._cache),
                "min_samples_threshold": self.min_samples,
            }
        except Exception:
            return {
                "has_data": False,
                "total_experiments": 0,
                "cache_size": len(self._cache),
            }


# Global adaptive learner instance
_global_learner: AdaptiveLearner | None = None


def get_adaptive_learner() -> AdaptiveLearner:
    """Get the global AdaptiveLearner instance."""
    global _global_learner
    if _global_learner is None:
        _global_learner = AdaptiveLearner()
    return _global_learner


def enhance_estimate_with_history(
    model_family: str,
    bits: int,
    pruning_ratio: float,
    base_estimate: dict[str, Any],
) -> dict[str, Any]:
    """Enhance a compression estimate with adaptive learning data.

    Takes a base estimate (from heuristics) and improves it using
    historical data when available.

    Args:
        model_family: Model family name
        bits: Target quantization bits
        pruning_ratio: Pruning ratio
        base_estimate: Base estimate dict from heuristic formula

    Returns:
        Enhanced estimate with history-calibrated values
    """
    learner = get_adaptive_learner()
    stats = learner.get_statistics()

    if not stats.get("has_data"):
        # No history available, return base estimate as-is
        base_estimate["adaptive_learning"] = "no_history_available"
        return base_estimate

    # Try to calibrate each metric
    calibrated = dict(base_estimate)

    # Calibrate accuracy
    hist_accuracy_loss = learner.get_accuracy_correction(
        model_family, bits, base_estimate.get("estimated_accuracy_loss", 0.04)
    )
    if hist_accuracy_loss != base_estimate.get("estimated_accuracy_loss"):
        calibrated["estimated_accuracy_loss"] = hist_accuracy_loss
        calibrated["accuracy_source"] = "adaptive_learning"

    # Calibrate TPS
    hist_tps = learner.get_tps_estimate(
        model_family, bits, pruning_ratio,
        sum(base_estimate.get("estimated_tps_range", (15, 25))) / 2,
    )
    base_tps = base_estimate.get("estimated_tps_range", (15, 25))
    if hist_tps != base_tps:
        calibrated["estimated_tps_range"] = hist_tps
        calibrated["tps_source"] = "adaptive_learning"

    # Calibrate size
    hist_size = learner.get_size_estimate(
        model_family, bits, base_estimate.get("estimated_size_gb", 4.0)
    )
    if hist_size != base_estimate.get("estimated_size_gb"):
        calibrated["estimated_size_gb"] = hist_size
        calibrated["size_source"] = "adaptive_learning"

    # Boost confidence if we have history
    base_confidence = base_estimate.get("confidence", 0.5)
    num_experiments = stats.get("total_experiments", 0)
    confidence_boost = min(0.2, num_experiments * 0.01)
    calibrated["confidence"] = min(0.95, base_confidence + confidence_boost)

    calibrated["adaptive_learning"] = "applied"
    calibrated["history_experiments"] = num_experiments

    return calibrated
