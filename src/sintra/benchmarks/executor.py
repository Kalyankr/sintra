"""Benchmark executors for running model compression experiments."""

import logging
import os
import random
import subprocess
from abc import ABC, abstractmethod

from sintra.profiles.models import ExperimentResult, HardwareProfile, ModelRecipe
from sintra.ui.console import console, log_transition

logger = logging.getLogger(__name__)


class BenchmarkExecutor(ABC):
    """Abstract base class for benchmark executors."""

    @abstractmethod
    def run_benchmark(
        self, recipe: ModelRecipe, profile: HardwareProfile
    ) -> ExperimentResult:
        """Execute a benchmark with the given recipe and hardware profile.

        Args:
            recipe: The compression recipe to test.
            profile: The target hardware profile.

        Returns:
            ExperimentResult with metrics and success status.
        """
        pass


class StandaloneExecutor(BenchmarkExecutor):
    def run_benchmark(
        self, recipe: ModelRecipe, profile: HardwareProfile
    ) -> ExperimentResult:
        log_transition(
            "Lab",
            f"Starting isolated worker: {recipe.bits}-bit | {recipe.method}",
            "lab.node",
        )

        recipe_json = recipe.model_dump_json()
        env = os.environ.copy()
        env["VRAM_LIMIT_GB"] = str(profile.constraints.vram_gb)

        # Find the worker script
        import pathlib

        import sintra

        package_dir = pathlib.Path(sintra.__file__).parent
        worker_script = package_dir / "benchmarks" / "worker" / "runner.py"

        if not worker_script.exists():
            return self._error_result(
                f"Worker script not found at {worker_script}. Please reinstall sintra."
            )

        with console.status(f"[bold green]Running {recipe.method} Surgery...") as _:
            try:
                process = subprocess.Popen(
                    ["uv", "run", str(worker_script)],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Communicate: Send recipe, get results
                stdout, stderr = process.communicate(input=recipe_json, timeout=600)

            except FileNotFoundError:
                return self._error_result(
                    "'uv' command not found. Please install uv: "
                    "curl -LsSf https://astral.sh/uv/install.sh | sh"
                )
            except subprocess.TimeoutExpired:
                process.kill()
                return self._error_result("Process timed out (600s limit)")
            except Exception as e:
                return self._error_result(f"Executor internal error: {str(e)}")

        # Handle Exit Codes & Empty Outputs
        if process.returncode != 0:
            log_transition("Status", "Surgery process crashed.", "status.fail")
            # Log stderr for debugging if it exists
            return self._error_result(
                stderr.strip() if stderr else "Unknown process crash"
            )

        if not stdout or not stdout.strip():
            return self._error_result("Worker completed but returned no data.")

        # Parse JSON Output
        try:
            return ExperimentResult.model_validate_json(stdout)
        except Exception as e:
            return self._error_result(f"Failed to parse worker output: {str(e)}")

    def _error_result(self, msg: str) -> ExperimentResult:
        """Helper to create a failed experiment result."""
        return ExperimentResult(
            actual_tps=0.0,
            actual_vram_usage=0.0,
            accuracy_score=0.0,
            was_successful=False,
            error_log=msg,
        )


class MockExecutor(BenchmarkExecutor):
    """Simulates hardware behavior for testing the Agent's logic loop.

    Uses deterministic calculations with optional random noise for realistic
    simulation of compression trade-offs.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the mock executor.

        Args:
            seed: Random seed for reproducible results. If None, uses system random.
        """
        self._random = random.Random(seed)

    def run_benchmark(
        self, recipe: ModelRecipe, profile: HardwareProfile
    ) -> ExperimentResult:
        """Simulate a benchmark run with realistic mock metrics."""
        # Logic: Lower bits = Higher Speed, but Lower Accuracy
        # Base speed 2.0 TPS, +3.0 TPS if we use 4-bit
        speed_boost = 3.0 if recipe.bits == 4 else 0.0
        prune_boost = recipe.pruning_ratio * 10

        mock_tps = 2.0 + speed_boost + prune_boost + self._random.uniform(-0.5, 0.5)

        # Logic: More pruning/lower bits = Lower Accuracy
        accuracy_penalty = (recipe.pruning_ratio * 0.5) + (
            0.2 if recipe.bits == 4 else 0
        )
        mock_accuracy = max(
            0.1, 0.9 - accuracy_penalty + self._random.uniform(-0.05, 0.05)
        )

        return ExperimentResult(
            actual_tps=round(mock_tps, 2),
            actual_vram_usage=4.2 if recipe.bits == 8 else 2.1,
            accuracy_score=round(mock_accuracy, 2),
            was_successful=True,
        )
