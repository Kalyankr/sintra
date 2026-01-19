"""Progress tracking for long-running operations.

This module provides a unified progress callback system for
compression, quantization, and other long-running operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class ProgressStage(str, Enum):
    """Stages of the compression pipeline."""

    DOWNLOAD = "download"
    QUANTIZE = "quantize"
    PRUNE = "prune"
    EVALUATE = "evaluate"
    EXPORT = "export"
    BENCHMARK = "benchmark"


@dataclass
class ProgressInfo:
    """Information about current progress."""

    stage: ProgressStage
    current: int = 0
    total: int = 100
    message: str = ""
    details: dict = field(default_factory=dict)

    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100)

    @property
    def is_complete(self) -> bool:
        """Check if this stage is complete."""
        return self.current >= self.total


# Type alias for progress callbacks
ProgressCallback = Callable[[ProgressInfo], None]


class ProgressReporter(ABC):
    """Abstract base class for progress reporters."""

    @abstractmethod
    def update(self, info: ProgressInfo) -> None:
        """Report progress update."""
        pass

    @abstractmethod
    def complete(self, stage: ProgressStage, message: str = "") -> None:
        """Mark a stage as complete."""
        pass

    @abstractmethod
    def error(self, stage: ProgressStage, error: str) -> None:
        """Report an error in a stage."""
        pass


class ConsoleProgressReporter(ProgressReporter):
    """Rich console-based progress reporter."""

    def __init__(self, show_details: bool = False) -> None:
        self.show_details = show_details
        self._stage_names = {
            ProgressStage.DOWNLOAD: "📥 Downloading",
            ProgressStage.QUANTIZE: "🔢 Quantizing",
            ProgressStage.PRUNE: "✂️  Pruning",
            ProgressStage.EVALUATE: "📊 Evaluating",
            ProgressStage.EXPORT: "📦 Exporting",
            ProgressStage.BENCHMARK: "⚡ Benchmarking",
        }

    def update(self, info: ProgressInfo) -> None:
        """Report progress update to console."""
        from sintra.ui.console import console

        stage_name = self._stage_names.get(info.stage, info.stage.value)
        pct = info.percentage

        # Create progress bar
        bar_width = 20
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        message = f"  {stage_name}: [{bar}] {pct:.0f}%"
        if info.message:
            message += f" - {info.message}"

        # Use carriage return for in-place updates
        console.print(message, end="\r")

        if info.is_complete:
            console.print()  # New line when complete

    def complete(self, stage: ProgressStage, message: str = "") -> None:
        """Mark stage as complete."""
        from sintra.ui.console import console

        stage_name = self._stage_names.get(stage, stage.value)
        msg = f"  {stage_name}: [green]✓ Complete[/green]"
        if message:
            msg += f" - {message}"
        console.print(msg)

    def error(self, stage: ProgressStage, error: str) -> None:
        """Report error in stage."""
        from sintra.ui.console import console

        stage_name = self._stage_names.get(stage, stage.value)
        console.print(f"  {stage_name}: [red]✗ Error[/red] - {error}")


class SilentProgressReporter(ProgressReporter):
    """No-op progress reporter for quiet mode."""

    def update(self, info: ProgressInfo) -> None:
        pass

    def complete(self, stage: ProgressStage, message: str = "") -> None:
        pass

    def error(self, stage: ProgressStage, error: str) -> None:
        pass


class CallbackProgressReporter(ProgressReporter):
    """Progress reporter that calls a user-provided callback."""

    def __init__(self, callback: ProgressCallback) -> None:
        self.callback = callback

    def update(self, info: ProgressInfo) -> None:
        self.callback(info)

    def complete(self, stage: ProgressStage, message: str = "") -> None:
        self.callback(
            ProgressInfo(
                stage=stage,
                current=100,
                total=100,
                message=message or "Complete",
            )
        )

    def error(self, stage: ProgressStage, error: str) -> None:
        self.callback(
            ProgressInfo(
                stage=stage,
                current=0,
                total=100,
                message=f"Error: {error}",
                details={"error": error},
            )
        )


# Global progress reporter (can be set by CLI/main)
_global_reporter: ProgressReporter | None = None


def set_global_reporter(reporter: ProgressReporter | None) -> None:
    """Set the global progress reporter."""
    global _global_reporter
    _global_reporter = reporter


def get_global_reporter() -> ProgressReporter:
    """Get the global progress reporter (silent if not set)."""
    if _global_reporter is None:
        return SilentProgressReporter()
    return _global_reporter


def report_progress(
    stage: ProgressStage,
    current: int,
    total: int,
    message: str = "",
    **details: any,
) -> None:
    """Convenience function to report progress using global reporter."""
    reporter = get_global_reporter()
    reporter.update(
        ProgressInfo(
            stage=stage,
            current=current,
            total=total,
            message=message,
            details=dict(details),
        )
    )


def report_complete(stage: ProgressStage, message: str = "") -> None:
    """Convenience function to mark stage complete."""
    reporter = get_global_reporter()
    reporter.complete(stage, message)


def report_error(stage: ProgressStage, error: str) -> None:
    """Convenience function to report error."""
    reporter = get_global_reporter()
    reporter.error(stage, error)
