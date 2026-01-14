"""Sintra UI module - Console output and logging utilities."""

from sintra.ui.console import console, log_transition
from sintra.ui.progress import (
    CallbackProgressReporter,
    ConsoleProgressReporter,
    ProgressCallback,
    ProgressInfo,
    ProgressReporter,
    ProgressStage,
    SilentProgressReporter,
    get_global_reporter,
    report_complete,
    report_error,
    report_progress,
    set_global_reporter,
)

__all__ = [
    "CallbackProgressReporter",
    "ConsoleProgressReporter",
    "ProgressCallback",
    "ProgressInfo",
    "ProgressReporter",
    "ProgressStage",
    "SilentProgressReporter",
    "console",
    "get_global_reporter",
    "log_transition",
    "report_complete",
    "report_error",
    "report_progress",
    "set_global_reporter",
]
