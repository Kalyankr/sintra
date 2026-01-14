"""Tests for progress tracking module."""

import pytest

from sintra.ui.progress import (
    CallbackProgressReporter,
    ConsoleProgressReporter,
    ProgressInfo,
    ProgressStage,
    SilentProgressReporter,
    get_global_reporter,
    report_complete,
    report_error,
    report_progress,
    set_global_reporter,
)


class TestProgressInfo:
    """Tests for ProgressInfo dataclass."""

    def test_percentage_calculation(self):
        """Should calculate percentage correctly."""
        info = ProgressInfo(stage=ProgressStage.DOWNLOAD, current=50, total=100)
        assert info.percentage == 50.0

    def test_percentage_zero_total(self):
        """Should handle zero total gracefully."""
        info = ProgressInfo(stage=ProgressStage.DOWNLOAD, current=10, total=0)
        assert info.percentage == 0.0

    def test_percentage_caps_at_100(self):
        """Should cap percentage at 100%."""
        info = ProgressInfo(stage=ProgressStage.DOWNLOAD, current=150, total=100)
        assert info.percentage == 100.0

    def test_is_complete(self):
        """Should detect completion correctly."""
        info = ProgressInfo(stage=ProgressStage.DOWNLOAD, current=100, total=100)
        assert info.is_complete is True

        info = ProgressInfo(stage=ProgressStage.DOWNLOAD, current=50, total=100)
        assert info.is_complete is False


class TestProgressStage:
    """Tests for ProgressStage enum."""

    def test_all_stages_exist(self):
        """Should have all expected stages."""
        assert ProgressStage.DOWNLOAD == "download"
        assert ProgressStage.QUANTIZE == "quantize"
        assert ProgressStage.PRUNE == "prune"
        assert ProgressStage.EVALUATE == "evaluate"
        assert ProgressStage.EXPORT == "export"
        assert ProgressStage.BENCHMARK == "benchmark"


class TestSilentProgressReporter:
    """Tests for SilentProgressReporter."""

    def test_update_does_nothing(self):
        """Should not raise on update."""
        reporter = SilentProgressReporter()
        reporter.update(
            ProgressInfo(stage=ProgressStage.DOWNLOAD, current=50, total=100)
        )

    def test_complete_does_nothing(self):
        """Should not raise on complete."""
        reporter = SilentProgressReporter()
        reporter.complete(ProgressStage.DOWNLOAD, "Done")

    def test_error_does_nothing(self):
        """Should not raise on error."""
        reporter = SilentProgressReporter()
        reporter.error(ProgressStage.DOWNLOAD, "Failed")


class TestCallbackProgressReporter:
    """Tests for CallbackProgressReporter."""

    def test_calls_callback_on_update(self):
        """Should call callback on update."""
        calls = []
        reporter = CallbackProgressReporter(lambda info: calls.append(info))

        info = ProgressInfo(stage=ProgressStage.DOWNLOAD, current=50, total=100)
        reporter.update(info)

        assert len(calls) == 1
        assert calls[0] == info

    def test_calls_callback_on_complete(self):
        """Should call callback on complete."""
        calls = []
        reporter = CallbackProgressReporter(lambda info: calls.append(info))

        reporter.complete(ProgressStage.DOWNLOAD, "Done")

        assert len(calls) == 1
        assert calls[0].stage == ProgressStage.DOWNLOAD
        assert calls[0].current == 100
        assert calls[0].is_complete is True

    def test_calls_callback_on_error(self):
        """Should call callback on error."""
        calls = []
        reporter = CallbackProgressReporter(lambda info: calls.append(info))

        reporter.error(ProgressStage.DOWNLOAD, "Network failure")

        assert len(calls) == 1
        assert calls[0].stage == ProgressStage.DOWNLOAD
        assert "Network failure" in calls[0].message


class TestGlobalReporter:
    """Tests for global reporter functions."""

    def test_default_is_silent(self):
        """Should return SilentProgressReporter by default."""
        set_global_reporter(None)
        reporter = get_global_reporter()
        assert isinstance(reporter, SilentProgressReporter)

    def test_can_set_custom_reporter(self):
        """Should allow setting custom reporter."""
        custom = CallbackProgressReporter(lambda x: None)
        set_global_reporter(custom)
        assert get_global_reporter() is custom

        # Reset for other tests
        set_global_reporter(None)

    def test_report_progress_uses_global(self):
        """report_progress should use global reporter."""
        calls = []
        custom = CallbackProgressReporter(lambda info: calls.append(info))
        set_global_reporter(custom)

        report_progress(ProgressStage.DOWNLOAD, 50, 100, "Halfway")

        assert len(calls) == 1
        assert calls[0].stage == ProgressStage.DOWNLOAD
        assert calls[0].percentage == 50.0

        set_global_reporter(None)

    def test_report_complete_uses_global(self):
        """report_complete should use global reporter."""
        calls = []
        custom = CallbackProgressReporter(lambda info: calls.append(info))
        set_global_reporter(custom)

        report_complete(ProgressStage.DOWNLOAD, "All done")

        assert len(calls) == 1
        assert calls[0].is_complete is True

        set_global_reporter(None)

    def test_report_error_uses_global(self):
        """report_error should use global reporter."""
        calls = []
        custom = CallbackProgressReporter(lambda info: calls.append(info))
        set_global_reporter(custom)

        report_error(ProgressStage.DOWNLOAD, "Failed!")

        assert len(calls) == 1
        assert "Failed!" in calls[0].message

        set_global_reporter(None)


class TestConsoleProgressReporter:
    """Tests for ConsoleProgressReporter."""

    def test_can_instantiate(self):
        """Should be able to create instance."""
        reporter = ConsoleProgressReporter()
        assert reporter is not None

    def test_has_stage_names(self):
        """Should have names for all stages."""
        reporter = ConsoleProgressReporter()
        assert len(reporter._stage_names) == len(ProgressStage)
