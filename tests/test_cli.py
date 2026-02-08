"""Tests for CLI argument parsing."""

import sys
from unittest.mock import patch

import pytest


class TestCliArgs:
    """Tests for CLI argument parsing."""

    def test_auto_detect_is_default(self):
        """Auto-detect should be enabled by default (no flags needed)."""
        from sintra.cli import parse_args

        with patch.object(sys, "argv", ["sintra"]):
            args = parse_args()
            assert args.auto_detect is True
            assert args.profile is None

    def test_no_auto_detect_requires_profile(self):
        """Should require profile when --no-auto-detect is used."""
        from sintra.cli import parse_args

        with (
            patch.object(sys, "argv", ["sintra", "--no-auto-detect"]),
            pytest.raises(SystemExit),
        ):
            parse_args()

    def test_profile_optional_with_auto_detect(self):
        """Should not require profile when auto-detect is enabled (default)."""
        from sintra.cli import parse_args

        with patch.object(sys, "argv", ["sintra"]):
            args = parse_args()
            assert args.auto_detect is True
            assert args.profile is None

    def test_output_dir_flag(self):
        """Should parse --output-dir flag."""
        from sintra.cli import parse_args

        with patch.object(sys, "argv", ["sintra", "--output-dir", "/tmp/models"]):
            args = parse_args()
            assert args.output_dir == "/tmp/models"

    def test_dry_run_flag(self):
        """Should parse --dry-run flag."""
        from sintra.cli import parse_args

        with patch.object(sys, "argv", ["sintra", "--dry-run"]):
            args = parse_args()
            assert args.dry_run is True

    def test_target_tps_flag(self):
        """Should parse --target-tps flag."""
        from sintra.cli import parse_args

        with patch.object(sys, "argv", ["sintra", "--target-tps", "50.0"]):
            args = parse_args()
            assert args.target_tps == 50.0

    def test_target_accuracy_flag(self):
        """Should parse --target-accuracy flag."""
        from sintra.cli import parse_args

        with patch.object(sys, "argv", ["sintra", "--target-accuracy", "0.8"]):
            args = parse_args()
            assert args.target_accuracy == 0.8

    def test_default_values(self):
        """Should have sensible defaults with zero flags."""
        from sintra.cli import parse_args

        with patch.object(sys, "argv", ["sintra"]):
            args = parse_args()
            # Auto-detect is now default
            assert args.auto_detect is True
            # Baseline comparison is now default
            assert args.baseline is True
            # Standard defaults
            assert args.backend == "gguf"
            assert args.output_dir is None
            assert args.dry_run is False
            assert args.target_tps == 30.0
            assert args.target_accuracy == 0.65
            assert args.max_iters == 10

    def test_all_flags_together(self):
        """Should handle all flags together."""
        from sintra.cli import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "sintra",
                "--dry-run",
                "--output-dir",
                "/tmp/out",
                "--target-tps",
                "100",
                "--target-accuracy",
                "0.9",
                "--backend",
                "bnb",
                "--max-iters",
                "20",
            ],
        ):
            args = parse_args()
            assert args.auto_detect is True  # Default
            assert args.baseline is True  # Default
            assert args.dry_run is True
            assert args.output_dir == "/tmp/out"
            assert args.target_tps == 100.0
            assert args.target_accuracy == 0.9
            assert args.backend == "bnb"
            assert args.max_iters == 20

    def test_no_baseline_flag(self):
        """Should disable baseline with --no-baseline flag."""
        from sintra.cli import parse_args

        with patch.object(sys, "argv", ["sintra", "--no-baseline"]):
            args = parse_args()
            assert args.baseline is False

    def test_profile_with_flags(self):
        """Should accept profile with other flags."""
        from sintra.cli import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "sintra",
                "profiles/test.yaml",
                "--dry-run",
                "--output-dir",
                "/tmp/out",
            ],
        ):
            args = parse_args()
            assert args.profile == "profiles/test.yaml"
            assert args.dry_run is True
            assert args.output_dir == "/tmp/out"

    def test_resume_flag_latest(self):
        """Should parse --resume with no argument as 'latest'."""
        from sintra.cli import parse_args

        with patch.object(sys, "argv", ["sintra", "--auto-detect", "--resume"]):
            args = parse_args()
            assert args.resume == "latest"

    def test_resume_flag_with_run_id(self):
        """Should parse --resume with a specific run_id."""
        from sintra.cli import parse_args

        with patch.object(
            sys, "argv", ["sintra", "--auto-detect", "--resume", "abc-123"]
        ):
            args = parse_args()
            assert args.resume == "abc-123"

    def test_list_checkpoints_flag(self):
        """Should parse --list-checkpoints flag."""
        from sintra.cli import parse_args

        with patch.object(
            sys, "argv", ["sintra", "--auto-detect", "--list-checkpoints"]
        ):
            args = parse_args()
            assert args.list_checkpoints is True

    def test_resume_default_none(self):
        """Resume should default to None."""
        from sintra.cli import parse_args

        with patch.object(sys, "argv", ["sintra", "--auto-detect"]):
            args = parse_args()
            assert args.resume is None
            assert args.list_checkpoints is False
