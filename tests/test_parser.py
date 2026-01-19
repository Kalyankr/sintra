"""Tests for the hardware profile parser."""

from pathlib import Path

import pytest
import yaml

from sintra.profiles.models import HardwareProfile
from sintra.profiles.parser import ProfileLoadError, load_hardware_profile


class TestLoadHardwareProfile:
    """Tests for load_hardware_profile function."""

    def test_load_valid_profile(self, tmp_path: Path) -> None:
        """Test loading a valid hardware profile."""
        profile_data = {
            "name": "Test Device",
            "constraints": {"vram_gb": 8.0},
            "targets": {"min_tokens_per_second": 50.0, "min_accuracy_score": 0.7},
        }
        profile_file = tmp_path / "profile.yaml"
        profile_file.write_text(yaml.dump(profile_data))

        result = load_hardware_profile(profile_file)

        assert isinstance(result, HardwareProfile)
        assert result.name == "Test Device"
        assert result.constraints.vram_gb == 8.0
        assert result.targets.min_tokens_per_second == 50.0

    def test_load_profile_with_optional_fields(self, tmp_path: Path) -> None:
        """Test loading a profile with all optional fields."""
        profile_data = {
            "name": "Full Device",
            "constraints": {"vram_gb": 16.0, "cpu_arch": "arm64", "has_cuda": False},
            "targets": {
                "min_tokens_per_second": 80.0,
                "min_accuracy_score": 0.65,
                "max_latency_ms": 300.0,
            },
            "supported_quantizations": ["GGUF", "AWQ"],
        }
        profile_file = tmp_path / "full_profile.yaml"
        profile_file.write_text(yaml.dump(profile_data))

        result = load_hardware_profile(profile_file)

        assert result.constraints.cpu_arch == "arm64"
        assert result.constraints.has_cuda is False
        assert result.targets.max_latency_ms == 300.0
        assert result.supported_quantizations == ["GGUF", "AWQ"]

    def test_file_not_found(self) -> None:
        """Test that missing file raises ProfileLoadError."""
        with pytest.raises(ProfileLoadError, match="Profile file not found"):
            load_hardware_profile("/nonexistent/path/profile.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        """Test that invalid YAML raises ProfileLoadError."""
        profile_file = tmp_path / "invalid.yaml"
        profile_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ProfileLoadError, match="Invalid YAML"):
            load_hardware_profile(profile_file)

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test that empty file raises ProfileLoadError."""
        profile_file = tmp_path / "empty.yaml"
        profile_file.write_text("")

        with pytest.raises(ProfileLoadError, match="Profile file is empty"):
            load_hardware_profile(profile_file)

    def test_non_dict_yaml(self, tmp_path: Path) -> None:
        """Test that non-dict YAML raises ProfileLoadError."""
        profile_file = tmp_path / "list.yaml"
        profile_file.write_text("- item1\n- item2")

        with pytest.raises(ProfileLoadError, match="must be a YAML mapping"):
            load_hardware_profile(profile_file)

    def test_missing_required_field(self, tmp_path: Path) -> None:
        """Test that missing required fields raise ProfileLoadError."""
        profile_data = {
            "name": "Incomplete",
            # Missing constraints and targets
        }
        profile_file = tmp_path / "incomplete.yaml"
        profile_file.write_text(yaml.dump(profile_data))

        with pytest.raises(ProfileLoadError, match="Profile validation failed"):
            load_hardware_profile(profile_file)

    def test_directory_instead_of_file(self, tmp_path: Path) -> None:
        """Test that passing a directory raises ProfileLoadError."""
        with pytest.raises(ProfileLoadError, match="not a file"):
            load_hardware_profile(tmp_path)
