"""Tests for hardware auto-detection module."""

import platform
from unittest.mock import MagicMock, patch

import pytest

from sintra.profiles.hardware import (
    auto_detect_hardware,
    detect_cpu_arch,
    detect_cuda,
    detect_supported_quantizations,
    detect_system_name,
    get_gpu_vram_gb,
)


class TestDetectCuda:
    """Tests for CUDA detection."""

    def test_no_nvidia_smi(self):
        """Should return False when nvidia-smi is not found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            assert detect_cuda() is False

    def test_nvidia_smi_success(self):
        """Should return True when nvidia-smi succeeds."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert detect_cuda() is True

    def test_nvidia_smi_failure(self):
        """Should return False when nvidia-smi fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert detect_cuda() is False


class TestDetectCpuArch:
    """Tests for CPU architecture detection."""

    def test_detects_arm64(self):
        """Should detect ARM64 architecture."""
        with patch("platform.machine", return_value="aarch64"):
            assert detect_cpu_arch() == "arm64"

        with patch("platform.machine", return_value="arm64"):
            assert detect_cpu_arch() == "arm64"

    def test_detects_x86_64(self):
        """Should detect x86_64 architecture."""
        with patch("platform.machine", return_value="x86_64"):
            assert detect_cpu_arch() == "x86_64"

        with patch("platform.machine", return_value="AMD64"):
            assert detect_cpu_arch() == "x86_64"

    def test_fallback_for_unknown(self):
        """Should return raw machine name for unknown architectures."""
        with patch("platform.machine", return_value="riscv64"):
            assert detect_cpu_arch() == "riscv64"


class TestDetectSystemName:
    """Tests for system name detection."""

    def test_includes_cpu_count_and_ram(self):
        """Should include CPU count and RAM in name."""
        with patch("platform.system", return_value="Linux"):
            with patch("psutil.cpu_count", return_value=8):
                with patch("psutil.virtual_memory") as mock_mem:
                    mock_mem.return_value.total = 16 * 1024**3  # 16GB
                    name = detect_system_name()
                    assert "8 cores" in name
                    assert "16GB" in name


class TestDetectSupportedQuantizations:
    """Tests for quantization method detection."""

    def test_cuda_enables_bnb(self):
        """CUDA systems should support BNB."""
        quants = detect_supported_quantizations(has_cuda=True)
        assert "GGUF" in quants
        assert "BNB" in quants
        assert "ONNX" in quants

    def test_no_cuda_still_has_onnx(self):
        """Non-CUDA systems should still support ONNX."""
        quants = detect_supported_quantizations(has_cuda=False)
        assert "GGUF" in quants
        assert "ONNX" in quants
        assert "BNB" not in quants


class TestAutoDetectHardware:
    """Tests for full hardware auto-detection."""

    def test_returns_hardware_profile(self):
        """Should return a valid HardwareProfile."""
        profile = auto_detect_hardware()
        
        assert profile.name.startswith("Auto-detected:")
        assert profile.constraints.vram_gb > 0
        assert profile.constraints.cpu_arch in ["arm64", "x86_64", "arm"] or len(profile.constraints.cpu_arch) > 0
        assert isinstance(profile.constraints.has_cuda, bool)
        assert profile.targets.min_tokens_per_second > 0
        assert profile.targets.min_accuracy_score > 0

    def test_respects_custom_targets(self):
        """Should use custom target values when provided."""
        profile = auto_detect_hardware(
            target_tps=100.0,
            target_accuracy=0.9,
            max_latency_ms=50.0,
        )
        
        assert profile.targets.min_tokens_per_second == 100.0
        assert profile.targets.min_accuracy_score == 0.9
        assert profile.targets.max_latency_ms == 50.0

    def test_auto_calculates_latency_for_high_vram(self):
        """Should calculate lower latency for high-VRAM systems."""
        with patch("sintra.profiles.hardware.detect_available_memory_gb", return_value=32.0):
            profile = auto_detect_hardware()
            assert profile.targets.max_latency_ms == 200.0

    def test_auto_calculates_latency_for_medium_vram(self):
        """Should calculate medium latency for medium-VRAM systems."""
        with patch("sintra.profiles.hardware.detect_available_memory_gb", return_value=10.0):
            profile = auto_detect_hardware()
            assert profile.targets.max_latency_ms == 500.0

    def test_auto_calculates_latency_for_low_vram(self):
        """Should calculate higher latency for low-VRAM systems."""
        with patch("sintra.profiles.hardware.detect_available_memory_gb", return_value=4.0):
            profile = auto_detect_hardware()
            assert profile.targets.max_latency_ms == 1000.0


class TestGetGpuVramGb:
    """Tests for GPU VRAM detection."""

    def test_no_gpu_returns_none(self):
        """Should return None when no GPU is available."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            assert get_gpu_vram_gb() is None

    def test_parses_nvidia_smi_output(self):
        """Should parse nvidia-smi output correctly."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="8192\n"  # 8GB in MB
            )
            result = get_gpu_vram_gb()
            assert result == 8.0

    def test_sums_multiple_gpus(self):
        """Should sum VRAM from multiple GPUs."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="8192\n8192\n"  # Two 8GB GPUs
            )
            result = get_gpu_vram_gb()
            assert result == 16.0
