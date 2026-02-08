"""Tests for Ollama exporter."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from sintra.compression.ollama_exporter import (
    OllamaExporter,
    OllamaExportError,
    OllamaExportResult,
    export_to_ollama,
    generate_modelfile,
    is_ollama_available,
)


class TestIsOllamaAvailable:
    """Tests for is_ollama_available function."""

    def test_returns_false_when_not_found(self):
        """Test that returns False when ollama is not installed."""
        with patch("shutil.which", return_value=None):
            available, message = is_ollama_available()
            assert available is False
            assert "not found" in message.lower()

    def test_returns_true_when_available(self):
        """Test that returns True when ollama is running."""
        with (
            patch("shutil.which", return_value="/usr/bin/ollama"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            available, message = is_ollama_available()
            assert available is True
            assert "running" in message.lower()

    def test_returns_false_when_not_running(self):
        """Test that returns False when ollama is installed but not running."""
        with (
            patch("shutil.which", return_value="/usr/bin/ollama"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr="connection refused")
            available, message = is_ollama_available()
            assert available is False
            assert "not running" in message.lower()

    def test_handles_timeout(self):
        """Test that handles timeout gracefully."""
        with (
            patch("shutil.which", return_value="/usr/bin/ollama"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 5)
            available, message = is_ollama_available()
            assert available is False
            assert "timed out" in message.lower()


class TestGenerateModelfile:
    """Tests for generate_modelfile function."""

    def test_basic_modelfile(self, tmp_path):
        """Test basic Modelfile generation."""
        gguf_path = tmp_path / "model.gguf"
        content = generate_modelfile(gguf_path)
        assert f"FROM {gguf_path.absolute()}" in content

    def test_with_system_prompt(self, tmp_path):
        """Test Modelfile with system prompt."""
        gguf_path = tmp_path / "model.gguf"
        content = generate_modelfile(gguf_path, system_prompt="You are helpful.")
        assert "SYSTEM" in content
        assert "You are helpful." in content

    def test_with_parameters(self, tmp_path):
        """Test Modelfile with custom parameters."""
        gguf_path = tmp_path / "model.gguf"
        content = generate_modelfile(
            gguf_path, parameters={"temperature": 0.7, "top_p": 0.9}
        )
        assert "PARAMETER temperature 0.7" in content
        assert "PARAMETER top_p 0.9" in content


class TestOllamaExporter:
    """Tests for OllamaExporter class."""

    def test_check_availability(self):
        """Test check_availability method."""
        exporter = OllamaExporter()
        with patch(
            "sintra.compression.ollama_exporter.is_ollama_available",
            return_value=(True, "OK"),
        ):
            available, _msg = exporter.check_availability()
            assert available is True

    def test_export_validates_gguf_path(self, tmp_path):
        """Test that export validates GGUF file exists."""
        exporter = OllamaExporter()
        non_existent = tmp_path / "does_not_exist.gguf"

        with pytest.raises(OllamaExportError) as exc:
            exporter.export(non_existent, "test-model")
        assert "not found" in str(exc.value).lower()

    def test_export_validates_gguf_extension(self, tmp_path):
        """Test that export validates GGUF extension."""
        exporter = OllamaExporter()
        not_gguf = tmp_path / "model.txt"
        not_gguf.touch()

        with pytest.raises(OllamaExportError) as exc:
            exporter.export(not_gguf, "test-model")
        assert "not a gguf" in str(exc.value).lower()

    def test_export_checks_ollama_availability(self, tmp_path):
        """Test that export checks if ollama is available."""
        exporter = OllamaExporter()
        gguf_file = tmp_path / "model.gguf"
        gguf_file.touch()

        with patch(
            "sintra.compression.ollama_exporter.is_ollama_available",
            return_value=(False, "Not installed"),
        ):
            with pytest.raises(OllamaExportError) as exc:
                exporter.export(gguf_file, "test-model")
            assert "not installed" in str(exc.value).lower()

    def test_export_success(self, tmp_path):
        """Test successful export."""
        exporter = OllamaExporter()
        gguf_file = tmp_path / "model.gguf"
        gguf_file.touch()

        with (
            patch(
                "sintra.compression.ollama_exporter.is_ollama_available",
                return_value=(True, "OK"),
            ),
            patch.object(exporter, "_list_models", return_value=[]),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = exporter.export(gguf_file, "test-model")

        assert result.success is True
        assert result.model_name == "test-model"

    def test_export_with_force_overwrites(self, tmp_path):
        """Test that force=True allows overwriting existing model."""
        exporter = OllamaExporter()
        gguf_file = tmp_path / "model.gguf"
        gguf_file.touch()

        with (
            patch(
                "sintra.compression.ollama_exporter.is_ollama_available",
                return_value=(True, "OK"),
            ),
            patch.object(exporter, "_list_models", return_value=["test-model"]),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = exporter.export(gguf_file, "test-model", force=True)

        assert result.success is True


class TestExportToOllamaConvenience:
    """Tests for export_to_ollama convenience function."""

    def test_calls_exporter(self, tmp_path):
        """Test that convenience function calls exporter."""
        gguf_file = tmp_path / "model.gguf"
        gguf_file.touch()

        with (
            patch(
                "sintra.compression.ollama_exporter.is_ollama_available",
                return_value=(True, "OK"),
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = export_to_ollama(gguf_file, "test-model", force=True)

        assert isinstance(result, OllamaExportResult)


class TestOllamaExportResult:
    """Tests for OllamaExportResult dataclass."""

    def test_str_success(self):
        """Test string representation for successful export."""
        result = OllamaExportResult(
            model_name="my-model",
            gguf_path="/path/to/model.gguf",
            modelfile_path="/path/to/Modelfile",
            success=True,
            message="OK",
        )
        output = str(result)
        assert "successfully" in output.lower()
        assert "my-model" in output
        assert "ollama run" in output

    def test_str_failure(self):
        """Test string representation for failed export."""
        result = OllamaExportResult(
            model_name="my-model",
            gguf_path="/path/to/model.gguf",
            modelfile_path="/path/to/Modelfile",
            success=False,
            message="Connection refused",
        )
        output = str(result)
        assert "failed" in output.lower()
        assert "Connection refused" in output
