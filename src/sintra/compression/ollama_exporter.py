"""Ollama model export for seamless local deployment.

Exports optimized GGUF models directly to Ollama for inference.
Supports automatic Modelfile generation and model registration.

Usage:
    >>> from sintra.compression.ollama_exporter import OllamaExporter
    >>> exporter = OllamaExporter()
    >>> exporter.export("/path/to/model.gguf", "my-optimized-model")
    # Model now available as: ollama run my-optimized-model
"""

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class OllamaExportError(Exception):
    """Raised when Ollama export fails."""

    pass


@dataclass
class OllamaExportResult:
    """Result of an Ollama export operation."""

    model_name: str
    gguf_path: str
    modelfile_path: str
    success: bool
    message: str

    def __str__(self) -> str:
        if self.success:
            return (
                f"✓ Model exported successfully!\n"
                f"  Name: {self.model_name}\n"
                f"  Run:  ollama run {self.model_name}"
            )
        return f"✗ Export failed: {self.message}"


def is_ollama_available() -> tuple[bool, str]:
    """Check if Ollama is installed and running.

    Returns:
        Tuple of (is_available, status_message)
    """
    # Check if ollama command exists
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        return False, "Ollama not found. Install from https://ollama.ai"

    # Check if ollama is running
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, "Ollama is installed and running"
        else:
            return False, f"Ollama is installed but not running: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, "Ollama timed out - may not be running (try: ollama serve)"
    except Exception as e:
        return False, f"Error checking Ollama: {e}"


def generate_modelfile(
    gguf_path: Path,
    system_prompt: str | None = None,
    template: str | None = None,
    parameters: dict | None = None,
) -> str:
    """Generate an Ollama Modelfile for a GGUF model.

    Args:
        gguf_path: Path to the GGUF model file
        system_prompt: Optional system prompt for the model
        template: Optional chat template
        parameters: Optional model parameters (temperature, top_p, etc.)

    Returns:
        Modelfile content as string
    """
    lines = [f"FROM {gguf_path.absolute()}"]

    if system_prompt:
        # Escape quotes in system prompt
        escaped_prompt = system_prompt.replace('"', '\\"')
        lines.append(f'SYSTEM "{escaped_prompt}"')

    if template:
        lines.append(f"TEMPLATE {template!r}")

    if parameters:
        for key, value in parameters.items():
            lines.append(f"PARAMETER {key} {value}")

    return "\n".join(lines)


class OllamaExporter:
    """Exports GGUF models to Ollama for local inference.

    Example:
        >>> exporter = OllamaExporter()
        >>> result = exporter.export(
        ...     Path("/models/tinyllama-q4.gguf"),
        ...     model_name="my-tiny-llama",
        ...     system_prompt="You are a helpful assistant."
        ... )
        >>> print(result)
        ✓ Model exported successfully!
          Name: my-tiny-llama
          Run:  ollama run my-tiny-llama
    """

    def __init__(self, timeout: int = 300):
        """Initialize the exporter.

        Args:
            timeout: Timeout in seconds for ollama create command
        """
        self.timeout = timeout

    def check_availability(self) -> tuple[bool, str]:
        """Check if Ollama is available."""
        return is_ollama_available()

    def export(
        self,
        gguf_path: Path,
        model_name: str,
        system_prompt: str | None = None,
        template: str | None = None,
        parameters: dict | None = None,
        force: bool = False,
    ) -> OllamaExportResult:
        """Export a GGUF model to Ollama.

        Args:
            gguf_path: Path to the GGUF model file
            model_name: Name for the Ollama model (e.g., "my-model:q4")
            system_prompt: Optional system prompt
            template: Optional chat template
            parameters: Optional model parameters
            force: Overwrite existing model with same name

        Returns:
            OllamaExportResult with export status

        Raises:
            OllamaExportError: If export fails
        """
        gguf_path = Path(gguf_path)

        # Validate GGUF file
        if not gguf_path.exists():
            raise OllamaExportError(f"GGUF file not found: {gguf_path}")

        if not gguf_path.suffix.lower() == ".gguf":
            raise OllamaExportError(f"File is not a GGUF model: {gguf_path}")

        # Check Ollama availability
        available, message = self.check_availability()
        if not available:
            raise OllamaExportError(message)

        # Check if model already exists
        if not force:
            existing = self._list_models()
            if model_name in existing:
                raise OllamaExportError(
                    f"Model '{model_name}' already exists. Use force=True to overwrite."
                )

        # Generate Modelfile
        modelfile_content = generate_modelfile(
            gguf_path,
            system_prompt=system_prompt,
            template=template,
            parameters=parameters,
        )

        logger.info(f"Exporting {gguf_path.name} to Ollama as '{model_name}'...")

        # Create temporary Modelfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".Modelfile", delete=False
        ) as f:
            f.write(modelfile_content)
            modelfile_path = f.name

        try:
            # Run ollama create
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", modelfile_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode == 0:
                logger.info(f"Successfully exported model as '{model_name}'")
                return OllamaExportResult(
                    model_name=model_name,
                    gguf_path=str(gguf_path.absolute()),
                    modelfile_path=modelfile_path,
                    success=True,
                    message=f"Model available as: ollama run {model_name}",
                )
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                logger.error(f"Ollama create failed: {error_msg}")
                return OllamaExportResult(
                    model_name=model_name,
                    gguf_path=str(gguf_path.absolute()),
                    modelfile_path=modelfile_path,
                    success=False,
                    message=error_msg,
                )

        except subprocess.TimeoutExpired:
            raise OllamaExportError(
                f"Ollama create timed out after {self.timeout}s. "
                "Try increasing timeout or check if Ollama is running."
            )
        except Exception as e:
            raise OllamaExportError(f"Failed to create Ollama model: {e}")

    def _list_models(self) -> list[str]:
        """List existing Ollama models."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []

            # Parse output (format: NAME ID SIZE MODIFIED)
            models = []
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            return models
        except Exception:
            return []

    def delete(self, model_name: str) -> bool:
        """Delete an Ollama model.

        Args:
            model_name: Name of the model to delete

        Returns:
            True if deleted successfully
        """
        try:
            result = subprocess.run(
                ["ollama", "rm", model_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False


def export_to_ollama(
    gguf_path: Path,
    model_name: str,
    system_prompt: str | None = None,
    force: bool = False,
) -> OllamaExportResult:
    """Convenience function to export a GGUF model to Ollama.

    Args:
        gguf_path: Path to the GGUF model file
        model_name: Name for the Ollama model
        system_prompt: Optional system prompt
        force: Overwrite existing model

    Returns:
        OllamaExportResult with export status
    """
    exporter = OllamaExporter()
    return exporter.export(
        gguf_path,
        model_name,
        system_prompt=system_prompt,
        force=force,
    )
