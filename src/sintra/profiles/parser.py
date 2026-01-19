import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import HardwareProfile

logger = logging.getLogger(__name__)


class ProfileLoadError(Exception):
    """Raised when a hardware profile cannot be loaded."""

    pass


def load_hardware_profile(file_path: str | Path) -> HardwareProfile:
    """Loads and validates a hardware profile from a YAML file.

    Args:
        file_path: Path to the YAML profile file.

    Returns:
        A validated HardwareProfile instance.

    Raises:
        ProfileLoadError: If the file cannot be read, parsed, or validated.
    """
    path = Path(file_path)

    if not path.exists():
        raise ProfileLoadError(f"Profile file not found: {path}")

    if not path.is_file():
        raise ProfileLoadError(f"Profile path is not a file: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ProfileLoadError(f"Invalid YAML in profile file: {e}") from e
    except OSError as e:
        raise ProfileLoadError(f"Cannot read profile file: {e}") from e

    if config is None:
        raise ProfileLoadError(f"Profile file is empty: {path}")

    if not isinstance(config, dict):
        raise ProfileLoadError(
            f"Profile must be a YAML mapping, got {type(config).__name__}"
        )

    try:
        return HardwareProfile(**config)
    except ValidationError as e:
        raise ProfileLoadError(f"Profile validation failed: {e}") from e
