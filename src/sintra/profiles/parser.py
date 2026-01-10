from pathlib import Path
from typing import Union

import yaml

from .models import HardwareProfile


def load_hardware_profile(file_path: Union[str, Path]) -> HardwareProfile:
    """Loads and validates a hardware profile from a YAML file."""
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return HardwareProfile(**config)
