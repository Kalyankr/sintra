"""Sintra console UI with Nord-Industrial theme."""

from datetime import datetime
from typing import Literal

from rich.console import Console
from rich.style import Style
from rich.theme import Theme

# Valid style keys for log_transition
StyleKey = Literal[
    "arch.node",
    "lab.node",
    "critic.node",
    "status.success",
    "status.fail",
    "status.warn",
    "hw.profile",
]

# Define the Nord-Industrial Theme
SINTRA_THEME = Theme(
    {
        "arch.node": Style(color="cyan", bold=True),
        "lab.node": Style(color="magenta", bold=True),
        "critic.node": Style(color="yellow", bold=True),
        "status.success": Style(color="bright_green"),
        "status.fail": Style(color="red", bold=True),
        "status.warn": Style(color="yellow", bold=True),
        "hw.profile": Style(color="blue", italic=True),
    }
)

# Initialize the global console
console = Console(theme=SINTRA_THEME)


def log_transition(node_name: str, message: str, style_key: str) -> None:
    """Standardized logger for Agent transitions.
    
    Args:
        node_name: Name of the agent node (e.g., 'Architect', 'Lab').
        message: The message to display.
        style_key: Theme style key for coloring.
    
    Example:
        >>> log_transition("Architect", "Analyzing hardware...", "arch.node")
        [12:00:01] | ARCHITECT  | Analyzing hardware...
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print(
        f"[dim][{timestamp}][/][white] | [/]"
        f"[{style_key}]{node_name.upper():<10}[/]"
        f"[white] | {message}[/]"
    )
