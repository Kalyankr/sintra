from datetime import datetime

from rich.console import Console
from rich.style import Style
from rich.theme import Theme

# Define the Nord-Industrial Theme
SINTRA_THEME = Theme(
    {
        "arch.node": Style(color="cyan", bold=True),
        "lab.node": Style(color="magenta", bold=True),
        "critic.node": Style(color="yellow", bold=True),
        "status.success": Style(color="bright_green"),
        "status.fail": Style(color="red", bold=True),
        "hw.profile": Style(color="blue", italic=True),
    }
)

# Initialize the global console
console = Console(theme=SINTRA_THEME)


def log_transition(node_name: str, message: str, style_key: str):
    """
    Standardized logger for Agent transitions.
    Example: [12:00:01] ARCHITECT | Analyzing hardware...
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.print(
        f"[dim][{timestamp}][/][white] | [/]"
        f"[{style_key}]{node_name.upper():<10}[/]"
        f"[white] | {message}[/]"
    )
