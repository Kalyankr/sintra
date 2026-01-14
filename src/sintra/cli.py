import argparse

from sintra.profiles.models import LLMConfig, LLMProvider

# Default model to optimize
DEFAULT_TARGET_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def parse_args() -> argparse.Namespace:
    """Defines and parses the Sintra Command Line Interface."""

    # config
    llm_defaults = LLMConfig()
    parser = argparse.ArgumentParser(
        description="SINTRA: Autonomous Edge AI Distiller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required: What are we optimizing for?
    parser.add_argument(
        "profile",
        type=str,
        nargs="?",  # Optional when using --auto-detect
        help="Path to hardware YAML (e.g., profiles/pi5.yaml)",
    )

    # Target Model Configuration
    group_target = parser.add_argument_group("Target Model")
    group_target.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_TARGET_MODEL,
        help="HuggingFace model ID to optimize (e.g., meta-llama/Llama-3-8B)",
    )
    group_target.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (or set HF_TOKEN env var)",
    )
    group_target.add_argument(
        "--backend",
        type=str,
        default="gguf",
        choices=["gguf", "bnb", "onnx"],
        help="Quantization backend: gguf (llama.cpp), bnb (bitsandbytes), onnx (optimum)",
    )
    group_target.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save optimized models (default: ./outputs)",
    )

    # Hardware Configuration
    group_hw = parser.add_argument_group("Hardware Configuration")
    group_hw.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect hardware specs instead of using a YAML profile",
    )
    group_hw.add_argument(
        "--target-tps",
        type=float,
        default=30.0,
        help="Target tokens per second (used with --auto-detect)",
    )
    group_hw.add_argument(
        "--target-accuracy",
        type=float,
        default=0.65,
        help="Minimum accuracy score (used with --auto-detect)",
    )

    # Brain Configuration
    group_brain = parser.add_argument_group("Architect Brain")
    group_brain.add_argument(
        "--provider",
        type=str,
        default=llm_defaults.provider.value,
        choices=[p.value for p in LLMProvider],
        help=f"The LLM provider acting as the Architect (default: {llm_defaults.provider.value})",
    )
    group_brain.add_argument(
        "--model",
        type=str,
        default=llm_defaults.model_name,
        help="Specific model name for the Architect",
    )

    # Execution Flags
    group_exec = parser.add_argument_group("Execution Settings")
    group_exec.add_argument(
        "--mock",
        action="store_true",
        help="Use MockExecutor instead of actual model surgery",
    )
    group_exec.add_argument(
        "--max-iters",
        type=int,
        default=10,
        help="Maximum optimization attempts before halting",
    )
    group_exec.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing compression",
    )
    group_exec.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="latest",
        default=None,
        metavar="RUN_ID",
        help="Resume from checkpoint. Use 'latest' or specify a run_id",
    )
    group_exec.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List all available checkpoints and exit",
    )

    # Agentic Features
    group_agentic = parser.add_argument_group("Agentic Features")
    group_agentic.add_argument(
        "--plan",
        action="store_true",
        help="Enable planner agent to create optimization strategy",
    )
    group_agentic.add_argument(
        "--react",
        action="store_true",
        help="Use ReAct-style architect with tool use (more agentic)",
    )
    group_agentic.add_argument(
        "--reflect",
        action="store_true",
        help="Enable self-reflection node for failure analysis",
    )
    group_agentic.add_argument(
        "--llm-routing",
        action="store_true",
        help="Use LLM for routing decisions instead of rules",
    )
    group_agentic.add_argument(
        "--agentic",
        action="store_true",
        help="Enable all agentic features (--plan --react --reflect --llm-routing)",
    )

    # Debug Mode
    group_debug = parser.add_argument_group("Debugging")
    group_debug.add_argument(
        "--debug",
        action="store_true",
        help="Run a single-loop test without calling the LLM API",
    )

    args = parser.parse_args()

    # Validation: either profile or --auto-detect is required
    if not args.profile and not args.auto_detect:
        parser.error("Either a profile path or --auto-detect is required")

    return args
