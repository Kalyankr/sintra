import argparse

from sintra.profiles.models import LLMConfig, LLMProvider

# Default model to optimize
DEFAULT_TARGET_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def parse_args() -> argparse.Namespace:
    """Defines and parses the Sintra Command Line Interface.

    Default behavior (no flags needed):
    - Auto-detects hardware specs
    - Enables all agentic features (planner, ReAct, reflection, LLM routing)
    - Compares accuracy against baseline original model
    - Uses GGUF backend with llama.cpp
    """

    # config
    llm_defaults = LLMConfig()
    parser = argparse.ArgumentParser(
        description="SINTRA: Autonomous Edge AI Distiller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Optional: Hardware profile (auto-detect is now default)
    parser.add_argument(
        "profile",
        type=str,
        nargs="?",  # Optional - auto-detect is default
        help="Path to hardware YAML (e.g., profiles/pi5.yaml). If not provided, hardware is auto-detected.",
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
        default=True,  # Now default!
        help="Auto-detect hardware specs (default: enabled)",
    )
    group_hw.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Disable auto-detection (requires a profile YAML)",
    )
    group_hw.add_argument(
        "--target-tps",
        type=float,
        default=30.0,
        help="Target tokens per second",
    )
    group_hw.add_argument(
        "--target-accuracy",
        type=float,
        default=0.65,
        help="Minimum accuracy score",
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

    # Agentic Features (enabled by default)
    group_agentic = parser.add_argument_group("Agentic Features (enabled by default)")
    group_agentic.add_argument(
        "--simple",
        action="store_true",
        help="Disable all agentic features (use basic architect without tools)",
    )
    group_agentic.add_argument(
        "--no-plan",
        action="store_true",
        help="Disable planner agent",
    )
    group_agentic.add_argument(
        "--no-react",
        action="store_true",
        help="Disable ReAct-style architect (use basic prompt)",
    )
    group_agentic.add_argument(
        "--no-reflect",
        action="store_true",
        help="Disable self-reflection node",
    )
    group_agentic.add_argument(
        "--no-llm-routing",
        action="store_true",
        help="Use rule-based routing instead of LLM",
    )

    # Evaluation Settings (baseline comparison enabled by default)
    group_eval = parser.add_argument_group("Evaluation Settings")
    group_eval.add_argument(
        "--baseline",
        action="store_true",
        default=True,  # Now default!
        help="Compare accuracy against original model (default: enabled)",
    )
    group_eval.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline comparison for faster runs",
    )
    group_eval.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Skip all accuracy evaluation",
    )

    # Debug Mode
    group_debug = parser.add_argument_group("Debugging")
    group_debug.add_argument(
        "--debug",
        action="store_true",
        help="Run a single-loop test without calling the LLM API",
    )
    group_debug.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity (-v for info, -vv for debug)",
    )

    # Export Options
    group_export = parser.add_argument_group("Export Options")
    group_export.add_argument(
        "--export-ollama",
        type=str,
        metavar="MODEL_NAME",
        default=None,
        help="Export optimized model to Ollama with given name (e.g., --export-ollama my-model:q4)",
    )
    group_export.add_argument(
        "--ollama-system-prompt",
        type=str,
        default=None,
        help="System prompt for exported Ollama model",
    )

    args = parser.parse_args()

    # Handle --no-auto-detect flag
    if args.no_auto_detect:
        args.auto_detect = False

    # Handle --no-baseline flag
    if args.no_baseline:
        args.baseline = False

    # Validation: either profile or auto-detect is required
    if not args.profile and not args.auto_detect:
        parser.error(
            "Either a profile path or auto-detection is required. Use --no-auto-detect only with a profile."
        )

    return args
