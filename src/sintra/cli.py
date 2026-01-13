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
        "profile", type=str, help="Path to hardware YAML (e.g., profiles/pi5.yaml)"
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

    # Debug Mode
    group_debug = parser.add_argument_group("Debugging")
    group_debug.add_argument(
        "--debug",
        action="store_true",
        help="Run a single-loop test without calling the LLM API",
    )

    return parser.parse_args()
