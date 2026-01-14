import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

# Load environment variables from .env file (if exists)
load_dotenv()

from sintra.agents.factory import MissingAPIKeyError
from sintra.agents.nodes import (
    LLMConnectionError,
    architect_node,
    benchmarker_node,
    critic_node,
    critic_router,
    reporter_node,
)
from sintra.agents.state import SintraState
from sintra.cli import parse_args
from sintra.profiles.hardware import auto_detect_hardware, print_hardware_info
from sintra.profiles.models import LLMConfig, LLMProvider
from sintra.profiles.parser import ProfileLoadError, load_hardware_profile
from sintra.ui.console import console, log_transition
from sintra.ui.progress import ConsoleProgressReporter, set_global_reporter


def build_sintra_workflow():
    workflow = StateGraph(SintraState)

    # Define the "Actors"
    workflow.add_node("architect", architect_node)
    workflow.add_node("benchmarker", benchmarker_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("reporter", reporter_node)

    # Define the "Path"
    workflow.set_entry_point("architect")
    workflow.add_edge("architect", "benchmarker")
    workflow.add_edge("benchmarker", "critic")

    # The Decision Point: Critic determines if we repeat or stop
    workflow.add_conditional_edges(
        "critic",
        critic_router,
        {
            "architect": "architect",
            "reporter": "reporter",
        },
    )
    # reporter leads to END
    workflow.add_edge("reporter", END)

    return workflow.compile()


def main():
    # Get CLI arguments
    args = parse_args()

    # Setup UI
    console.rule("[arch.node] SINTRA: Edge AI Distiller")
    
    # Setup progress reporter
    set_global_reporter(ConsoleProgressReporter(show_details=args.debug))

    # Load Hardware Context - either from YAML or auto-detect
    if args.auto_detect:
        print_hardware_info()
        profile = auto_detect_hardware(
            target_tps=args.target_tps,
            target_accuracy=args.target_accuracy,
        )
        log_transition("System", "Using auto-detected hardware profile", "hw.profile")
    else:
        try:
            profile = load_hardware_profile(args.profile)
        except ProfileLoadError as e:
            console.print(f"[status.fail] Failed to load hardware profile: {e}")
            sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SINTRA_OUTPUT_DIR"] = str(output_dir.absolute())

    # Handle dry-run mode
    if args.dry_run:
        _run_dry_mode(args, profile, output_dir)
        return

    # Set up environment for worker subprocess
    os.environ["SINTRA_REAL_COMPRESSION"] = "true"
    os.environ["SINTRA_MODEL_ID"] = args.model_id
    os.environ["SINTRA_BACKEND"] = args.backend
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    
    backend_names = {"gguf": "GGUF/llama.cpp", "bnb": "bitsandbytes", "onnx": "ONNX/Optimum"}
    log_transition(
        "System", 
        f"Compression: {args.model_id} via {backend_names.get(args.backend, args.backend)}", 
        "hw.profile"
    )

    # Initialize State
    initial_state: SintraState = {
        "profile": profile,
        "llm_config": LLMConfig(
            provider=LLMProvider(args.provider),
            model_name=args.model,
        ),
        "use_debug": args.debug,
        "use_mock": args.mock,
        "target_model_id": args.model_id,
        "iteration": 0,
        "history": [],
        "is_converged": False,
        "current_recipe": None,
        "critic_feedback": "",
        "best_recipe": None,
    }

    log_transition(
        "System", f"Ready. Target: {profile.name} | Brain: {args.model}", "hw.profile"
    )
    if not args.debug:
        log_transition("System", f"Optimizing: {args.model_id}", "hw.profile")

    # Run Workflow
    app = build_sintra_workflow()

    # Streaming the graph for real-time console updates
    try:
        for _ in app.stream(initial_state, config={"recursion_limit": 50}):
            pass
        console.rule("[status.success] OPTIMIZATION COMPLETE")
    except MissingAPIKeyError as e:
        console.print(f"\n[bold red]✗ Missing API Key[/bold red]")
        console.print(f"  {e}")
        console.print("\n[dim]Setup:[/dim]")
        console.print(
            "  1. Copy .env.example to .env: [cyan]cp .env.example .env[/cyan]"
        )
        console.print("  2. Edit .env and add your API key")
        console.print("  3. Or export it: [cyan]export OPENAI_API_KEY=sk-...[/cyan]")
        sys.exit(1)
    except LLMConnectionError as e:
        console.print(f"\n[bold red]✗ LLM Connection Failed[/bold red]")
        console.print(f"  {e}")
        console.print("\n[dim]Suggestions:[/dim]")
        if args.provider == "ollama":
            console.print("  • Start Ollama: [cyan]ollama serve[/cyan]")
            console.print(
                "  • Or use debug mode: [cyan]sintra --debug <profile>[/cyan]"
            )
        console.print(
            "  • Or try a different provider: [cyan]--provider openai --model gpt-4o[/cyan]"
        )
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Optimization cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        # Catch-all for unexpected errors
        console.print(f"\n[bold red]✗ Unexpected Error[/bold red]")
        console.print(f"  {type(e).__name__}: {e}")
        console.print("\n[dim]This might be a bug. Please report it with:[/dim]")
        console.print("  • The command you ran")
        console.print("  • Your hardware profile")
        console.print("  • The full error message above")
        if args.debug:
            # In debug mode, show full traceback
            import traceback

            console.print("\n[dim]Full traceback:[/dim]")
            console.print(traceback.format_exc())
        sys.exit(1)


def _run_dry_mode(args, profile, output_dir: Path) -> None:
    """Execute dry-run mode: show what would happen without running compression."""
    from sintra.profiles.models import HardwareProfile
    
    console.print("\n[bold yellow]🔍 DRY RUN MODE[/bold yellow]")
    console.print("[dim]No actual compression will be performed.[/dim]\n")
    
    # Show configuration
    console.print("[bold cyan]Configuration Summary[/bold cyan]")
    console.print(f"  Model: {args.model_id}")
    console.print(f"  Backend: {args.backend}")
    console.print(f"  Output Directory: {output_dir.absolute()}")
    console.print(f"  Max Iterations: {args.max_iters}")
    console.print(f"  Mock Mode: {args.mock}")
    console.print()
    
    # Show hardware profile
    console.print("[bold cyan]Hardware Profile[/bold cyan]")
    console.print(f"  Name: {profile.name}")
    console.print(f"  Available Memory: {profile.constraints.vram_gb} GB")
    console.print(f"  CPU Architecture: {profile.constraints.cpu_arch}")
    console.print(f"  CUDA Available: {profile.constraints.has_cuda}")
    if profile.supported_quantizations:
        console.print(f"  Supported Quantizations: {', '.join(profile.supported_quantizations)}")
    console.print()
    
    # Show targets
    console.print("[bold cyan]Optimization Targets[/bold cyan]")
    console.print(f"  Min Tokens/Second: {profile.targets.min_tokens_per_second}")
    console.print(f"  Min Accuracy: {profile.targets.min_accuracy_score}")
    if profile.targets.max_latency_ms:
        console.print(f"  Max Latency: {profile.targets.max_latency_ms} ms")
    console.print()
    
    # Show what would happen
    console.print("[bold cyan]Planned Actions[/bold cyan]")
    console.print("  1. Initialize LangGraph workflow with architect, benchmarker, critic nodes")
    console.print(f"  2. Download model from HuggingFace: {args.model_id}")
    console.print(f"  3. Apply quantization using {args.backend} backend")
    console.print("  4. Benchmark compressed model for TPS and accuracy")
    console.print("  5. Iterate until targets met or max iterations reached")
    console.print(f"  6. Save optimized model to: {output_dir.absolute()}")
    console.print()
    
    # Save dry-run config to JSON
    dry_run_config = {
        "mode": "dry-run",
        "model_id": args.model_id,
        "backend": args.backend,
        "output_dir": str(output_dir.absolute()),
        "max_iters": args.max_iters,
        "profile": {
            "name": profile.name,
            "constraints": {
                "vram_gb": profile.constraints.vram_gb,
                "cpu_arch": profile.constraints.cpu_arch,
                "has_cuda": profile.constraints.has_cuda,
            },
            "targets": {
                "min_tokens_per_second": profile.targets.min_tokens_per_second,
                "min_accuracy_score": profile.targets.min_accuracy_score,
                "max_latency_ms": profile.targets.max_latency_ms,
            },
        },
    }
    
    config_path = output_dir / "dry_run_config.json"
    with open(config_path, "w") as f:
        json.dump(dry_run_config, f, indent=2)
    
    console.print(f"[green]✓ Dry-run config saved to: {config_path}[/green]")
    console.print("\n[dim]To run actual optimization, remove the --dry-run flag.[/dim]")


if __name__ == "__main__":
    main()
