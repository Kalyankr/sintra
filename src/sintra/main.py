import json
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

# Load environment variables from .env file (if exists)
load_dotenv()

from sintra.agents.experts import expert_collaboration_node  # noqa: E402
from sintra.agents.factory import MissingAPIKeyError  # noqa: E402
from sintra.agents.nodes import (  # noqa: E402
    LLMConnectionError,
    architect_node,
    benchmarker_node,
    critic_node,
    critic_router,
    critic_router_llm,
    reporter_node,
)
from sintra.agents.planner import planner_node  # noqa: E402
from sintra.agents.react_architect import react_architect_node  # noqa: E402
from sintra.agents.reflector import reflector_node  # noqa: E402
from sintra.agents.state import SintraState  # noqa: E402
from sintra.checkpoint import (  # noqa: E402
    find_latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from sintra.cli import parse_args  # noqa: E402
from sintra.persistence import get_history_db  # noqa: E402
from sintra.profiles.hardware import (  # noqa: E402
    auto_detect_hardware,
    print_hardware_info,
)
from sintra.profiles.models import LLMConfig, LLMProvider  # noqa: E402
from sintra.profiles.parser import ProfileLoadError, load_hardware_profile  # noqa: E402
from sintra.ui.console import console, log_transition  # noqa: E402
from sintra.ui.progress import (  # noqa: E402
    ConsoleProgressReporter,
    set_global_reporter,
)


def build_sintra_workflow(
    use_planner: bool = False,
    use_react: bool = False,
    use_reflection: bool = False,
    use_llm_routing: bool = False,
    use_experts: bool = False,
):
    workflow = StateGraph(SintraState)

    # Optional: Add planner for strategic optimization
    if use_planner:
        workflow.add_node("planner", planner_node)

    # Optional: Add expert collaboration before architect
    if use_experts:
        workflow.add_node("experts", expert_collaboration_node)

    # Define the "Actors" - choose architect based on mode
    if use_react:
        workflow.add_node("architect", react_architect_node)
    else:
        workflow.add_node("architect", architect_node)

    workflow.add_node("benchmarker", benchmarker_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("reporter", reporter_node)

    # Optional: Add reflector for self-analysis
    if use_reflection:
        workflow.add_node("reflector", reflector_node)

    # Define the "Path"
    if use_planner and use_experts:
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "experts")
        workflow.add_edge("experts", "architect")
    elif use_planner:
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "architect")
    elif use_experts:
        workflow.set_entry_point("experts")
        workflow.add_edge("experts", "architect")
    else:
        workflow.set_entry_point("architect")

    workflow.add_edge("architect", "benchmarker")
    workflow.add_edge("benchmarker", "critic")

    # Choose routing function
    router_fn = critic_router_llm if use_llm_routing else critic_router

    # The Decision Point: Critic determines if we repeat or stop
    if use_reflection:
        # With reflection: critic → reflector → architect (when continuing)
        workflow.add_conditional_edges(
            "critic",
            router_fn,
            {
                "architect": "reflector",  # Go through reflector first
                "reporter": "reporter",
            },
        )
        workflow.add_edge("reflector", "architect")
    else:
        # Without reflection: critic → architect directly
        workflow.add_conditional_edges(
            "critic",
            router_fn,
            {
                "architect": "architect",
                "reporter": "reporter",
            },
        )

    # reporter leads to END
    workflow.add_edge("reporter", END)

    return workflow.compile()


def _setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level.

    Args:
        verbosity: 0=warning, 1=info, 2+=debug
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Set specific loggers
    loggers = [
        "sintra",
        "sintra.compression",
        "sintra.agents",
        "sintra.benchmarks",
    ]
    for logger_name in loggers:
        logging.getLogger(logger_name).setLevel(level)


def main():
    # Get CLI arguments
    args = parse_args()

    # Setup logging based on verbosity
    _setup_logging(getattr(args, "verbose", 0))

    # Setup UI
    console.rule("[arch.node] SINTRA: Edge AI Distiller")

    # Handle --list-checkpoints
    if args.list_checkpoints:
        _list_checkpoints()
        return

    # Handle --ui flag (launch web dashboard)
    if getattr(args, "ui", False):
        from sintra.ui.dashboard import check_gradio_available, create_dashboard

        if not check_gradio_available():
            console.print("\n[bold red]Gradio not installed.[/bold red]")
            console.print("Install with: [cyan]pip install sintra[ui][/cyan]")
            console.print("Or: [cyan]pip install gradio>=4.0.0[/cyan]")
            sys.exit(1)
        port = getattr(args, "ui_port", 7860)
        create_dashboard(port=port)
        return

    # Setup progress reporter
    show_details = args.debug or getattr(args, "verbose", 0) >= 1
    set_global_reporter(ConsoleProgressReporter(show_details=show_details))

    # Setup output directory first (needed for saving detected profile)
    output_dir = Path(args.output_dir) if args.output_dir else Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SINTRA_OUTPUT_DIR"] = str(output_dir.absolute())

    # Load Hardware Context - either from YAML or auto-detect
    if args.auto_detect:
        # Auto-detect hardware and calculate smart targets
        # Pass None to let the system calculate targets, or use CLI overrides if provided
        target_tps = args.target_tps if args.target_tps != 30.0 else None
        target_accuracy = args.target_accuracy if args.target_accuracy != 0.65 else None

        profile = auto_detect_hardware(
            target_tps=target_tps,
            target_accuracy=target_accuracy,
        )

        # Save the detected profile for review/editing
        profile_path = output_dir / "detected_profile.yaml"
        print_hardware_info(save_path=profile_path)
        log_transition("System", "Using auto-detected hardware profile", "hw.profile")
    else:
        try:
            profile = load_hardware_profile(args.profile)
        except ProfileLoadError as e:
            console.print(f"[status.fail] Failed to load hardware profile: {e}")
            sys.exit(1)

    # Handle dry-run mode
    if args.dry_run:
        _run_dry_mode(args, profile, output_dir)
        return

    # Set up environment for worker subprocess
    os.environ["SINTRA_REAL_COMPRESSION"] = "true"
    os.environ["SINTRA_MODEL_ID"] = args.model_id
    os.environ["SINTRA_BACKEND"] = args.backend
    os.environ["SINTRA_USE_BASELINE"] = "true" if args.baseline else "false"
    os.environ["SINTRA_SKIP_ACCURACY"] = "true" if args.skip_accuracy else "false"

    # Set up Ollama export if requested
    if getattr(args, "export_ollama", None):
        os.environ["SINTRA_EXPORT_OLLAMA"] = args.export_ollama
        if getattr(args, "ollama_system_prompt", None):
            os.environ["SINTRA_OLLAMA_SYSTEM_PROMPT"] = args.ollama_system_prompt
        log_transition(
            "System",
            f"Will export to Ollama as '{args.export_ollama}' on completion",
            "hw.profile",
        )
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Check backend dependencies (skip for mock and debug modes)
    if not args.mock and not args.debug:
        _check_backend_dependencies(args.backend)

    backend_names = {
        "gguf": "GGUF/llama.cpp",
        "bnb": "bitsandbytes",
        "onnx": "ONNX/Optimum",
    }
    log_transition(
        "System",
        f"Compression: {args.model_id} via {backend_names.get(args.backend, args.backend)}",
        "hw.profile",
    )

    # Handle resume mode
    initial_state = None
    run_id = None

    if args.resume:
        checkpoint_data = _load_resume_checkpoint(args.resume, args.model_id)
        if checkpoint_data:
            initial_state = checkpoint_data["state"]
            run_id = checkpoint_data["run_id"]
            iteration = checkpoint_data["iteration"]
            log_transition(
                "System",
                f"Resuming run {run_id[:8]}... from iteration {iteration}",
                "status.success",
            )
            # Override profile from checkpoint
            profile = initial_state["profile"]
        else:
            console.print("[yellow]No checkpoint found, starting fresh run[/yellow]")

    # Generate unique run ID if not resuming
    if run_id is None:
        run_id = str(uuid.uuid4())

    db = get_history_db()

    # Only start a new run if not resuming
    if initial_state is None:
        db.start_run(
            run_id=run_id,
            model_id=args.model_id,
            profile=profile,
            backend=args.backend,
        )

        # Initialize State (fresh run)
        initial_state: SintraState = {
            "profile": profile,
            "llm_config": LLMConfig(
                provider=LLMProvider(args.provider),
                model_name=args.model,
            ),
            "use_debug": args.debug,
            "use_mock": args.mock,
            "use_react": getattr(args, "react", False),
            "target_model_id": args.model_id,
            "run_id": run_id,
            "backend": args.backend,
            "iteration": 0,
            "history": [],
            "is_converged": False,
            "current_recipe": None,
            "critic_feedback": "",
            "best_recipe": None,
            "reasoning_chain": None,
            "reasoning_summary": None,
            "reflection": None,
            "strategy_adjustments": None,
            "optimization_plan": None,
            "expert_consensus": None,
        }
    else:
        # Update resumed state with current session settings
        initial_state["use_debug"] = args.debug
        initial_state["use_mock"] = args.mock
        initial_state["use_react"] = getattr(args, "react", False)
        initial_state["run_id"] = run_id

    # Agentic features are ON by default, use --simple or --no-* to disable
    use_simple = getattr(args, "simple", False)
    use_planner = not use_simple and not getattr(args, "no_plan", False)
    use_react = not use_simple and not getattr(args, "no_react", False)
    use_reflect = not use_simple and not getattr(args, "no_reflect", False)
    use_llm_routing = not use_simple and not getattr(args, "no_llm_routing", False)
    use_experts = not use_simple and not getattr(args, "no_experts", False)

    log_transition(
        "System", f"Ready. Target: {profile.name} | Brain: {args.model}", "hw.profile"
    )

    # Show mode
    if use_simple:
        log_transition(
            "System", "Running in simple mode (agentic features disabled)", "hw.profile"
        )
    else:
        active_features = []
        if use_planner:
            active_features.append("planner")
        if use_react:
            active_features.append("ReAct")
        if use_reflect:
            active_features.append("reflection")
        if use_llm_routing:
            active_features.append("LLM-routing")
        if use_experts:
            active_features.append("multi-agent")
        if active_features:
            log_transition(
                "System", f"Agentic mode: {', '.join(active_features)}", "arch.node"
            )
    if not args.debug:
        log_transition("System", f"Optimizing: {args.model_id}", "hw.profile")

    # Run Workflow - use ReAct architect and reflection if flags are set
    app = build_sintra_workflow(
        use_planner=use_planner,
        use_react=use_react,
        use_reflection=use_reflect,
        use_llm_routing=use_llm_routing,
        use_experts=use_experts,
    )

    # Streaming the graph for real-time console updates
    final_state = None
    current_iteration = initial_state.get("iteration", 0)

    try:
        for state in app.stream(initial_state, config={"recursion_limit": 50}):
            final_state = state

            # Save checkpoint after each iteration (when we have benchmarker output)
            for node_name, node_output in state.items():
                if node_name == "benchmarker" and isinstance(node_output, dict):
                    # Merge node output with current state for checkpoint
                    checkpoint_state = {**initial_state}
                    if "history" in node_output:
                        checkpoint_state["history"] = (
                            checkpoint_state.get("history", []) + node_output["history"]
                        )
                    current_iteration += 1
                    checkpoint_state["iteration"] = current_iteration

                    save_checkpoint(run_id, checkpoint_state, current_iteration)
                    log_transition(
                        "System",
                        f"Checkpoint saved (iteration {current_iteration})",
                        "status.dim",
                    )

        # Mark run as complete in database
        best_recipe = None
        if final_state:
            # Get the best recipe from the final state
            for node_output in final_state.values():
                if isinstance(node_output, dict) and "best_recipe" in node_output:
                    best_recipe = node_output.get("best_recipe")
                    break

        db.finish_run(
            run_id=run_id,
            final_iteration=current_iteration,
            is_converged=True,
            best_recipe=best_recipe["recipe"] if best_recipe else None,
            status="completed",
        )
        console.rule("[status.success] OPTIMIZATION COMPLETE")
    except MissingAPIKeyError as e:
        db.finish_run(
            run_id=run_id,
            final_iteration=current_iteration,
            is_converged=False,
            status="failed",
        )
        console.print("\n[bold red]✗ Missing API Key[/bold red]")
        console.print(f"  {e}")
        console.print("\n[dim]Setup:[/dim]")
        console.print(
            "  1. Copy .env.example to .env: [cyan]cp .env.example .env[/cyan]"
        )
        console.print("  2. Edit .env and add your API key")
        console.print("  3. Or export it: [cyan]export OPENAI_API_KEY=sk-...[/cyan]")
        sys.exit(1)
    except LLMConnectionError as e:
        db.finish_run(
            run_id=run_id,
            final_iteration=current_iteration,
            is_converged=False,
            status="failed",
        )
        console.print("\n[bold red]✗ LLM Connection Failed[/bold red]")
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
        db.finish_run(
            run_id=run_id,
            final_iteration=current_iteration,
            is_converged=False,
            status="interrupted",
        )
        console.print("\n[yellow]Optimization cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        db.finish_run(
            run_id=run_id,
            final_iteration=current_iteration,
            is_converged=False,
            status="failed",
        )
        # Catch-all for unexpected errors
        console.print("\n[bold red]✗ Unexpected Error[/bold red]")
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
        console.print(
            f"  Supported Quantizations: {', '.join(profile.supported_quantizations)}"
        )
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
    console.print(
        "  1. Initialize LangGraph workflow with architect, benchmarker, critic nodes"
    )
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


def _list_checkpoints() -> None:
    """Display all available checkpoints."""
    checkpoints = list_checkpoints()

    if not checkpoints:
        console.print("[yellow]No checkpoints found.[/yellow]")
        console.print(
            "[dim]Checkpoints are saved automatically during optimization runs.[/dim]"
        )
        return

    console.print(
        f"\n[bold cyan]Available Checkpoints ({len(checkpoints)} total)[/bold cyan]\n"
    )

    for cp in checkpoints[:20]:  # Show at most 20
        console.print(f"  [bold]{cp['run_id'][:8]}...[/bold]")
        console.print(f"    Model: {cp['model_id']}")
        console.print(f"    Iteration: {cp['iteration']}")
        console.print(f"    Time: {cp['timestamp']}")
        console.print()

    if len(checkpoints) > 20:
        console.print(f"  [dim]... and {len(checkpoints) - 20} more[/dim]")

    console.print("[dim]Resume with: sintra --resume <run_id> --auto-detect[/dim]")
    console.print("[dim]Or resume latest: sintra --resume --auto-detect[/dim]")


def _load_resume_checkpoint(
    resume_arg: str, model_id: str | None = None
) -> dict | None:
    """Load checkpoint for resume.

    Args:
        resume_arg: Either 'latest' or a specific run_id
        model_id: Filter by model ID when finding latest

    Returns:
        Checkpoint data or None
    """
    if resume_arg == "latest":
        return find_latest_checkpoint(model_id=model_id)
    else:
        return load_checkpoint(resume_arg)


def _check_gguf_dependencies():
    """Check if llama.cpp is available for GGUF backend."""
    from sintra.compression.quantizer import (
        LLAMA_CPP_INSTALL_INSTRUCTIONS,
        check_llama_cpp_available,
    )

    available, message = check_llama_cpp_available()

    if not available:
        console.print("\n[bold red]⚠️  GGUF Backend Not Available[/bold red]\n")
        console.print(f"[red]Issue:[/red] {message}\n")
        console.print(LLAMA_CPP_INSTALL_INSTRUCTIONS)
        console.print("\n[bold yellow]Alternative Options:[/bold yellow]")
        console.print("  [cyan]1.[/cyan] Install llama.cpp (see instructions above)")
        console.print(
            "  [cyan]2.[/cyan] Use BitsAndBytes backend (requires NVIDIA GPU):"
        )
        console.print("     [dim]sintra --backend bnb --model-id <model>[/dim]")
        console.print("  [cyan]3.[/cyan] Use ONNX backend (cross-platform):")
        console.print("     [dim]sintra --backend onnx --model-id <model>[/dim]")
        console.print("  [cyan]4.[/cyan] Use mock mode for testing:")
        console.print("     [dim]sintra --mock[/dim]")
        console.print()

        # Ask user if they want to continue
        try:
            response = input("Continue anyway? [y/N]: ").strip().lower()
            if response != "y":
                sys.exit(1)
        except (EOFError, KeyboardInterrupt):
            sys.exit(1)


def _check_backend_dependencies(backend: str) -> None:
    """Check if required dependencies are available for the selected backend.

    Args:
        backend: Backend name ('gguf', 'bnb', 'onnx')

    Raises:
        SystemExit: If required dependencies are missing and user declines to continue
    """
    if backend == "gguf":
        _check_gguf_dependencies()
    elif backend == "bnb":
        _check_bnb_dependencies()
    elif backend == "onnx":
        _check_onnx_dependencies()


def _check_bnb_dependencies():
    """Check if BitsAndBytes is available."""
    try:
        import accelerate  # noqa: F401
        import bitsandbytes  # noqa: F401
        import torch

        if not torch.cuda.is_available():
            console.print("\n[bold yellow]⚠️  BitsAndBytes Warning[/bold yellow]\n")
            console.print(
                "[yellow]CUDA is not available. BitsAndBytes requires an NVIDIA GPU.[/yellow]"
            )
            console.print("\n[bold]Options:[/bold]")
            console.print("  [cyan]1.[/cyan] Use GGUF backend instead (CPU/Metal):")
            console.print("     [dim]sintra --backend gguf --model-id <model>[/dim]")
            console.print("  [cyan]2.[/cyan] Use ONNX backend:")
            console.print("     [dim]sintra --backend onnx --model-id <model>[/dim]")
            console.print()

            try:
                response = input("Continue anyway? [y/N]: ").strip().lower()
                if response != "y":
                    sys.exit(1)
            except (EOFError, KeyboardInterrupt):
                sys.exit(1)

    except ImportError:
        console.print("\n[bold red]⚠️  BitsAndBytes Not Installed[/bold red]\n")
        console.print(
            "[red]The BitsAndBytes backend requires additional packages.[/red]"
        )
        console.print("\n[bold]Install with:[/bold]")
        console.print("  [cyan]pip install sintra[bnb][/cyan]")
        console.print("  [dim]or: pip install bitsandbytes accelerate[/dim]")
        console.print("\n[bold]Alternative backends:[/bold]")
        console.print("  [cyan]--backend gguf[/cyan]  CPU/Metal via llama.cpp")
        console.print("  [cyan]--backend onnx[/cyan]  Cross-platform via ONNX Runtime")
        console.print()
        sys.exit(1)


def _check_onnx_dependencies():
    """Check if ONNX/Optimum is available."""
    try:
        import onnx  # noqa: F401
        import optimum  # noqa: F401
    except ImportError:
        console.print("\n[bold red]⚠️  ONNX Backend Not Installed[/bold red]\n")
        console.print("[red]The ONNX backend requires additional packages.[/red]")
        console.print("\n[bold]Install with:[/bold]")
        console.print("  [cyan]pip install sintra[onnx][/cyan]")
        console.print("  [dim]or: pip install optimum[onnxruntime] onnx[/dim]")
        console.print("\n[bold]Alternative backends:[/bold]")
        console.print(
            "  [cyan]--backend gguf[/cyan]  CPU/Metal via llama.cpp (default)"
        )
        console.print("  [cyan]--backend bnb[/cyan]   GPU-accelerated via BitsAndBytes")
        console.print()
        sys.exit(1)


if __name__ == "__main__":
    main()
