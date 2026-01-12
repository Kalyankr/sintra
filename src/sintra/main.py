import sys

from langgraph.graph import END, StateGraph

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
from sintra.profiles.models import LLMConfig, LLMProvider
from sintra.profiles.parser import ProfileLoadError, load_hardware_profile
from sintra.ui.console import console, log_transition


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

    # Load Hardware Context
    try:
        profile = load_hardware_profile(args.profile)
    except ProfileLoadError as e:
        console.print(f"[status.fail] Failed to load hardware profile: {e}")
        sys.exit(1)

    # Initialize State
    initial_state: SintraState = {
        "profile": profile,
        "llm_config": LLMConfig(
            provider=LLMProvider(args.provider),
            model_name=args.model,
        ),
        "use_debug": args.debug,
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

    # Run Workflow
    app = build_sintra_workflow()

    # Streaming the graph for real-time console updates
    try:
        for _ in app.stream(initial_state, config={"recursion_limit": 50}):
            pass
        console.rule("[status.success] OPTIMIZATION COMPLETE")
    except LLMConnectionError as e:
        console.print(f"\n[bold red]✗ LLM Connection Failed[/bold red]")
        console.print(f"  {e}")
        console.print("\n[dim]Suggestions:[/dim]")
        if args.provider == "ollama":
            console.print("  • Start Ollama: [cyan]ollama serve[/cyan]")
            console.print("  • Or use debug mode: [cyan]sintra --debug <profile>[/cyan]")
        console.print("  • Or try a different provider: [cyan]--provider openai --model gpt-4o[/cyan]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Optimization cancelled by user[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
