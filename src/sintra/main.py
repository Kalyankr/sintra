import sys

from langgraph.graph import END, StateGraph

from sintra.agents.nodes import (
    architect_node,
    benchmarker_node,
    critic_router,
    reporter_node,
)
from sintra.agents.state import SintraState
from sintra.profiles.models import LLMConfig, LLMProvider
from sintra.profiles.parser import load_hardware_profile
from sintra.ui.console import console, log_transition


def build_sintra_workflow():
    workflow = StateGraph(SintraState)

    # Define the "Actors"
    workflow.add_node("architect", architect_node)
    workflow.add_node("benchmarker", benchmarker_node)
    workflow.add_node("reporter", reporter_node)

    # Define the "Path"
    workflow.set_entry_point("architect")
    workflow.add_edge("architect", "benchmarker")

    # The Decision Point: Critic determines if we repeat or stop
    workflow.add_conditional_edges(
        "benchmarker",
        critic_router,
        {
            "continue": "architect",
            "end": "reporter",
        },
    )
    # reporter leads to END
    workflow.add_edge("reporter", END)

    return workflow.compile()


def run_cli(profile_path: str):
    """Entry point to initiate the agentic loop."""
    console.rule("[arch.node] SINTRA: Edge AI Distiller")

    # Load context
    log_transition("System", f"Loading hardware profile: {profile_path}", "hw.profile")
    profile = load_hardware_profile(profile_path)

    # Initial State Setup
    initial_state: SintraState = {
        "profile": profile,
        "llm_config": LLMConfig(
            provider=LLMProvider.GOOGLE, model_name="gemini-2.5-flash"
        ),
        "current_recipe": None,
        "history": [],
        "iteration": 0,
        "is_converged": False,
    }

    # Compile and Stream
    app = build_sintra_workflow()

    log_transition("System", "Starting optimization cycle...", "status.success")

    # We use .stream to see the transition between nodes in real-time
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            # The UI logic is handled inside the nodes via log_transition
            pass

    console.rule("[status.success] OPTIMIZATION COMPLETE")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print(
            "[status.fail] Usage: uv run python -m sintra.main profiles/pi5.yaml"
        )
    else:
        run_cli(sys.argv[1])
