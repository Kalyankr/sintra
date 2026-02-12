"""Gradio web UI dashboard for Sintra.

Provides an interactive web interface for:
- Viewing optimization history and experiment results
- Monitoring live optimization runs
- Comparing compression recipes side-by-side
- Browsing hardware profiles
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def check_gradio_available() -> bool:
    """Check if gradio is installed."""
    try:
        import gradio  # noqa: F401

        return True
    except ImportError:
        return False


def format_experiment_table(experiments: list[Any]) -> list[list[str]]:
    """Format experiment records into a table for display.

    Args:
        experiments: List of ExperimentRecord objects

    Returns:
        List of rows for tabular display
    """
    rows = []
    for exp in experiments:
        status = "✓" if exp.result.was_successful else "✗"
        rows.append(
            [
                status,
                str(exp.iteration),
                str(exp.recipe.bits),
                f"{exp.recipe.pruning_ratio:.0%}",
                str(len(exp.recipe.layers_to_drop)),
                f"{exp.result.actual_tps:.1f}",
                f"{exp.result.accuracy_score:.2f}",
                f"{exp.result.actual_vram_usage:.2f}",
                exp.backend,
                exp.created_at.strftime("%H:%M:%S"),
            ]
        )
    return rows


def format_run_info(run: dict) -> str:
    """Format a run metadata dict into a displayable string.

    Args:
        run: Run metadata dictionary from the database

    Returns:
        Formatted string
    """
    if not run:
        return "No run selected"

    lines = [
        f"**Run ID:** {run.get('run_id', 'N/A')[:12]}...",
        f"**Model:** {run.get('model_id', 'N/A')}",
        f"**Hardware:** {run.get('hardware_name', 'N/A')}",
        f"**Backend:** {run.get('backend', 'N/A')}",
        f"**Status:** {run.get('status', 'N/A')}",
        f"**Started:** {run.get('started_at', 'N/A')}",
        f"**Finished:** {run.get('finished_at', 'N/A') or 'In progress'}",
        f"**Iterations:** {run.get('final_iteration', 0)}",
        f"**Converged:** {'Yes' if run.get('is_converged') else 'No'}",
    ]
    return "\n".join(lines)


def create_dashboard(host: str = "127.0.0.1", port: int = 7860) -> None:
    """Create and launch the Gradio dashboard.

    Args:
        host: Host to bind to
        port: Port to listen on
    """
    try:
        import gradio as gr
    except ImportError:
        print("\n❌ Gradio not installed. Install with:")
        print("   pip install sintra[ui]")
        print("   or: pip install gradio>=4.0.0")
        return

    from sintra.persistence import get_history_db

    db = get_history_db()

    # ── Tab 1: History Explorer ──

    def load_history(model_filter: str, hardware_filter: str) -> list[list[str]]:
        """Load experiment history with optional filters."""
        experiments = db.find_similar_experiments(
            model_id=model_filter or "%",
            hardware_name=hardware_filter if hardware_filter else None,
            limit=100,
        )
        return format_experiment_table(experiments)

    def load_stats() -> str:
        """Load database statistics."""
        stats = db.get_statistics()
        return (
            f"**Total Experiments:** {stats['total_experiments']}\n"
            f"**Successful:** {stats['successful_experiments']}\n"
            f"**Success Rate:** {stats['success_rate']:.0%}\n"
            f"**Total Runs:** {stats['total_runs']}\n"
            f"**Unique Models:** {stats['unique_models']}"
        )

    # ── Tab 2: Run Details ──

    def list_runs() -> list[list[str]]:
        """List all runs."""
        try:
            from sintra.persistence import get_history_db

            db_inst = get_history_db()
            # Get recent runs by querying incomplete + we need a general query
            rows = []
            with db_inst._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT run_id, model_id, hardware_name, backend, status, "
                    "started_at, final_iteration FROM runs ORDER BY started_at DESC LIMIT 50"
                )
                for row in cursor.fetchall():
                    rows.append(
                        [
                            str(row["run_id"])[:12],
                            str(row["model_id"]),
                            str(row["hardware_name"]),
                            str(row["backend"]),
                            str(row["status"]),
                            str(row["started_at"]),
                            str(row["final_iteration"]),
                        ]
                    )
            return rows
        except Exception as e:
            return [[f"Error: {e}", "", "", "", "", "", ""]]

    def get_run_details(run_id_short: str) -> str:
        """Get details for a specific run."""
        try:
            with db._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM runs WHERE run_id LIKE ?",
                    (f"{run_id_short}%",),
                ).fetchone()
            if row:
                return format_run_info(dict(row))
            return "Run not found"
        except Exception as e:
            return f"Error: {e}"

    def get_run_experiments(run_id_short: str) -> list[list[str]]:
        """Get experiments for a run."""
        try:
            with db._get_connection() as conn:
                row = conn.execute(
                    "SELECT run_id FROM runs WHERE run_id LIKE ?",
                    (f"{run_id_short}%",),
                ).fetchone()
            if not row:
                return []
            experiments = db.get_experiments_for_run(row["run_id"])
            return format_experiment_table(experiments)
        except Exception as e:
            return [[f"Error: {e}"] + [""] * 9]

    # ── Tab 3: Profile Viewer ──

    def list_profiles() -> list[list[str]]:
        """List available hardware profiles."""
        profiles_dir = Path("profiles")
        rows = []
        if profiles_dir.exists():
            for path in sorted(profiles_dir.glob("*.yaml")):
                try:
                    import yaml

                    with open(path) as f:
                        data = yaml.safe_load(f)
                    name = data.get("name", path.stem)
                    vram = data.get("constraints", {}).get("vram_gb", "?")
                    cpu = data.get("constraints", {}).get("cpu_arch", "?")
                    rows.append([str(path.name), name, str(vram), cpu])
                except Exception:
                    rows.append([str(path.name), "Error loading", "", ""])
        return rows or [["No profiles found", "", "", ""]]

    def load_profile_yaml(filename: str) -> str:
        """Load a profile YAML file contents."""
        path = Path("profiles") / filename
        if path.exists():
            return path.read_text()
        return "Profile not found"

    # ── Build the UI ──

    experiment_headers = [
        "Status",
        "Iter",
        "Bits",
        "Pruning",
        "Layers Dropped",
        "TPS",
        "Accuracy",
        "VRAM (GB)",
        "Backend",
        "Time",
    ]

    with gr.Blocks(
        title="Sintra Dashboard",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# 🔧 Sintra Dashboard\n*LLM Optimization Monitor*")

        with gr.Tab("📊 History"):
            with gr.Row():
                model_input = gr.Textbox(
                    label="Filter by Model ID", placeholder="e.g., llama"
                )
                hw_input = gr.Textbox(
                    label="Filter by Hardware", placeholder="e.g., Mac Mini"
                )
                search_btn = gr.Button("Search", variant="primary")

            stats_md = gr.Markdown()
            history_table = gr.Dataframe(
                headers=experiment_headers,
                label="Experiment History",
            )

            search_btn.click(
                load_history,
                inputs=[model_input, hw_input],
                outputs=[history_table],
            )
            app.load(load_stats, outputs=[stats_md])

        with gr.Tab("🏃 Runs"):
            runs_table = gr.Dataframe(
                headers=[
                    "Run ID",
                    "Model",
                    "Hardware",
                    "Backend",
                    "Status",
                    "Started",
                    "Iterations",
                ],
                label="Optimization Runs",
            )
            app.load(list_runs, outputs=[runs_table])

            with gr.Row():
                run_id_input = gr.Textbox(label="Run ID (first 12 chars)")
                detail_btn = gr.Button("Load Details")

            run_info_md = gr.Markdown(label="Run Details")
            run_experiments_table = gr.Dataframe(
                headers=experiment_headers,
                label="Run Experiments",
            )

            detail_btn.click(
                get_run_details, inputs=[run_id_input], outputs=[run_info_md]
            )
            detail_btn.click(
                get_run_experiments,
                inputs=[run_id_input],
                outputs=[run_experiments_table],
            )

        with gr.Tab("🖥️ Profiles"):
            profiles_table = gr.Dataframe(
                headers=["Filename", "Name", "VRAM (GB)", "CPU Arch"],
                label="Hardware Profiles",
            )
            app.load(list_profiles, outputs=[profiles_table])

            with gr.Row():
                profile_select = gr.Textbox(
                    label="Profile Filename", placeholder="e.g., raspberry_pi_5.yaml"
                )
                load_btn = gr.Button("Load Profile")

            profile_yaml = gr.Code(label="Profile YAML", language="yaml")
            load_btn.click(
                load_profile_yaml,
                inputs=[profile_select],
                outputs=[profile_yaml],
            )

        with gr.Tab("About"):
            gr.Markdown("""
## Sintra: Autonomous Edge AI Distiller

Sintra is an autonomous AI agent that optimizes Large Language Models
for edge hardware deployment. It uses LLM-driven architects to iteratively
propose compression strategies, benchmark them, and converge on optimal
configurations.

### Features
- **Multi-agent collaboration** — Specialized experts for quantization, pruning, and integration
- **ReAct reasoning** — Thought → Action → Observation reasoning chains
- **Self-reflection** — Analyzes failures and adjusts strategy
- **Adaptive learning** — Improves estimates from historical data
- **Multiple backends** — GGUF, BitsAndBytes, ONNX

### Commands
```
sintra --auto-detect                    # Optimize with all features
sintra --auto-detect --simple           # Basic mode
sintra --auto-detect --dry-run          # Preview only
sintra --auto-detect --ui               # Launch this dashboard
```
            """)

    print(f"\n🚀 Starting Sintra Dashboard at http://{host}:{port}\n")
    app.launch(server_name=host, server_port=port)
