"""SQLite persistence for experiment history.

This module provides persistent storage for optimization experiments,
allowing the agent to learn from past runs and avoid repeating failures.
"""

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from sintra.profiles.models import ExperimentResult, HardwareProfile, ModelRecipe

# Default database location
DEFAULT_DB_PATH = Path.home() / ".sintra" / "history.db"


class ExperimentRecord:
    """A record of a single optimization experiment."""

    def __init__(
        self,
        id: int | None,
        run_id: str,
        model_id: str,
        hardware_name: str,
        recipe: ModelRecipe,
        result: ExperimentResult,
        backend: str,
        created_at: datetime,
        iteration: int = 0,
    ):
        self.id = id
        self.run_id = run_id
        self.model_id = model_id
        self.hardware_name = hardware_name
        self.recipe = recipe
        self.result = result
        self.backend = backend
        self.created_at = created_at
        self.iteration = iteration

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "hardware_name": self.hardware_name,
            "recipe": self.recipe.model_dump(),
            "result": self.result.model_dump(),
            "backend": self.backend,
            "created_at": self.created_at.isoformat(),
            "iteration": self.iteration,
        }


class HistoryDB:
    """SQLite database for experiment history.

    Provides persistent storage for:
    - Past experiments (recipes + results)
    - Run metadata (hardware, model, timestamps)
    - Query capabilities for learning from history

    Example:
        >>> db = HistoryDB()
        >>> db.save_experiment(run_id, model_id, hardware, recipe, result, backend)
        >>> similar = db.find_similar_experiments(model_id, hardware_name)
    """

    def __init__(self, db_path: Path | None = None):
        """Initialize the database.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.sintra/history.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Experiments table: stores all optimization attempts
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    hardware_name TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    iteration INTEGER DEFAULT 0,
                    
                    -- Recipe fields
                    bits INTEGER NOT NULL,
                    pruning_ratio REAL NOT NULL,
                    layers_to_drop TEXT NOT NULL,  -- JSON array
                    method TEXT NOT NULL,
                    
                    -- Result fields
                    actual_tps REAL NOT NULL,
                    actual_vram_usage REAL NOT NULL,
                    accuracy_score REAL NOT NULL,
                    was_successful INTEGER NOT NULL,
                    error_log TEXT,
                    
                    -- Metadata
                    created_at TEXT NOT NULL,
                    
                    -- Indexes for common queries
                    UNIQUE(run_id, iteration)
                );
                
                CREATE INDEX IF NOT EXISTS idx_model_hardware 
                    ON experiments(model_id, hardware_name);
                CREATE INDEX IF NOT EXISTS idx_successful 
                    ON experiments(was_successful);
                CREATE INDEX IF NOT EXISTS idx_created 
                    ON experiments(created_at);
                
                -- Runs table: stores run-level metadata
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    model_id TEXT NOT NULL,
                    hardware_name TEXT NOT NULL,
                    hardware_profile TEXT NOT NULL,  -- JSON
                    backend TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    final_iteration INTEGER DEFAULT 0,
                    is_converged INTEGER DEFAULT 0,
                    best_recipe TEXT,  -- JSON
                    status TEXT DEFAULT 'running'  -- running, completed, failed, interrupted
                );
                
                CREATE INDEX IF NOT EXISTS idx_run_status 
                    ON runs(status);
            """)
            conn.commit()

    def save_experiment(
        self,
        run_id: str,
        model_id: str,
        hardware_name: str,
        recipe: ModelRecipe,
        result: ExperimentResult,
        backend: str,
        iteration: int = 0,
    ) -> int:
        """Save an experiment to the database.

        Args:
            run_id: Unique identifier for this optimization run
            model_id: HuggingFace model ID
            hardware_name: Target hardware profile name
            recipe: The compression recipe tried
            result: The benchmark result
            backend: Quantization backend used
            iteration: Iteration number within the run

        Returns:
            The database ID of the saved experiment
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiments (
                    run_id, model_id, hardware_name, backend, iteration,
                    bits, pruning_ratio, layers_to_drop, method,
                    actual_tps, actual_vram_usage, accuracy_score, 
                    was_successful, error_log, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    model_id,
                    hardware_name,
                    backend,
                    iteration,
                    recipe.bits,
                    recipe.pruning_ratio,
                    json.dumps(recipe.layers_to_drop),
                    recipe.method,
                    result.actual_tps,
                    result.actual_vram_usage,
                    result.accuracy_score,
                    1 if result.was_successful else 0,
                    result.error_log,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def start_run(
        self,
        run_id: str,
        model_id: str,
        profile: HardwareProfile,
        backend: str,
    ) -> None:
        """Record the start of an optimization run.

        Args:
            run_id: Unique identifier for this run
            model_id: HuggingFace model ID
            profile: Target hardware profile
            backend: Quantization backend
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, model_id, hardware_name, hardware_profile,
                    backend, started_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'running')
            """,
                (
                    run_id,
                    model_id,
                    profile.name,
                    profile.model_dump_json(),
                    backend,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def finish_run(
        self,
        run_id: str,
        final_iteration: int,
        is_converged: bool,
        best_recipe: ModelRecipe | None = None,
        status: str = "completed",
    ) -> None:
        """Record the completion of an optimization run.

        Args:
            run_id: The run identifier
            final_iteration: Last iteration completed
            is_converged: Whether optimization converged
            best_recipe: The best recipe found (if any)
            status: Final status (completed, failed, interrupted)
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE runs SET
                    finished_at = ?,
                    final_iteration = ?,
                    is_converged = ?,
                    best_recipe = ?,
                    status = ?
                WHERE run_id = ?
            """,
                (
                    datetime.now().isoformat(),
                    final_iteration,
                    1 if is_converged else 0,
                    best_recipe.model_dump_json() if best_recipe else None,
                    status,
                    run_id,
                ),
            )
            conn.commit()

    def find_similar_experiments(
        self,
        model_id: str,
        hardware_name: str | None = None,
        successful_only: bool = False,
        limit: int = 50,
    ) -> list[ExperimentRecord]:
        """Find experiments similar to the current setup.

        Args:
            model_id: HuggingFace model ID to match
            hardware_name: Optional hardware profile to match
            successful_only: Only return successful experiments
            limit: Maximum number of results

        Returns:
            List of matching experiment records
        """
        query = """
            SELECT * FROM experiments
            WHERE model_id = ?
        """
        params: list = [model_id]

        if hardware_name:
            query += " AND hardware_name = ?"
            params.append(hardware_name)

        if successful_only:
            query += " AND was_successful = 1"

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_best_recipe_for_hardware(
        self,
        model_id: str,
        hardware_name: str,
    ) -> tuple[ModelRecipe, ExperimentResult] | None:
        """Get the best successful recipe for a model/hardware combo.

        "Best" = highest accuracy among successful experiments.

        Args:
            model_id: HuggingFace model ID
            hardware_name: Hardware profile name

        Returns:
            Tuple of (recipe, result) or None if no successful experiments
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM experiments
                WHERE model_id = ? 
                  AND hardware_name = ?
                  AND was_successful = 1
                ORDER BY accuracy_score DESC
                LIMIT 1
            """,
                (model_id, hardware_name),
            ).fetchone()

        if row is None:
            return None

        record = self._row_to_record(row)
        return (record.recipe, record.result)

    def get_failed_recipes(
        self,
        model_id: str,
        hardware_name: str | None = None,
    ) -> list[ModelRecipe]:
        """Get recipes that failed for a model/hardware combo.

        Useful for avoiding repeated failures.

        Args:
            model_id: HuggingFace model ID
            hardware_name: Optional hardware profile name

        Returns:
            List of recipes that failed
        """
        query = """
            SELECT bits, pruning_ratio, layers_to_drop, method
            FROM experiments
            WHERE model_id = ? AND was_successful = 0
        """
        params: list = [model_id]

        if hardware_name:
            query += " AND hardware_name = ?"
            params.append(hardware_name)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            ModelRecipe(
                bits=row["bits"],
                pruning_ratio=row["pruning_ratio"],
                layers_to_drop=json.loads(row["layers_to_drop"]),
                method=row["method"],
            )
            for row in rows
        ]

    def get_run(self, run_id: str) -> dict | None:
        """Get run metadata by ID.

        Args:
            run_id: The run identifier

        Returns:
            Run metadata dict or None
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def get_experiments_for_run(self, run_id: str) -> list[ExperimentRecord]:
        """Get all experiments for a specific run.

        Args:
            run_id: The run identifier

        Returns:
            List of experiments in iteration order
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM experiments
                WHERE run_id = ?
                ORDER BY iteration ASC
            """,
                (run_id,),
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_incomplete_runs(self) -> list[dict]:
        """Get runs that didn't complete (for potential resume).

        Returns:
            List of incomplete run metadata dicts
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM runs
                WHERE status = 'running'
                ORDER BY started_at DESC
            """).fetchall()

        return [dict(row) for row in rows]

    def get_statistics(self) -> dict:
        """Get overall statistics about experiment history.

        Returns:
            Dictionary with stats about experiments and runs
        """
        with self._get_connection() as conn:
            exp_count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

            success_count = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE was_successful = 1"
            ).fetchone()[0]

            run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

            model_count = conn.execute(
                "SELECT COUNT(DISTINCT model_id) FROM experiments"
            ).fetchone()[0]

        return {
            "total_experiments": exp_count,
            "successful_experiments": success_count,
            "success_rate": success_count / exp_count if exp_count > 0 else 0,
            "total_runs": run_count,
            "unique_models": model_count,
        }

    def _row_to_record(self, row: sqlite3.Row) -> ExperimentRecord:
        """Convert a database row to an ExperimentRecord."""
        recipe = ModelRecipe(
            bits=row["bits"],
            pruning_ratio=row["pruning_ratio"],
            layers_to_drop=json.loads(row["layers_to_drop"]),
            method=row["method"],
        )

        result = ExperimentResult(
            actual_tps=row["actual_tps"],
            actual_vram_usage=row["actual_vram_usage"],
            accuracy_score=row["accuracy_score"],
            was_successful=bool(row["was_successful"]),
            error_log=row["error_log"],
        )

        return ExperimentRecord(
            id=row["id"],
            run_id=row["run_id"],
            model_id=row["model_id"],
            hardware_name=row["hardware_name"],
            recipe=recipe,
            result=result,
            backend=row["backend"],
            created_at=datetime.fromisoformat(row["created_at"]),
            iteration=row["iteration"],
        )


# Convenience functions for global database access
_global_db: HistoryDB | None = None


def get_history_db(db_path: Path | None = None) -> HistoryDB:
    """Get the global history database instance.

    Args:
        db_path: Optional custom database path

    Returns:
        The HistoryDB instance
    """
    global _global_db
    if _global_db is None or db_path is not None:
        _global_db = HistoryDB(db_path)
    return _global_db


def format_history_from_db(
    model_id: str,
    hardware_name: str,
    limit: int = 10,
) -> str:
    """Format experiment history from DB for LLM context.

    Args:
        model_id: HuggingFace model ID
        hardware_name: Hardware profile name
        limit: Max experiments to include

    Returns:
        Formatted string for LLM prompt
    """
    db = get_history_db()
    experiments = db.find_similar_experiments(model_id, hardware_name, limit=limit)

    if not experiments:
        return "No previous experiments found for this model/hardware combo."

    lines = ["Previous experiments from history:"]
    for exp in experiments:
        status = "✓" if exp.result.was_successful else "✗"
        lines.append(
            f"  {status} bits={exp.recipe.bits}, prune={exp.recipe.pruning_ratio:.0%}, "
            f"layers_dropped={len(exp.recipe.layers_to_drop)} → "
            f"TPS={exp.result.actual_tps:.1f}, acc={exp.result.accuracy_score:.2f}"
        )

    return "\n".join(lines)
