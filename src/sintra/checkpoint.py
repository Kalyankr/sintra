"""Checkpoint management for resumable optimization runs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sintra.profiles.models import HardwareProfile, LLMConfig, ModelRecipe


def get_checkpoint_dir() -> Path:
    """Get the default checkpoint directory."""
    checkpoint_dir = Path.home() / ".sintra" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_checkpoint_path(run_id: str) -> Path:
    """Get the path to a checkpoint file for a given run."""
    return get_checkpoint_dir() / f"{run_id}.json"


def serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Convert state to JSON-serializable format."""
    serialized = {}
    
    for key, value in state.items():
        if key == "profile" and isinstance(value, HardwareProfile):
            serialized[key] = {
                "name": value.name,
                "constraints": {
                    "vram_gb": value.constraints.vram_gb,
                    "cpu_arch": value.constraints.cpu_arch,
                    "has_cuda": value.constraints.has_cuda,
                },
                "targets": {
                    "min_tokens_per_second": value.targets.min_tokens_per_second,
                    "min_accuracy_score": value.targets.min_accuracy_score,
                    "max_latency_ms": value.targets.max_latency_ms,
                },
                "supported_quantizations": value.supported_quantizations,
            }
        elif key == "llm_config" and isinstance(value, LLMConfig):
            serialized[key] = {
                "provider": value.provider.value,
                "model_name": value.model_name,
            }
        elif key == "current_recipe" and value is not None:
            serialized[key] = value.model_dump() if hasattr(value, "model_dump") else value
        elif key == "best_recipe" and value is not None:
            if isinstance(value, dict):
                serialized[key] = {
                    "recipe": value["recipe"].model_dump() if hasattr(value["recipe"], "model_dump") else value["recipe"],
                    "metrics": {
                        "actual_tps": value["metrics"].actual_tps,
                        "accuracy_score": value["metrics"].accuracy_score,
                        "peak_vram_gb": value["metrics"].peak_vram_gb,
                        "was_successful": value["metrics"].was_successful,
                    } if hasattr(value["metrics"], "actual_tps") else value["metrics"],
                }
            else:
                serialized[key] = value
        elif key == "history" and isinstance(value, list):
            serialized[key] = []
            for entry in value:
                if isinstance(entry, dict):
                    serialized_entry = {
                        "recipe": entry["recipe"].model_dump() if hasattr(entry["recipe"], "model_dump") else entry["recipe"],
                        "metrics": {
                            "actual_tps": entry["metrics"].actual_tps,
                            "accuracy_score": entry["metrics"].accuracy_score,
                            "peak_vram_gb": entry["metrics"].peak_vram_gb,
                            "was_successful": entry["metrics"].was_successful,
                        } if hasattr(entry["metrics"], "actual_tps") else entry["metrics"],
                    }
                    serialized[key].append(serialized_entry)
                else:
                    serialized[key].append(entry)
        else:
            serialized[key] = value
    
    return serialized


def deserialize_state(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON data back to state format with proper types."""
    from sintra.profiles.models import (
        Constraints,
        ExperimentResult,
        HardwareProfile,
        LLMConfig,
        LLMProvider,
        ModelRecipe,
        Targets,
    )
    
    deserialized = {}
    
    for key, value in data.items():
        if key == "profile" and isinstance(value, dict):
            deserialized[key] = HardwareProfile(
                name=value["name"],
                constraints=Constraints(**value["constraints"]),
                targets=Targets(**value["targets"]),
                supported_quantizations=value.get("supported_quantizations"),
            )
        elif key == "llm_config" and isinstance(value, dict):
            deserialized[key] = LLMConfig(
                provider=LLMProvider(value["provider"]),
                model_name=value["model_name"],
            )
        elif key == "current_recipe" and value is not None:
            deserialized[key] = ModelRecipe(**value)
        elif key == "best_recipe" and value is not None:
            deserialized[key] = {
                "recipe": ModelRecipe(**value["recipe"]),
                "metrics": ExperimentResult(**value["metrics"]),
            }
        elif key == "history" and isinstance(value, list):
            deserialized[key] = []
            for entry in value:
                deserialized[key].append({
                    "recipe": ModelRecipe(**entry["recipe"]),
                    "metrics": ExperimentResult(**entry["metrics"]),
                })
        else:
            deserialized[key] = value
    
    return deserialized


def save_checkpoint(run_id: str, state: Dict[str, Any], iteration: int) -> Path:
    """Save a checkpoint of the current state.
    
    Args:
        run_id: Unique identifier for this optimization run
        state: Current workflow state
        iteration: Current iteration number
        
    Returns:
        Path to the saved checkpoint file
    """
    checkpoint_path = get_checkpoint_path(run_id)
    
    checkpoint_data = {
        "run_id": run_id,
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "state": serialize_state(state),
    }
    
    # Write atomically to prevent corruption
    temp_path = checkpoint_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)
    temp_path.rename(checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(run_id: str) -> Optional[Dict[str, Any]]:
    """Load a checkpoint by run ID.
    
    Args:
        run_id: Unique identifier for the optimization run
        
    Returns:
        Checkpoint data with deserialized state, or None if not found
    """
    checkpoint_path = get_checkpoint_path(run_id)
    
    if not checkpoint_path.exists():
        return None
    
    with open(checkpoint_path, "r") as f:
        data = json.load(f)
    
    # Deserialize the state
    data["state"] = deserialize_state(data["state"])
    
    return data


def find_latest_checkpoint(model_id: str = None) -> Optional[Dict[str, Any]]:
    """Find the most recent checkpoint, optionally filtering by model.
    
    Args:
        model_id: If provided, only consider checkpoints for this model
        
    Returns:
        Most recent checkpoint data, or None if no checkpoints exist
    """
    checkpoint_dir = get_checkpoint_dir()
    
    if not checkpoint_dir.exists():
        return None
    
    latest = None
    latest_time = None
    
    for checkpoint_file in checkpoint_dir.glob("*.json"):
        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
            
            # Filter by model if specified
            if model_id and data.get("state", {}).get("target_model_id") != model_id:
                continue
            
            timestamp = datetime.fromisoformat(data["timestamp"])
            if latest_time is None or timestamp > latest_time:
                latest_time = timestamp
                latest = data
        except (json.JSONDecodeError, KeyError, ValueError):
            # Skip corrupted checkpoints
            continue
    
    if latest:
        latest["state"] = deserialize_state(latest["state"])
    
    return latest


def list_checkpoints() -> List[Dict[str, Any]]:
    """List all available checkpoints.
    
    Returns:
        List of checkpoint metadata (run_id, model_id, iteration, timestamp)
    """
    checkpoint_dir = get_checkpoint_dir()
    
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    
    for checkpoint_file in checkpoint_dir.glob("*.json"):
        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
            
            checkpoints.append({
                "run_id": data["run_id"],
                "model_id": data.get("state", {}).get("target_model_id", "unknown"),
                "iteration": data["iteration"],
                "timestamp": data["timestamp"],
                "path": str(checkpoint_file),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    
    # Sort by timestamp, most recent first
    checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return checkpoints


def delete_checkpoint(run_id: str) -> bool:
    """Delete a checkpoint file.
    
    Args:
        run_id: Unique identifier for the optimization run
        
    Returns:
        True if deleted, False if not found
    """
    checkpoint_path = get_checkpoint_path(run_id)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        return True
    
    return False


def cleanup_old_checkpoints(max_age_days: int = 7, max_count: int = 50) -> int:
    """Remove old checkpoints to save disk space.
    
    Args:
        max_age_days: Delete checkpoints older than this
        max_count: Keep at most this many checkpoints
        
    Returns:
        Number of checkpoints deleted
    """
    checkpoints = list_checkpoints()
    
    if not checkpoints:
        return 0
    
    deleted = 0
    cutoff = datetime.now()
    
    for i, cp in enumerate(checkpoints):
        should_delete = False
        
        # Delete if over max count
        if i >= max_count:
            should_delete = True
        
        # Delete if too old
        try:
            timestamp = datetime.fromisoformat(cp["timestamp"])
            age_days = (cutoff - timestamp).days
            if age_days > max_age_days:
                should_delete = True
        except ValueError:
            pass
        
        if should_delete:
            if delete_checkpoint(cp["run_id"]):
                deleted += 1
    
    return deleted
