"""Model pruning and layer dropping for LLM compression.

Provides two main strategies:
1. Layer Dropping: Remove entire transformer layers before GGUF conversion
2. Structured Pruning: Zero out or remove attention/FFN weights by ratio

Both operations are applied to HuggingFace models BEFORE GGUF conversion.
"""

import json
import logging
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sintra"


# Pre-compiled regex patterns for layer index extraction (shared by dropper & pruner)
_LAYER_INDEX_PATTERNS = [
    re.compile(r"\.layers\.(\d+)\."),  # Llama, Mistral, etc.
    re.compile(r"\.h\.(\d+)\."),  # GPT-2, GPT-J
    re.compile(r"\.layer\.(\d+)\."),  # BERT, RoBERTa
    re.compile(r"\.blocks\.(\d+)\."),  # Some vision models
    re.compile(r"\.transformer\.(\d+)\."),  # Some architectures
]

# Pre-compiled replacement patterns (old_pattern, format_string)
_LAYER_REPLACE_PATTERNS = [
    (re.compile(r"\.layers\.(\d+)\."), ".layers.{}."),
    (re.compile(r"\.h\.(\d+)\."), ".h.{}."),
    (re.compile(r"\.layer\.(\d+)\."), ".layer.{}."),
    (re.compile(r"\.blocks\.(\d+)\."), ".blocks.{}."),
    (re.compile(r"\.transformer\.(\d+)\."), ".transformer.{}."),
]


def _extract_layer_index(key: str) -> int | None:
    """Extract layer index from a tensor key.

    Handles various naming conventions:
    - model.layers.0.self_attn.q_proj.weight (Llama)
    - transformer.h.0.attn.c_attn.weight (GPT-2)
    - encoder.layer.0.attention.self.query.weight (BERT)
    """
    for pattern in _LAYER_INDEX_PATTERNS:
        match = pattern.search(key)
        if match:
            return int(match.group(1))
    return None


def _replace_layer_index(key: str, old_idx: int, new_idx: int) -> str:
    """Replace layer index in a tensor key."""
    for pattern, fmt in _LAYER_REPLACE_PATTERNS:
        match = pattern.search(key)
        if match and int(match.group(1)) == old_idx:
            return pattern.sub(fmt.format(new_idx), key)
    return key


def _copy_non_weight_files(src: Path, dst: Path) -> None:
    """Copy config and other non-weight files between model directories."""
    for file in src.iterdir():
        if (
            file.is_file()
            and file.suffix not in [".safetensors", ".bin"]
            and file.name not in ["model.safetensors.index.json"]
        ):
            shutil.copy2(file, dst / file.name)


class PruningError(Exception):
    """Raised when pruning or layer dropping fails."""

    pass


class ModelConfig:
    """Helper class to read and modify model configuration."""

    def __init__(self, config_path: Path):
        """Load model config from JSON file."""
        if not config_path.exists():
            raise PruningError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            self.config = json.load(f)
        self.config_path = config_path

    @property
    def num_layers(self) -> int:
        """Get the number of transformer layers."""
        # Different architectures use different key names
        for key in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
            if key in self.config:
                return self.config[key]
        raise PruningError("Could not determine number of layers from config")

    @num_layers.setter
    def num_layers(self, value: int):
        """Set the number of transformer layers."""
        for key in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
            if key in self.config:
                self.config[key] = value
                return
        # Default to most common key
        self.config["num_hidden_layers"] = value

    @property
    def architecture(self) -> str:
        """Get model architecture type."""
        archs = self.config.get("architectures", [])
        if archs:
            return archs[0]
        return self.config.get("model_type", "unknown")

    def save(self, output_path: Path | None = None):
        """Save config to file."""
        save_path = output_path or self.config_path
        with open(save_path, "w") as f:
            json.dump(self.config, f, indent=2)


class LayerDropper:
    """Removes transformer layers from HuggingFace models.

    Layer dropping is an aggressive compression technique that removes
    entire transformer blocks. This reduces model size and increases
    inference speed, but can significantly impact quality.

    Guidelines:
    - Dropping 10-20% of layers: Minimal quality loss
    - Dropping 20-40% of layers: Noticeable degradation
    - Dropping >40% of layers: Significant quality loss

    Example:
        >>> dropper = LayerDropper()
        >>> pruned_path = dropper.drop_layers(
        ...     model_path="/path/to/model",
        ...     layers_to_drop=[0, 1, 31],  # Drop first, second, and last layers
        ... )
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the layer dropper.

        Args:
            cache_dir: Base cache directory. Defaults to ~/.cache/sintra/
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.pruned_dir = self.cache_dir / "pruned"
        self.pruned_dir.mkdir(parents=True, exist_ok=True)

    def drop_layers(
        self,
        model_path: Path,
        layers_to_drop: list[int],
        output_name: str | None = None,
    ) -> Path:
        """Remove specified layers from a model.

        Creates a new model directory with the specified layers removed.
        Modifies both the weights and the config.json.

        Args:
            model_path: Path to HuggingFace model directory
            layers_to_drop: List of layer indices to remove (0-indexed)
            output_name: Name for output directory

        Returns:
            Path to the pruned model directory

        Raises:
            PruningError: If layer dropping fails
        """
        if not model_path.is_dir():
            raise PruningError(f"Model path must be a directory: {model_path}")

        if not layers_to_drop:
            logger.info("No layers to drop, returning original model")
            return model_path

        # Load config to get layer count
        config_path = model_path / "config.json"
        config = ModelConfig(config_path)
        num_layers = config.num_layers

        # Validate layer indices
        invalid_layers = [idx for idx in layers_to_drop if idx < 0 or idx >= num_layers]
        if invalid_layers:
            raise PruningError(
                f"Invalid layer indices {invalid_layers}. "
                f"Model has {num_layers} layers (0-{num_layers - 1})"
            )

        # Create output directory
        if output_name is None:
            output_name = f"{model_path.name}-dropped{len(layers_to_drop)}"
        output_path = self.pruned_dir / output_name

        # Check if already exists
        if output_path.exists() and (output_path / "config.json").exists():
            logger.info(f"Pruned model already exists: {output_path}")
            return output_path

        logger.info(
            f"Dropping layers {layers_to_drop} from {model_path.name} "
            f"({num_layers} -> {num_layers - len(layers_to_drop)} layers)"
        )

        # Create output directory and copy non-weight files
        output_path.mkdir(parents=True, exist_ok=True)
        self._copy_non_weight_files(model_path, output_path)

        # Process weight files
        weight_files = list(model_path.glob("*.safetensors"))
        if not weight_files:
            # Try .bin files (older format)
            weight_files = list(model_path.glob("*.bin"))
            if not weight_files:
                raise PruningError("No weight files found (.safetensors or .bin)")

        # Drop layers from weights
        layers_kept = [i for i in range(num_layers) if i not in layers_to_drop]
        self._drop_layers_from_weights(
            model_path, output_path, layers_to_drop, layers_kept
        )

        # Update config with new layer count
        new_config = ModelConfig(output_path / "config.json")
        new_config.num_layers = len(layers_kept)
        new_config.save()

        logger.info(f"Layer-dropped model saved to {output_path}")
        return output_path

    def _copy_non_weight_files(self, src: Path, dst: Path):
        """Copy config and other non-weight files."""
        _copy_non_weight_files(src, dst)

    def _drop_layers_from_weights(
        self,
        model_path: Path,
        output_path: Path,
        layers_to_drop: list[int],
        layers_kept: list[int],
    ):
        """Remove specified layers from weight files.

        This renumbers the remaining layers to be sequential.
        """
        weight_files = list(model_path.glob("*.safetensors"))

        if weight_files:
            self._process_safetensors(
                weight_files, output_path, layers_to_drop, layers_kept
            )
        else:
            # Fallback to .bin files
            bin_files = list(model_path.glob("*.bin"))
            if bin_files:
                self._process_bin_files(
                    bin_files, output_path, layers_to_drop, layers_kept
                )

    def _process_safetensors(
        self,
        weight_files: list[Path],
        output_path: Path,
        layers_to_drop: list[int],
        layers_kept: list[int],
    ):
        """Process safetensors files and remove specified layers."""
        # Build layer index mapping (old -> new)
        layer_mapping = {old: new for new, old in enumerate(layers_kept)}

        # Collect all tensors from all files
        all_tensors = {}

        for wf in weight_files:
            with safe_open(wf, framework="pt", device="cpu") as f:
                for key in f:
                    tensor = f.get_tensor(key)

                    # Check if this tensor belongs to a layer we want to drop
                    layer_idx = _extract_layer_index(key)

                    if layer_idx is not None:
                        if layer_idx in layers_to_drop:
                            # Skip this tensor - it belongs to a dropped layer
                            continue
                        else:
                            # Renumber the layer in the key
                            new_idx = layer_mapping[layer_idx]
                            new_key = _replace_layer_index(key, layer_idx, new_idx)
                            all_tensors[new_key] = tensor
                    else:
                        # Non-layer tensor (embeddings, final norm, etc.)
                        all_tensors[key] = tensor

        # Save to single output file
        output_file = output_path / "model.safetensors"
        save_file(all_tensors, output_file)
        logger.info(f"Saved {len(all_tensors)} tensors to {output_file}")

    def _process_bin_files(
        self,
        bin_files: list[Path],
        output_path: Path,
        layers_to_drop: list[int],
        layers_kept: list[int],
    ):
        """Process .bin (PyTorch) files and remove specified layers."""
        layer_mapping = {old: new for new, old in enumerate(layers_kept)}
        all_tensors = {}

        for bf in bin_files:
            state_dict = torch.load(bf, map_location="cpu", weights_only=True)

            for key, tensor in state_dict.items():
                layer_idx = _extract_layer_index(key)

                if layer_idx is not None:
                    if layer_idx in layers_to_drop:
                        continue
                    else:
                        new_idx = layer_mapping[layer_idx]
                        new_key = _replace_layer_index(key, layer_idx, new_idx)
                        all_tensors[new_key] = tensor
                else:
                    all_tensors[key] = tensor

        # Save as safetensors (more efficient)
        output_file = output_path / "model.safetensors"
        save_file(all_tensors, output_file)
        logger.info(f"Saved {len(all_tensors)} tensors to {output_file}")

    def _extract_layer_index(self, key: str) -> int | None:
        """Extract layer index from a tensor key. Delegates to module-level function."""
        return _extract_layer_index(key)

    def _replace_layer_index(self, key: str, old_idx: int, new_idx: int) -> str:
        """Replace layer index in a tensor key. Delegates to module-level function."""
        return _replace_layer_index(key, old_idx, new_idx)


class StructuredPruner:
    """Applies structured pruning to model weights.

    Structured pruning removes entire neurons, attention heads, or channels
    rather than individual weights. This maintains model structure and allows
    efficient inference without sparse matrix operations.

    Pruning strategies:
    - Magnitude-based: Remove weights with smallest absolute values
    - Random: Randomly zero out weights (for comparison)

    Example:
        >>> pruner = StructuredPruner()
        >>> pruned_path = pruner.prune(
        ...     model_path="/path/to/model",
        ...     pruning_ratio=0.2,  # Remove 20% of weights
        ... )
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the structured pruner.

        Args:
            cache_dir: Base cache directory. Defaults to ~/.cache/sintra/
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.pruned_dir = self.cache_dir / "pruned"
        self.pruned_dir.mkdir(parents=True, exist_ok=True)

    def prune(
        self,
        model_path: Path,
        pruning_ratio: float,
        output_name: str | None = None,
        strategy: str = "magnitude",
    ) -> Path:
        """Apply structured pruning to model weights.

        Args:
            model_path: Path to HuggingFace model directory
            pruning_ratio: Fraction of weights to prune (0.0-1.0)
            output_name: Name for output directory
            strategy: Pruning strategy ("magnitude" or "random")

        Returns:
            Path to the pruned model directory

        Raises:
            PruningError: If pruning fails
        """
        if not model_path.is_dir():
            raise PruningError(f"Model path must be a directory: {model_path}")

        if pruning_ratio <= 0.0:
            logger.info("Pruning ratio is 0, returning original model")
            return model_path

        if pruning_ratio >= 1.0:
            raise PruningError("Pruning ratio must be less than 1.0")

        # Create output directory
        ratio_str = f"{int(pruning_ratio * 100)}pct"
        if output_name is None:
            output_name = f"{model_path.name}-pruned{ratio_str}"
        output_path = self.pruned_dir / output_name

        # Check if already exists
        if output_path.exists() and (output_path / "config.json").exists():
            logger.info(f"Pruned model already exists: {output_path}")
            return output_path

        logger.info(f"Pruning {pruning_ratio:.1%} of weights using {strategy} strategy")

        # Create output directory and copy non-weight files
        output_path.mkdir(parents=True, exist_ok=True)
        self._copy_non_weight_files(model_path, output_path)

        # Process weight files
        weight_files = list(model_path.glob("*.safetensors"))
        if not weight_files:
            weight_files = list(model_path.glob("*.bin"))
            if not weight_files:
                raise PruningError("No weight files found")

        # Apply pruning
        if weight_files[0].suffix == ".safetensors":
            self._prune_safetensors(weight_files, output_path, pruning_ratio, strategy)
        else:
            self._prune_bin_files(weight_files, output_path, pruning_ratio, strategy)

        logger.info(f"Pruned model saved to {output_path}")
        return output_path

    def _copy_non_weight_files(self, src: Path, dst: Path):
        """Copy config and other non-weight files."""
        _copy_non_weight_files(src, dst)

    def _prune_safetensors(
        self,
        weight_files: list[Path],
        output_path: Path,
        pruning_ratio: float,
        strategy: str,
    ):
        """Apply pruning to safetensors files."""
        all_tensors = {}

        for wf in weight_files:
            with safe_open(wf, framework="pt", device="cpu") as f:
                for key in f:
                    tensor = f.get_tensor(key)

                    # Only prune weight matrices, not biases or norms
                    if self._should_prune(key, tensor):
                        tensor = self._prune_tensor(tensor, pruning_ratio, strategy)

                    all_tensors[key] = tensor

        output_file = output_path / "model.safetensors"
        save_file(all_tensors, output_file)
        logger.info(f"Saved {len(all_tensors)} tensors to {output_file}")

    def _prune_bin_files(
        self,
        bin_files: list[Path],
        output_path: Path,
        pruning_ratio: float,
        strategy: str,
    ):
        """Apply pruning to .bin files."""
        all_tensors = {}

        for bf in bin_files:
            state_dict = torch.load(bf, map_location="cpu", weights_only=True)

            for key, tensor in state_dict.items():
                if self._should_prune(key, tensor):
                    tensor = self._prune_tensor(tensor, pruning_ratio, strategy)
                all_tensors[key] = tensor

        output_file = output_path / "model.safetensors"
        save_file(all_tensors, output_file)
        logger.info(f"Saved {len(all_tensors)} tensors to {output_file}")

    def _should_prune(self, key: str, tensor: torch.Tensor) -> bool:
        """Determine if a tensor should be pruned.

        We prune:
        - Linear projection weights (q, k, v, o, ffn)
        - MLP weights (gate, up, down)

        We don't prune:
        - Embeddings (critical for vocab)
        - LayerNorm/RMSNorm (small and important)
        - Biases (small)
        - 1D tensors
        """
        # Skip 1D tensors (biases, norms)
        if tensor.dim() < 2:
            return False

        # Skip embeddings
        if any(embed in key.lower() for embed in ["embed", "wte", "wpe", "lm_head"]):
            return False

        # Skip normalization layers
        if any(norm in key.lower() for norm in ["norm", "ln_", "layernorm"]):
            return False

        # Prune attention and MLP weights
        prune_patterns = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP (Llama)
            "c_attn",
            "c_proj",
            "c_fc",  # Attention (GPT-2)
            "mlp",
            "dense",  # Generic
        ]

        return any(pattern in key.lower() for pattern in prune_patterns)

    def _prune_tensor(
        self,
        tensor: torch.Tensor,
        pruning_ratio: float,
        strategy: str,
    ) -> torch.Tensor:
        """Apply pruning to a single tensor.

        Uses unstructured magnitude pruning: sets smallest weights to zero.
        """
        if strategy == "magnitude":
            return self._magnitude_prune(tensor, pruning_ratio)
        elif strategy == "random":
            return self._random_prune(tensor, pruning_ratio)
        else:
            raise PruningError(f"Unknown pruning strategy: {strategy}")

    def _magnitude_prune(
        self,
        tensor: torch.Tensor,
        pruning_ratio: float,
    ) -> torch.Tensor:
        """Prune by setting smallest magnitude weights to zero."""
        # Flatten to find threshold
        flat = tensor.abs().flatten()
        k = int(flat.numel() * pruning_ratio)

        if k == 0:
            return tensor

        # Find threshold value
        threshold = torch.kthvalue(flat, k).values

        # Create mask and apply
        mask = tensor.abs() >= threshold
        return tensor * mask.to(tensor.dtype)

    def _random_prune(
        self,
        tensor: torch.Tensor,
        pruning_ratio: float,
    ) -> torch.Tensor:
        """Randomly prune weights."""
        mask = torch.rand_like(tensor) > pruning_ratio
        return tensor * mask.to(tensor.dtype)


def drop_layers(
    model_path: Path,
    layers_to_drop: list[int],
    cache_dir: Path | None = None,
) -> Path:
    """Convenience function to drop layers from a model.

    Args:
        model_path: Path to HuggingFace model directory
        layers_to_drop: List of layer indices to remove
        cache_dir: Optional cache directory

    Returns:
        Path to the pruned model
    """
    dropper = LayerDropper(cache_dir=cache_dir)
    return dropper.drop_layers(model_path, layers_to_drop)


def prune_model(
    model_path: Path,
    pruning_ratio: float,
    cache_dir: Path | None = None,
    strategy: str = "magnitude",
) -> Path:
    """Convenience function to prune a model.

    Args:
        model_path: Path to HuggingFace model directory
        pruning_ratio: Fraction of weights to prune
        cache_dir: Optional cache directory
        strategy: Pruning strategy

    Returns:
        Path to the pruned model
    """
    pruner = StructuredPruner(cache_dir=cache_dir)
    return pruner.prune(model_path, pruning_ratio, strategy=strategy)


def apply_compression(
    model_path: Path,
    pruning_ratio: float = 0.0,
    layers_to_drop: list[int] | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Apply both pruning and layer dropping to a model.

    Operations are applied in order:
    1. Layer dropping (if specified)
    2. Structured pruning (if ratio > 0)

    Args:
        model_path: Path to HuggingFace model directory
        pruning_ratio: Fraction of weights to prune
        layers_to_drop: Layer indices to remove
        cache_dir: Optional cache directory

    Returns:
        Path to the compressed model
    """
    current_path = model_path

    # Step 1: Drop layers
    if layers_to_drop:
        dropper = LayerDropper(cache_dir=cache_dir)
        current_path = dropper.drop_layers(current_path, layers_to_drop)

    # Step 2: Apply structured pruning
    if pruning_ratio > 0.0:
        pruner = StructuredPruner(cache_dir=cache_dir)
        # Include layer drop info in name if applicable
        output_name = None
        if layers_to_drop:
            output_name = f"{model_path.name}-dropped{len(layers_to_drop)}-pruned{int(pruning_ratio * 100)}pct"
        current_path = pruner.prune(
            current_path, pruning_ratio, output_name=output_name
        )

    return current_path
