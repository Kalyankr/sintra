"""Hardware auto-detection using psutil.

This module provides automatic hardware detection to generate
HardwareProfile objects without requiring manual YAML files.
"""

import platform
import subprocess
from functools import lru_cache
from pathlib import Path

import psutil

from sintra.profiles.models import Constraints, HardwareProfile, Targets


@lru_cache(maxsize=1)
def detect_cuda() -> bool:
    """Check if CUDA is available on the system.

    Result is cached since CUDA availability doesn't change during process lifetime.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@lru_cache(maxsize=1)
def get_gpu_vram_gb() -> float | None:
    """Get GPU VRAM in GB if available.

    Result is cached since GPU VRAM doesn't change during process lifetime.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Sum all GPUs' VRAM
            total_mb = sum(
                int(line.strip())
                for line in result.stdout.strip().split("\n")
                if line.strip()
            )
            return total_mb / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


@lru_cache(maxsize=1)
def detect_cpu_arch() -> str:
    """Detect CPU architecture. Cached per process."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    elif machine in ("x86_64", "amd64"):
        return "x86_64"
    elif machine.startswith("arm"):
        return "arm"
    return machine


@lru_cache(maxsize=1)
def detect_system_name() -> str:
    """Generate a human-readable system name. Cached per process."""
    system = platform.system()
    machine = platform.machine()

    # Get CPU info
    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
    ram_gb = round(psutil.virtual_memory().total / (1024**3))

    # Detect specific platforms
    if system == "Darwin":
        # macOS - try to get chip name
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                cpu_name = result.stdout.strip()
                if "Apple" in cpu_name:
                    return f"Mac ({cpu_name}, {ram_gb}GB)"
                return f"Mac ({cpu_count} cores, {ram_gb}GB)"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return f"Mac ({machine}, {ram_gb}GB)"

    elif system == "Linux":
        # Check for Raspberry Pi
        try:
            with open("/proc/device-tree/model") as f:
                model = f.read().strip().rstrip("\x00")
                if "Raspberry Pi" in model:
                    return f"{model} ({ram_gb}GB)"
        except (FileNotFoundError, PermissionError):
            pass

        # Check for Jetson
        try:
            with open("/etc/nv_tegra_release") as f:
                return f"NVIDIA Jetson ({ram_gb}GB)"
        except (FileNotFoundError, PermissionError):
            pass

        return f"Linux ({cpu_count} cores, {ram_gb}GB)"

    elif system == "Windows":
        return f"Windows ({cpu_count} cores, {ram_gb}GB)"

    return f"{system} ({cpu_count} cores, {ram_gb}GB)"


def detect_available_memory_gb() -> float:
    """Get available memory for AI workloads in GB.

    For CUDA systems, returns GPU VRAM.
    For other systems, returns 70% of system RAM (leaving room for OS).
    """
    # Check for GPU VRAM first
    gpu_vram = get_gpu_vram_gb()
    if gpu_vram:
        return gpu_vram

    # Fall back to system RAM with overhead
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    # Reserve ~30% for OS and other processes
    available_gb = total_ram_gb * 0.7
    return round(available_gb, 1)


def detect_supported_quantizations(has_cuda: bool) -> list[str]:
    """Determine supported quantization methods based on hardware."""
    methods = ["GGUF"]  # Always supported

    if has_cuda:
        methods.extend(["BNB", "ONNX"])
    else:
        methods.append("ONNX")  # ONNX works on CPU too

    return methods


def estimate_target_tps(vram_gb: float, has_cuda: bool, cpu_arch: str) -> float:
    """Estimate a reasonable TPS target based on hardware capabilities.

    These are conservative estimates based on typical performance:
    - High-end GPU (24GB+): 80-150 TPS
    - Mid-range GPU (8-16GB): 40-80 TPS
    - Apple Silicon (M1/M2/M3/M4): 30-80 TPS depending on RAM
    - CPU-only x86: 5-20 TPS
    - ARM (Raspberry Pi): 5-15 TPS
    """
    if has_cuda:
        # NVIDIA GPU - scale with VRAM
        if vram_gb >= 24:
            return 100.0
        elif vram_gb >= 16:
            return 80.0
        elif vram_gb >= 8:
            return 50.0
        else:
            return 30.0

    elif cpu_arch == "arm64":
        # Apple Silicon or ARM - scale with RAM
        if vram_gb >= 24:
            return 80.0  # M2/M3 Max/Ultra
        elif vram_gb >= 14:
            return 60.0  # M4 Pro / M3 Pro
        elif vram_gb >= 8:
            return 40.0  # M1/M2/M3/M4 base
        else:
            return 15.0  # Raspberry Pi / Jetson

    else:
        # x86 CPU-only
        if vram_gb >= 32:
            return 20.0
        elif vram_gb >= 16:
            return 15.0
        else:
            return 10.0


def estimate_target_accuracy(vram_gb: float) -> float:
    """Estimate a reasonable accuracy target based on available memory.

    More memory = can run larger/less compressed models = higher accuracy possible.
    These are minimum acceptable accuracy scores (0-1 scale).
    """
    if vram_gb >= 24:
        return 0.75  # Can run larger models with light compression
    elif vram_gb >= 16:
        return 0.70
    elif vram_gb >= 8:
        return 0.65
    elif vram_gb >= 4:
        return 0.60
    else:
        return 0.55  # Aggressive compression needed


def auto_detect_hardware(
    target_tps: float | None = None,
    target_accuracy: float | None = None,
    max_latency_ms: float | None = None,
) -> HardwareProfile:
    """Auto-detect hardware and create a HardwareProfile.

    Args:
        target_tps: Target tokens per second (auto-calculated if None)
        target_accuracy: Minimum accuracy score (auto-calculated if None)
        max_latency_ms: Maximum latency in milliseconds (auto-calculated if None)

    Returns:
        HardwareProfile with detected hardware specs and smart defaults
    """
    has_cuda = detect_cuda()
    cpu_arch = detect_cpu_arch()
    vram_gb = detect_available_memory_gb()
    system_name = detect_system_name()
    supported_quants = detect_supported_quantizations(has_cuda)

    # Auto-calculate targets if not provided
    if target_tps is None:
        target_tps = estimate_target_tps(vram_gb, has_cuda, cpu_arch)

    if target_accuracy is None:
        target_accuracy = estimate_target_accuracy(vram_gb)

    # Auto-calculate max latency if not provided
    if max_latency_ms is None:
        # Lower latency target for more powerful hardware
        if vram_gb >= 16:
            max_latency_ms = 200.0
        elif vram_gb >= 8:
            max_latency_ms = 500.0
        else:
            max_latency_ms = 1000.0

    return HardwareProfile(
        name=f"Auto-detected: {system_name}",
        constraints=Constraints(
            vram_gb=vram_gb,
            cpu_arch=cpu_arch,
            has_cuda=has_cuda,
        ),
        targets=Targets(
            min_tokens_per_second=target_tps,
            min_accuracy_score=target_accuracy,
            max_latency_ms=max_latency_ms,
        ),
        supported_quantizations=supported_quants,
    )


def save_profile_to_yaml(profile: HardwareProfile, path: Path) -> Path:
    """Save a HardwareProfile to a YAML file for review/editing.

    Args:
        profile: The hardware profile to save
        path: Path to save the YAML file

    Returns:
        The path where the file was saved
    """
    import yaml

    # Convert profile to dict
    profile_dict = {
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
    }

    if profile.supported_quantizations:
        profile_dict["supported_quantizations"] = profile.supported_quantizations

    # Add helpful comments as a header
    header = """# Auto-detected hardware profile
# Generated by: sintra --auto-detect
#
# You can edit these values and use this file with:
#   sintra detected_profile.yaml
#
# Target estimates are based on your hardware capabilities:
#   - Higher VRAM/RAM = higher TPS targets
#   - CUDA GPU = significantly higher TPS
#   - ARM CPU = lower TPS but still usable
#
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header)
        yaml.dump(profile_dict, f, default_flow_style=False, sort_keys=False)

    return path


def print_hardware_info(save_path: Path | None = None) -> Path | None:
    """Print detected hardware information to console.

    Args:
        save_path: Optional path to save the profile as YAML

    Returns:
        Path where profile was saved, or None if not saved
    """
    from sintra.ui.console import console

    profile = auto_detect_hardware()

    console.print("\n[bold cyan]🔍 Detected Hardware[/bold cyan]")
    console.print(f"  System: {profile.name.replace('Auto-detected: ', '')}")
    console.print(f"  CPU Architecture: {profile.constraints.cpu_arch}")
    console.print(f"  Available Memory: {profile.constraints.vram_gb:.1f} GB")
    console.print(
        f"  CUDA Available: {'Yes' if profile.constraints.has_cuda else 'No'}"
    )
    console.print(
        f"  Supported Quantizations: {', '.join(profile.supported_quantizations or [])}"
    )

    console.print("\n[bold cyan]📊 Auto-calculated Targets[/bold cyan]")
    console.print(
        f"  Target TPS: {profile.targets.min_tokens_per_second:.0f} tokens/sec"
    )
    console.print(f"  Min Accuracy: {profile.targets.min_accuracy_score:.0%}")
    console.print(f"  Max Latency: {profile.targets.max_latency_ms:.0f} ms")
    console.print()

    saved_path = None
    if save_path:
        saved_path = save_profile_to_yaml(profile, save_path)
        console.print(f"[green]✓ Profile saved to: {saved_path}[/green]")
        console.print(f"[dim]  Edit this file and run: sintra {saved_path}[/dim]")
        console.print()

    return saved_path
