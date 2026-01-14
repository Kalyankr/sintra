"""Hardware auto-detection using psutil.

This module provides automatic hardware detection to generate
HardwareProfile objects without requiring manual YAML files.
"""

import platform
import subprocess
from typing import Optional

import psutil

from sintra.profiles.models import Constraints, HardwareProfile, Targets


def detect_cuda() -> bool:
    """Check if CUDA is available on the system."""
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


def get_gpu_vram_gb() -> Optional[float]:
    """Get GPU VRAM in GB if available."""
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
            total_mb = sum(int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip())
            return total_mb / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def detect_cpu_arch() -> str:
    """Detect CPU architecture."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    elif machine in ("x86_64", "amd64"):
        return "x86_64"
    elif machine.startswith("arm"):
        return "arm"
    return machine


def detect_system_name() -> str:
    """Generate a human-readable system name."""
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
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().strip().rstrip("\x00")
                if "Raspberry Pi" in model:
                    return f"{model} ({ram_gb}GB)"
        except (FileNotFoundError, PermissionError):
            pass
        
        # Check for Jetson
        try:
            with open("/etc/nv_tegra_release", "r") as f:
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


def auto_detect_hardware(
    target_tps: float = 30.0,
    target_accuracy: float = 0.65,
    max_latency_ms: Optional[float] = None,
) -> HardwareProfile:
    """Auto-detect hardware and create a HardwareProfile.
    
    Args:
        target_tps: Target tokens per second
        target_accuracy: Minimum accuracy score
        max_latency_ms: Maximum latency in milliseconds (auto-calculated if not provided)
        
    Returns:
        HardwareProfile with detected hardware specs
    """
    has_cuda = detect_cuda()
    cpu_arch = detect_cpu_arch()
    vram_gb = detect_available_memory_gb()
    system_name = detect_system_name()
    supported_quants = detect_supported_quantizations(has_cuda)
    
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


def print_hardware_info() -> None:
    """Print detected hardware information to console."""
    from sintra.ui.console import console
    
    profile = auto_detect_hardware()
    
    console.print("\n[bold cyan]🔍 Detected Hardware[/bold cyan]")
    console.print(f"  System: {profile.name.replace('Auto-detected: ', '')}")
    console.print(f"  CPU Architecture: {profile.constraints.cpu_arch}")
    console.print(f"  Available Memory: {profile.constraints.vram_gb:.1f} GB")
    console.print(f"  CUDA Available: {'Yes' if profile.constraints.has_cuda else 'No'}")
    console.print(f"  Supported Quantizations: {', '.join(profile.supported_quantizations or [])}")
    console.print()
