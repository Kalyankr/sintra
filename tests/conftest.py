"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _clear_hardware_caches():
    """Clear lru_cache on hardware detection functions between tests.
    
    These functions are cached for runtime efficiency but tests need
    to mock them independently.
    """
    from sintra.profiles.hardware import (
        detect_cpu_arch,
        detect_cuda,
        detect_system_name,
        get_gpu_vram_gb,
    )

    detect_cuda.cache_clear()
    get_gpu_vram_gb.cache_clear()
    detect_cpu_arch.cache_clear()
    detect_system_name.cache_clear()
    yield
    detect_cuda.cache_clear()
    get_gpu_vram_gb.cache_clear()
    detect_cpu_arch.cache_clear()
    detect_system_name.cache_clear()


@pytest.fixture(autouse=True)
def _clear_quantizer_caches():
    """Clear lru_cache on quantizer tool-path functions between tests."""
    from sintra.compression.quantizer import (
        _get_convert_script_paths,
        check_llama_cpp_available,
    )

    check_llama_cpp_available.cache_clear()
    _get_convert_script_paths.cache_clear()
    yield
    check_llama_cpp_available.cache_clear()
    _get_convert_script_paths.cache_clear()
