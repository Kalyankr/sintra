"""GGUF model quantization using llama.cpp.

Provides quantization of GGUF models to various bit depths.
"""

import logging
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sintra"


class QuantizationType(str, Enum):
    """Supported GGUF quantization types.
    
    Ordered from most aggressive compression to least.
    """
    Q2_K = "Q2_K"       # 2-bit, ~2.5 bpw, aggressive
    Q3_K_S = "Q3_K_S"   # 3-bit small
    Q3_K_M = "Q3_K_M"   # 3-bit medium
    Q3_K_L = "Q3_K_L"   # 3-bit large
    Q4_0 = "Q4_0"       # 4-bit, legacy
    Q4_K_S = "Q4_K_S"   # 4-bit small
    Q4_K_M = "Q4_K_M"   # 4-bit medium (recommended)
    Q5_0 = "Q5_0"       # 5-bit, legacy
    Q5_K_S = "Q5_K_S"   # 5-bit small
    Q5_K_M = "Q5_K_M"   # 5-bit medium
    Q6_K = "Q6_K"       # 6-bit
    Q8_0 = "Q8_0"       # 8-bit
    F16 = "F16"         # 16-bit float (no quantization)
    F32 = "F32"         # 32-bit float


# Map bits to recommended quantization type
BITS_TO_QUANT: dict[int, QuantizationType] = {
    2: QuantizationType.Q2_K,
    3: QuantizationType.Q3_K_M,
    4: QuantizationType.Q4_K_M,
    5: QuantizationType.Q5_K_M,
    6: QuantizationType.Q6_K,
    8: QuantizationType.Q8_0,
}


class QuantizationError(Exception):
    """Raised when quantization fails."""
    pass


class GGUFQuantizer:
    """Quantizes GGUF models using llama.cpp's llama-quantize.
    
    Handles conversion from HuggingFace format to GGUF and quantization
    to various bit depths.
    
    Example:
        >>> quantizer = GGUFQuantizer()
        >>> output = quantizer.quantize(
        ...     model_path="/path/to/model",
        ...     bits=4,
        ...     model_name="tinyllama"
        ... )
        >>> print(output)
        /home/user/.cache/sintra/quantized/tinyllama-q4_k_m.gguf
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the quantizer.
        
        Args:
            cache_dir: Base cache directory. Defaults to ~/.cache/sintra/
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.gguf_dir = self.cache_dir / "gguf"
        self.quantized_dir = self.cache_dir / "quantized"
        
        self.gguf_dir.mkdir(parents=True, exist_ok=True)
        self.quantized_dir.mkdir(parents=True, exist_ok=True)
        
        # Locate llama.cpp tools
        self._llama_quantize = self._find_llama_quantize()
        self._convert_script = self._find_convert_script()
    
    def _find_llama_quantize(self) -> Optional[Path]:
        """Find llama-quantize binary."""
        # Check common locations
        candidates = [
            shutil.which("llama-quantize"),
            shutil.which("quantize"),
            Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
            Path.home() / "llama.cpp" / "llama-quantize",
            Path("/usr/local/bin/llama-quantize"),
        ]
        
        for candidate in candidates:
            if candidate:
                path = Path(candidate) if isinstance(candidate, str) else candidate
                if path.exists():
                    logger.info(f"Found llama-quantize at {path}")
                    return path
        
        return None
    
    def _find_convert_script(self) -> Optional[Path]:
        """Find convert_hf_to_gguf.py script."""
        candidates = [
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
            Path("/usr/local/share/llama.cpp/convert_hf_to_gguf.py"),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                logger.info(f"Found convert script at {candidate}")
                return candidate
        
        return None
    
    def convert_to_gguf(
        self,
        model_path: Path,
        output_name: str,
        output_type: str = "f16",
    ) -> Path:
        """Convert HuggingFace model to GGUF format.
        
        Args:
            model_path: Path to HuggingFace model directory
            output_name: Base name for output file
            output_type: Output type (f16, f32, q8_0)
            
        Returns:
            Path to converted GGUF file
            
        Raises:
            QuantizationError: If conversion fails
        """
        output_file = self.gguf_dir / f"{output_name}-{output_type}.gguf"
        
        # Check if already converted
        if output_file.exists():
            logger.info(f"GGUF already exists: {output_file}")
            return output_file
        
        if not self._convert_script:
            raise QuantizationError(
                "convert_hf_to_gguf.py not found. "
                "Please install llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
            )
        
        logger.info(f"Converting {model_path} to GGUF...")
        
        try:
            result = subprocess.run(
                [
                    "python3",
                    str(self._convert_script),
                    str(model_path),
                    "--outfile", str(output_file),
                    "--outtype", output_type,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            if result.returncode != 0:
                raise QuantizationError(
                    f"GGUF conversion failed:\n{result.stderr}"
                )
            
            logger.info(f"Converted to {output_file}")
            return output_file
            
        except subprocess.TimeoutExpired:
            raise QuantizationError("GGUF conversion timed out (10 min limit)")
        except FileNotFoundError:
            raise QuantizationError("python3 not found")
    
    def quantize(
        self,
        model_path: Path,
        bits: int,
        model_name: Optional[str] = None,
        quant_type: Optional[QuantizationType] = None,
    ) -> Path:
        """Quantize a model to the specified bit depth.
        
        If given a HuggingFace model directory, converts to GGUF first.
        If given a GGUF file, quantizes directly.
        
        Args:
            model_path: Path to HF model dir or GGUF file
            bits: Target bit depth (2-8)
            model_name: Optional name for output file
            quant_type: Specific quantization type (overrides bits)
            
        Returns:
            Path to quantized GGUF file
            
        Raises:
            QuantizationError: If quantization fails
        """
        # Determine quantization type
        if quant_type is None:
            if bits not in BITS_TO_QUANT:
                raise QuantizationError(
                    f"Unsupported bit depth: {bits}. "
                    f"Supported: {list(BITS_TO_QUANT.keys())}"
                )
            quant_type = BITS_TO_QUANT[bits]
        
        # Determine model name
        if model_name is None:
            model_name = model_path.name.lower().replace("--", "-")
        
        # Check if already quantized
        output_file = self.quantized_dir / f"{model_name}-{quant_type.value.lower()}.gguf"
        if output_file.exists():
            logger.info(f"Quantized model already exists: {output_file}")
            return output_file
        
        # Determine input file
        if model_path.is_dir():
            # HuggingFace model - need to convert first
            gguf_path = self.convert_to_gguf(model_path, model_name, "f16")
        elif model_path.suffix == ".gguf":
            gguf_path = model_path
        else:
            raise QuantizationError(
                f"Unsupported model format: {model_path}. "
                "Expected HuggingFace directory or .gguf file"
            )
        
        # Run quantization
        return self._run_quantize(gguf_path, output_file, quant_type)
    
    def _run_quantize(
        self,
        input_file: Path,
        output_file: Path,
        quant_type: QuantizationType,
    ) -> Path:
        """Run llama-quantize on a GGUF file."""
        if not self._llama_quantize:
            raise QuantizationError(
                "llama-quantize not found. Install llama.cpp:\n"
                "  git clone https://github.com/ggerganov/llama.cpp\n"
                "  cd llama.cpp && make llama-quantize"
            )
        
        logger.info(f"Quantizing to {quant_type.value}...")
        
        try:
            result = subprocess.run(
                [
                    str(self._llama_quantize),
                    str(input_file),
                    str(output_file),
                    quant_type.value,
                ],
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min for large models
            )
            
            if result.returncode != 0:
                raise QuantizationError(
                    f"Quantization failed:\n{result.stderr}"
                )
            
            logger.info(f"Quantized to {output_file}")
            return output_file
            
        except subprocess.TimeoutExpired:
            raise QuantizationError("Quantization timed out (30 min limit)")
    
    def get_cached_quantizations(self, model_name: str) -> list[Path]:
        """List cached quantizations for a model."""
        pattern = f"{model_name.lower()}*.gguf"
        return list(self.quantized_dir.glob(pattern))


def quantize_model(
    model_path: Path,
    bits: int,
    cache_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
) -> Path:
    """Convenience function to quantize a model.
    
    Args:
        model_path: Path to model (HF dir or GGUF)
        bits: Target bit depth
        cache_dir: Optional cache directory
        model_name: Optional output name
        
    Returns:
        Path to quantized GGUF
    """
    quantizer = GGUFQuantizer(cache_dir=cache_dir)
    return quantizer.quantize(model_path, bits, model_name)
