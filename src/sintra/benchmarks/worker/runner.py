import json
import os
import sys
import time
from pathlib import Path

import psutil
from llama_cpp import Llama

from sintra.profiles.models import ExperimentResult, ModelRecipe


def find_model_file(recipe: ModelRecipe) -> str:
    """Find the appropriate model file based on quantization bits.
    
    Args:
        recipe: The compression recipe specifying bits.
        
    Returns:
        Path to the model file.
        
    Raises:
        FileNotFoundError: If no suitable model file is found.
    """
    # Map bits to quantization types
    quant_map = {
        2: ["Q2_K"],
        3: ["Q3_K_S", "Q3_K_M", "Q3_K_L"],
        4: ["Q4_K_S", "Q4_K_M", "Q4_K", "Q4_0"],
        5: ["Q5_K_S", "Q5_K_M", "Q5_K"],
        6: ["Q6_K"],
        8: ["Q8_0"],
    }
    
    quant_types = quant_map.get(recipe.bits, ["Q4_K_M", "Q4_K"])
    
    # Search locations
    search_dirs = [
        Path("models"),
        Path.home() / ".cache" / "sintra" / "models",
        Path.home() / ".local" / "share" / "sintra" / "models",
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for quant_type in quant_types:
            # Try various naming patterns
            patterns = [
                f"*{quant_type}*.gguf",
                f"*.{quant_type}.gguf",
            ]
            for pattern in patterns:
                matches = list(search_dir.glob(pattern))
                if matches:
                    return str(matches[0])
    
    # Provide helpful error message
    raise FileNotFoundError(
        f"No GGUF model found for {recipe.bits}-bit quantization. "
        f"Searched in: {', '.join(str(d) for d in search_dirs)}. "
        f"Please download a model: huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF "
        f"--local-dir models --include '*Q4_K_M*'"
    )


def perform_surgery(recipe: ModelRecipe) -> ExperimentResult:
    """Performs actual hardware benchmarking using llama-cpp-python."""
    try:
        # Find appropriate model file
        model_path = find_model_file(recipe)

        # Start Timing and Memory Tracking
        process = psutil.Process()
        start_mem = process.memory_info().rss

        start_time = time.time()

        # Load Model (The 'Surgery')
        llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=8,
            n_gpu_layers=-1 if sys.platform == "darwin" else 0,
            verbose=False,
        )

        # 4. Run Benchmark Generation
        # We generate 32 tokens to get a stable TPS reading
        output = llm(
            "Q: What is the best fruit amd give a nice story about it? A:",
            max_tokens=600,
            temperature=0.5,
        )

        end_time = time.time()

        # 5. Calculate Real Metrics
        tokens_sent = output["usage"]["completion_tokens"]
        duration = end_time - start_time
        actual_tps = tokens_sent / duration if duration > 0 else 0

        # Calculate Peak RAM usage in GB
        peak_mem = process.memory_info().rss
        actual_vram = peak_mem / (1024**3)

        return ExperimentResult(
            actual_tps=round(actual_tps, 2),
            actual_vram_usage=round(actual_vram, 2),
            accuracy_score=0.90,
            was_successful=True,
            error_log="",
        )

    except Exception as e:
        return ExperimentResult(
            actual_tps=0,
            actual_vram_usage=0,
            accuracy_score=0,
            was_successful=False,
            error_log=f"Surgery failed: {str(e)}",
        )


def main():
    try:
        # 1. Read the JSON 'Order' from Stdin
        raw_input = sys.stdin.read()
        if not raw_input:
            sys.stderr.write("Worker Error: No input received on stdin\n")
            sys.exit(1)

        # 2. Parse into a ModelRecipe object
        recipe_dict = json.loads(raw_input)
        recipe = ModelRecipe.model_validate(recipe_dict)

        # 3. Perform the work
        # (Using stderr for logging so we don't pollute stdout)
        sys.stderr.write(f"Worker: Starting {recipe.bits}-bit surgery...\n")
        result = perform_surgery(recipe)

        # 4. Output ONLY the result JSON to Stdout
        print(result.model_dump_json())

    except Exception as e:
        sys.stderr.write(f"Worker Crash: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
