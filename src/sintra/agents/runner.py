import json
import os
import time
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SintraRunner:
    """
    The 'Physical Lab' worker. Performs surgery, measures speed,
    and tests reasoning in a single isolated process.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def apply_surgery(self, bits: int, layers_to_drop: list[int]) -> Any:
        """Loads model and applies quantization/pruning instructions."""
        # Load with Quantization (bitsandbytes integration)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            load_in_4bit=(bits == 4),
            load_in_8bit=(bits == 8),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )

        # Structured Pruning: Removing specified layers
        if (
            layers_to_drop
            and hasattr(model, "model")
            and hasattr(model.model, "layers")
        ):
            current_layers = model.model.layers
            model.model.layers = torch.nn.ModuleList(
                [
                    layer
                    for i, layer in enumerate(current_layers)
                    if i not in layers_to_drop
                ]
            )

        return model

    def run_performance_test(self, model) -> dict[str, float]:
        """Measures Tokens Per Second (TPS) and Memory Usage."""
        prompt = "The key to efficient machine learning is"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

        max_new_tokens = 30

        # Warm-up (Compiles kernels, stabilizes cache)
        _ = model.generate(**inputs, max_new_tokens=5)

        # Benchmark run
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        end = time.perf_counter()

        # Calculate metrics
        gen_tokens = len(output[0]) - len(inputs[0])
        tps = gen_tokens / (end - start)

        # Memory check
        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
        else:
            mem_gb = psutil.Process().memory_info().rss / 1e9

        return {"tps": round(tps, 2), "vram_gb": round(mem_gb, 2)}

    def run_reasoning_test(self, model) -> float:
        """Briefly evaluates if the model is still 'intelligent'."""
        probes = [
            {"q": "Calculate 15 + 27:", "a": "42"},
            {"q": "What is the capital of France?", "a": "Paris"},
            {
                "q": "If A is bigger than B, and B is bigger than C, is A bigger than C? (Yes/No)",
                "a": "Yes",
            },
        ]

        score = 0
        for probe in probes:
            inputs = self.tokenizer(probe["q"], return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=15)
            answer = self.tokenizer.decode(out[0], skip_special_tokens=True)

            if probe["a"].lower() in answer.lower():
                score += 1

        return round(score / len(probes), 2)


if __name__ == "__main__":
    # Get instructions from Environment Variables
    try:
        cfg = {
            "model": os.getenv("MODEL_ID", "unsloth/Llama-3.2-1B"),
            "bits": int(os.getenv("BITS", "4")),
            "drop": json.loads(os.getenv("DROP", "[]")),
        }

        # Initialize and run surgery
        runner = SintraRunner(cfg["model"])
        model = runner.apply_surgery(cfg["bits"], cfg["drop"])

        # Get speed and intelligence data
        perf = runner.run_performance_test(model)
        accuracy = runner.run_reasoning_test(model)

        # Return JSON to the Agent via stdout
        result = {
            "status": "success",
            "tps": perf["tps"],
            "vram_gb": perf["vram_gb"],
            "accuracy": accuracy,
        }
        print(json.dumps(result))

    except Exception as e:
        # Crucial for Agentic Loop: Fail gracefully (Architect knows what went wrong)
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(e),
                    "tps": 0.0,
                    "vram_gb": 0.0,
                    "accuracy": 0.0,
                }
            )
        )
