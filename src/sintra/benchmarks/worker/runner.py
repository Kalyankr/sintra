import json
import sys
import time

from sintra.profiles.models import ExperimentResult, ModelRecipe


def perform_surgery(recipe: ModelRecipe) -> ExperimentResult:
    """
    This is where you would call llama.cpp or your quantization toolkit.
    """
    # Simulate processing time
    time.sleep(2)

    # MOCK LOGIC: Calculate simulated performance
    # Fewer bits = Higher Speed (TPS), Lower Accuracy
    base_tps = 10.0 if recipe.bits <= 4 else 5.0
    prune_boost = 1.0 + (recipe.pruning_ratio * 2)  # Pruning adds speed

    actual_tps = base_tps * prune_boost
    actual_vram = (recipe.bits * 0.8) * (1.0 - recipe.pruning_ratio)

    # Simulate accuracy (dropping layers reduces accuracy)
    accuracy = 0.95 - (recipe.pruning_ratio * 0.3)
    if len(recipe.layers_to_drop) > 5:
        accuracy -= 0.1

    return ExperimentResult(
        actual_tps=round(actual_tps, 2),
        actual_vram_usage=round(actual_vram, 2),
        accuracy_score=round(accuracy, 2),
        was_successful=True,
        error_log="",
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
