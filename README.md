# Sintra (Synthetic Intelligence for Targeted Runtime Architectures): The Edge AI Architect

**Sintra** is an autonomous agentic framework designed to bridge the gap between massive Large Language Models (LLMs) and resource constrained edge hardware (Raspberry Pi, NVIDIA Jetson, Mobile).

## The Problem
Running a 70B parameter model on a 8GB RAM device is physically impossible. Manual pruning and quantization (compression) often result in "lobotomized" models that lose reasoning capabilities.

## The Solution
An autonomous **Agentic Loop** that:
1.  **Analyzes** a hardware profile.
2.  **Hypothesizes** a compression "recipe" (Pruning + Quantization + NAS).
3.  **Executes** a benchmark in a sandboxed Docker container.
4.  **Evaluates** the "Student" model against a "Teacher" using RL-guided feedback.

## Key Innovations
* **Agentic Loop:** Uses LangGraph to iterate until hardware targets (Latency/Accuracy) are met.
* **Hardware-in-the-Loop:** Real-world benchmarking using Docker SDK to simulate edge constraints.
* **RL Controller:** Optimizes the compression strategy using a multi-objective reward function.

# Tech Stack & Tooling

| Category | Technology |
| :--- | :--- |
| **Orchestration** | **LangGraph** (Stateful, cyclic agent workflows) |
| **LLM Provider** | **OpenAI / Anthropic** (For the Architect's reasoning) |
| **ML Framework** | **PyTorch** & **Hugging Face Transformers** |
| **Compression** | **AutoGPTQ**, **BitsAndBytes**, **llama.cpp** |
| **Isolation** | **Docker** (For hardware-specific benchmarking) |
| **Database** | **SQLite / JSON** (For tracking experiment history) |
| **Env Management** | `uv` (Fast, reproducible Python environments) |


                                                Work in Progress
                                            Driven By Human Curiosity