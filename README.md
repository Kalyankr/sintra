# Sintra (Synthetic Intelligence for Targeted Runtime Architectures): The Edge AI Architect

**Sintra** is an autonomous agentic framework designed to bridge the gap between massive Large Language Models (LLMs) and resource constrained edge hardware (Raspberry Pi, NVIDIA Jetson, Mobile).

## The Problem
Running a 70B parameter model on a 8GB RAM device is physically impossible. Manual pruning and quantization (compression) often result in "lobotomized" models that lose reasoning capabilities.

## The Solution
An autonomous **Agentic Loop** that:
1.  **Analyzes** a hardware profile.
2.  **Hypothesizes** a compression "recipe" (Pruning + Quantization + Layer Dropping).
3.  **Executes** a benchmark with real compression.
4.  **Evaluates** the compressed model and iterates until targets are met.

## Key Features

### 🎯 Multiple Compression Backends
- **GGUF** (llama.cpp) - Best for CPU inference
- **BitsAndBytes** - GPU-optimized NF4/INT8 quantization  
- **ONNX** - Cross-platform optimization

### 🔧 Hardware Auto-Detection
```bash
# Auto-detect your hardware and set smart targets
sintra --auto-detect --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 💾 Experiment History
- SQLite database tracks all experiments across runs
- Agent learns from past failures to avoid repeating mistakes
- Find best recipes for your specific hardware

### ⏸️ Checkpointing & Resume
```bash
# List available checkpoints
sintra --list-checkpoints --auto-detect

# Resume from the latest checkpoint
sintra --resume --auto-detect

# Resume a specific run
sintra --resume abc12345 --auto-detect
```

### 🏃 Dry Run Mode
```bash
# Preview what would happen without running compression
sintra --auto-detect --dry-run
```

## Quick Start

### Installation
```bash
# Clone the repo
git clone https://github.com/yourusername/sintra.git
cd sintra

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Basic Usage
```bash
# Using auto-detect (recommended for most users)
sintra --auto-detect --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Using a hardware profile
sintra profiles/raspberry_pi_5.yaml --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Dry run to see configuration
sintra --auto-detect --dry-run

# Use a different backend
sintra --auto-detect --backend bnb  # bitsandbytes for GPU
sintra --auto-detect --backend onnx # ONNX optimization
```

### CLI Options
```
Usage: sintra [OPTIONS] [PROFILE]

Target Model:
  --model-id          HuggingFace model to optimize (default: TinyLlama)
  --backend           Quantization backend: gguf, bnb, onnx (default: gguf)
  --output-dir        Directory for optimized models (default: ./outputs)

Hardware Configuration:
  --auto-detect       Auto-detect hardware specs
  --target-tps        Target tokens per second (default: auto-calculated)
  --target-accuracy   Minimum accuracy score (default: auto-calculated)

Execution Settings:
  --mock              Use mock executor (for testing)
  --max-iters         Maximum optimization attempts (default: 10)
  --dry-run           Show what would be done without executing
  --resume [RUN_ID]   Resume from checkpoint (latest or specific run)
  --list-checkpoints  List all available checkpoints

Architect Brain:
  --provider          LLM provider: openai, anthropic, ollama
  --model             Specific model for the architect
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                        │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│  Architect  │ Benchmarker │   Critic    │    Reporter      │
│  (LLM)      │  (Real HW)  │ (Evaluator) │   (Output)       │
└─────────────┴─────────────┴─────────────┴──────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Compression Backends                        │
├─────────────────┬─────────────────┬────────────────────────┤
│  GGUF/llama.cpp │  BitsAndBytes   │   ONNX/Optimum         │
└─────────────────┴─────────────────┴────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Persistence Layer                         │
├─────────────────────────────────────────────────────────────┤
│  SQLite History DB  │  JSON Checkpoints  │  YAML Profiles   │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Category | Technology |
| :--- | :--- |
| **Orchestration** | **LangGraph** (Stateful, cyclic agent workflows) |
| **LLM Provider** | **OpenAI / Anthropic / Ollama** (Architect reasoning) |
| **ML Framework** | **PyTorch** & **Hugging Face Transformers** |
| **Compression** | **llama.cpp**, **BitsAndBytes**, **ONNX/Optimum** |
| **Database** | **SQLite** (Experiment history) |
| **Checkpoints** | **JSON** (Resumable runs) |
| **Env Management** | `uv` (Fast, reproducible Python environments) |

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run with debug mode (no LLM calls)
sintra --auto-detect --debug

# Run with mock executor (fast iteration)
sintra --auto-detect --mock
```

## Project Status

✅ **Implemented**
- Multi-backend compression (GGUF, BitsAndBytes, ONNX)
- Hardware auto-detection with smart targets
- SQLite experiment history
- Checkpointing and resume
- Progress callbacks
- Structured pruning and layer dropping

🚧 **In Progress**
- End-to-end integration tests
- Ollama model export

📋 **Planned**
- Web UI dashboard
- Distributed benchmarking
- Model quality evaluation improvements

---

<p align="center">
  <em>Work in Progress</em><br>
  <strong>Driven By Human Curiosity</strong>
</p>