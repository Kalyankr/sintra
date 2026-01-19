<div align="center">

# 🧠 Sintra

### Autonomous AI Agent for Edge Model Optimization

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-purple.svg)](https://github.com/langchain-ai/langgraph)

**Sintra** (Synthetic Intelligence for Targeted Runtime Architectures) is a fully autonomous agentic framework that optimizes Large Language Models for resource-constrained edge devices.

[Quick Start](#-quick-start) • [Features](#-key-features) • [Architecture](#-agentic-architecture) • [Documentation](#-documentation)

</div>

---

## 🎯 The Problem

Running a 70B parameter model on an 8GB RAM device is physically impossible. Manual pruning and quantization often result in "lobotomized" models that lose reasoning capabilities.

## 💡 The Solution

An **autonomous AI agent** that:

1. **Plans** an optimization strategy based on your hardware constraints
2. **Researches** model architecture and similar successful optimizations
3. **Experiments** with compression recipes (quantization + pruning + layer dropping)
4. **Reflects** on failures and adjusts strategy
5. **Iterates** until performance targets are met

## 🚀 Quick Start

```bash
# Install
git clone https://github.com/Kalyankr/sintra.git
cd sintra
pip install -e .

# Run - zero flags needed!
sintra
```

That's it. Sintra auto-detects your hardware, sets smart targets, and starts optimizing.

### Common Examples

```bash
# Optimize a specific model
sintra --model-id microsoft/phi-2

# Use GPU-accelerated quantization
sintra --backend bnb --model-id meta-llama/Llama-3.2-1B

# Preview without running
sintra --dry-run

# Resume an interrupted run
sintra --resume
```

## ✨ Key Features

### 🤖 Fully Agentic
- **Tool Calling**: 5 specialized tools for model research
- **ReAct Pattern**: Reason → Act → Observe loop
- **Self-Reflection**: Learns from failures automatically
- **LLM Routing**: Smart decisions on when to stop
- **Planning**: Strategic optimization before execution

### 🎯 Multi-Backend Compression
| Backend | Best For | Quantization |
|---------|----------|--------------|
| **GGUF** (default) | CPU inference | 2-8 bit |
| **BitsAndBytes** | GPU inference | NF4, INT8 |
| **ONNX** | Cross-platform | INT8 |

### 📊 Baseline Accuracy Comparison
Automatically compares optimized model against the original to measure accuracy retention:

```
Accuracy Comparison:
  Original:  85.0%
  Optimized: 81.2%
  Retention: 95.5%
```

### 💾 Persistence & Learning
- **SQLite database** tracks all experiments
- **Cross-run learning**: Agent avoids past mistakes
- **Checkpointing**: Resume interrupted optimizations

### 🔧 Hardware Auto-Detection
Automatically detects CPU, RAM, GPU and calculates achievable targets:

```
🔍 Detected Hardware
  System: Linux (8 cores, 32GB)
  CUDA Available: Yes (RTX 4090)

📊 Auto-calculated Targets
  Target TPS: 45 tokens/sec
  Min Accuracy: 70%
```

## 🏗️ Agentic Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        SINTRA AGENT                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌─────────────────────────────────────────┐    │
│  │ PLANNER  │───▶│           REACT ARCHITECT               │    │
│  │  (LLM)   │    │  ┌────────────────────────────────────┐ │    │
│  └──────────┘    │  │ TOOLS:                             │ │    │
│                  │  │ • get_model_architecture           │ │    │
│                  │  │ • search_similar_models            │ │    │
│                  │  │ • estimate_compression_impact      │ │    │
│                  │  │ • query_hardware_capabilities      │ │    │
│                  │  │ • lookup_quantization_benchmarks   │ │    │
│                  │  └────────────────────────────────────┘ │    │
│                  └───────────────────┬─────────────────────┘    │
│                                      │                          │
│                                      ▼                          │
│                             ┌──────────────┐                    │
│                             │ BENCHMARKER  │                    │
│                             │  (Executor)  │                    │
│                             └──────┬───────┘                    │
│                                    │                            │
│                                    ▼                            │
│  ┌───────────┐            ┌──────────────┐                     │
│  │ REFLECTOR │◀───────────│    CRITIC    │                     │
│  │   (LLM)   │            │ (LLM Router) │                     │
│  └─────┬─────┘            └──────┬───────┘                     │
│        │                         │                              │
│        │    ┌────────────────────┴────────────────┐            │
│        │    │                                     │            │
│        ▼    ▼                                     ▼            │
│   [Continue Loop]                            [REPORTER]        │
│                                               (Output)         │
└──────────────────────────────────────────────────────────────────┘
```

## 📖 CLI Reference

```bash
sintra [profile] [options]
```

### Core Options
| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | TinyLlama | HuggingFace model to optimize |
| `--backend` | gguf | Compression backend (gguf/bnb/onnx) |
| `--output-dir` | ./outputs | Output directory |

### Hardware
| Flag | Default | Description |
|------|---------|-------------|
| `--auto-detect` | ✅ ON | Auto-detect hardware (default) |
| `--no-auto-detect` | - | Use YAML profile instead |
| `--target-tps` | auto | Target tokens per second |
| `--target-accuracy` | auto | Minimum accuracy threshold |

### Evaluation
| Flag | Default | Description |
|------|---------|-------------|
| `--baseline` | ✅ ON | Compare against original model |
| `--no-baseline` | - | Skip baseline (faster) |
| `--skip-accuracy` | - | Skip accuracy evaluation |

### Agentic Features
| Flag | Default | Description |
|------|---------|-------------|
| `--simple` | - | Disable all agentic features |
| `--no-plan` | - | Disable planner |
| `--no-react` | - | Disable ReAct architect |
| `--no-reflect` | - | Disable self-reflection |
| `--no-llm-routing` | - | Use rule-based routing |

### Execution
| Flag | Description |
|------|-------------|
| `--dry-run` | Preview without execution |
| `--resume [ID]` | Resume from checkpoint |
| `--list-checkpoints` | Show available checkpoints |
| `--mock` | Use mock executor (testing) |
| `--debug` | Single loop without LLM |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Agent Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **LLM Integration** | [LangChain](https://github.com/langchain-ai/langchain) |
| **LLM Providers** | OpenAI, Anthropic, Google, Ollama |
| **Model Hub** | HuggingFace Hub API |
| **Compression** | llama.cpp, BitsAndBytes, ONNX Runtime |
| **Persistence** | SQLite |
| **Testing** | pytest (tests) |

## 🧪 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sintra

# Format code
ruff format src tests
ruff check --fix src tests

# Debug mode (no LLM calls)
sintra --debug

# Mock mode (fast iteration)
sintra --mock
```

## 📁 Project Structure

```
sintra/
├── src/sintra/
│   ├── agents/           # LangGraph nodes & tools
│   │   ├── factory.py    # LLM factory (OpenAI/Anthropic/Ollama)
│   │   ├── nodes.py      # Architect, Benchmarker, Critic, Reporter
│   │   ├── planner.py    # Strategic optimization planner
│   │   ├── react_architect.py  # ReAct pattern implementation
│   │   ├── reflector.py  # Self-reflection on failures
│   │   └── tools.py      # 5 architect tools
│   ├── benchmarks/       # Execution & measurement
│   ├── compression/      # GGUF, BnB, ONNX backends
│   ├── profiles/         # Hardware detection & profiles
│   ├── persistence/      # SQLite history database
│   ├── cli.py            # Command-line interface
│   └── main.py           # LangGraph workflow
├── tests/                # tests
├── profiles/             # Example hardware profiles
└── outputs/              # Optimized models & configs
```
---

<div align="center">

**Built with curiosity 🔬**

[Report Bug](https://github.com/Kalyankr/sintra/issues) · [Request Feature](https://github.com/Kalyankr/sintra/issues)

</div>