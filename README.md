<div align="center">

# 🧠 Sintra

### Autonomous AI Agent for Edge Model Optimization

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Tests](https://img.shields.io/badge/tests-404%20passed-brightgreen.svg)]()
[![CI](https://github.com/Kalyankr/sintra/actions/workflows/ci.yml/badge.svg)](https://github.com/Kalyankr/sintra/actions/workflows/ci.yml)

**Sintra** (Synthetic Intelligence for Targeted Runtime Architectures) is a fully autonomous agentic framework that optimizes Large Language Models for resource-constrained edge devices.

[Quick Start](#-quick-start) • [Features](#-key-features) • [Architecture](#-agentic-architecture) • [Dashboard](#-web-dashboard)

</div>

---

## 🎬 Demo

<p align="center">
  <img src="assets/sintra_demo.gif" alt="Sintra Demo" width="700">
</p>

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
uv sync --extra all

# Run - zero flags needed!
uv run sintra
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
- **Tool Calling**: 6 specialized tools for model research & benchmarking
- **Multi-Agent Experts**: 3 domain experts (quantization, pruning, integration) collaborate on recipes
- **ReAct Pattern**: Reason → Act → Observe loop
- **Self-Reflection**: Learns from failures automatically
- **Adaptive Learning**: Calibrates predictions from past experiment history
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
- **Adaptive calibration**: Accuracy/TPS/size estimates improve over time
- **Checkpointing**: Resume interrupted optimizations

### 📈 Community Benchmarks
- **Open LLM Leaderboard** integration via HuggingFace Hub
- Look up MMLU, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K scores
- Fallback reference data for 8 major model families

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
│  ┌──────────┐    ┌──────────────┐                               │
│  │ PLANNER  │───▶│   EXPERTS    │                               │
│  │  (LLM)   │    │ ┌──────────┐ │                               │
│  └──────────┘    │ │ Quant    │ │                               │
│                  │ │ Pruning  │ │                               │
│                  │ │ Integr.  │ │                               │
│                  │ └──────────┘ │                               │
│                  └──────┬───────┘                               │
│                         │                                       │
│                         ▼                                       │
│        ┌─────────────────────────────────────────┐              │
│        │           REACT ARCHITECT               │              │
│        │  ┌────────────────────────────────────┐ │              │
│        │  │ TOOLS:                             │ │              │
│        │  │ • get_model_architecture           │ │              │
│        │  │ • search_similar_models            │ │              │
│        │  │ • estimate_compression_impact      │ │              │
│        │  │ • query_hardware_capabilities      │ │              │
│        │  │ • lookup_quantization_benchmarks   │ │              │
│        │  │ • query_community_benchmarks       │ │              │
│        │  └────────────────────────────────────┘ │              │
│        └───────────────────┬─────────────────────┘              │
│                            │                                    │
│                            ▼                                    │
│                   ┌──────────────┐                              │
│                   │ BENCHMARKER  │                              │
│                   │  (Executor)  │                              │
│                   └──────┬───────┘                              │
│                          │                                      │
│                          ▼                                      │
│  ┌───────────┐  ┌──────────────┐                               │
│  │ REFLECTOR │◀─│    CRITIC    │                               │
│  │   (LLM)   │  │ (LLM Router) │                               │
│  └─────┬─────┘  └──────┬───────┘                               │
│        │               │                                        │
│        │    ┌──────────┴──────────┐                             │
│        │    │                     │                              │
│        ▼    ▼                     ▼                              │
│   [Continue Loop]            [REPORTER]                         │
│                               (Output)                          │
└──────────────────────────────────────────────────────────────────┘
```

## �️ Web Dashboard

Launch an interactive Gradio dashboard to explore optimization history, compare runs, and browse hardware profiles:

```bash
# Launch dashboard
sintra --ui

# Custom port
sintra --ui --ui-port 8080
```

<p align="center">
  <strong>Tabs:</strong> History • Runs • Profiles • About
</p>

> **Note:** Requires `gradio` — install with `uv sync --extra ui`

## �📖 CLI Reference

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
| `--no-experts` | - | Disable multi-agent experts |
| `--no-llm-routing` | - | Use rule-based routing |

### Web Dashboard
| Flag | Default | Description |
|------|---------|-------------|
| `--ui` | - | Launch Gradio web dashboard |
| `--ui-port` | 7860 | Dashboard port |

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
| **Benchmarks** | Open LLM Leaderboard (HuggingFace) |
| **Compression** | llama.cpp, BitsAndBytes, ONNX Runtime |
| **Persistence** | SQLite |
| **Web Dashboard** | [Gradio](https://gradio.app) (optional) |
| **CI/CD** | GitHub Actions |
| **Testing** | pytest (404 tests) |

## 🧪 Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest -v

# Run with coverage
uv run pytest --cov=sintra

# Lint & format
uv run ruff format src tests
uv run ruff check --fix src tests

# Type checking
uv run mypy src/sintra

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
│   │   ├── experts.py    # Multi-agent expert collaboration
│   │   ├── leaderboard.py # Open LLM Leaderboard integration
│   │   ├── adaptive.py   # Adaptive learning from history
│   │   └── tools.py      # 6 architect tools
│   ├── benchmarks/       # Execution & measurement
│   ├── compression/      # GGUF, BnB, ONNX backends
│   ├── profiles/         # Hardware detection & profiles
│   ├── persistence/      # SQLite history database
│   ├── ui/               # Console, progress & Gradio dashboard
│   ├── cli.py            # Command-line interface
│   └── main.py           # LangGraph workflow
├── tests/                # 404 tests
├── profiles/             # Example hardware profiles
└── outputs/              # Optimized models & configs
```
---

<div align="center">

**Built with curiosity 🔬**

[Report Bug](https://github.com/Kalyankr/sintra/issues) · [Request Feature](https://github.com/Kalyankr/sintra/issues)

</div>