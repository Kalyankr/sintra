# Sintra Project Report

**Date:** January 12, 2026  
**Version:** 0.1.0  
**Repository:** https://github.com/Kalyankr/sintra

---

## Executive Summary

**Sintra** is an autonomous AI agent that optimizes Large Language Models (LLMs) for edge hardware deployment. It uses an LLM "Architect" to iteratively propose compression strategies, benchmark them on target hardware, and converge on optimal configurations.

**Current Status:** 🟡 **Functional Prototype** — Core workflow operational, real compression pipeline implemented but requires llama.cpp toolchain.

---

## Project Goal

> Automatically compress any HuggingFace LLM to run efficiently on resource-constrained edge devices (Raspberry Pi, Jetson, Apple Silicon) while meeting user-defined performance targets (TPS, accuracy, VRAM).

---

## What Has Been Completed

### Phase 0: Bug Fixes & Code Quality (Session 1)
| Task | Status | Details |
|------|--------|---------|
| Entry point fix | ✅ | Was printing "Hello" instead of running workflow |
| Pydantic model alignment | ✅ | Added missing fields for YAML parsing |
| Router logic fix | ✅ | Invalid return value `"continue"` → `"architect"` |
| State type fixes | ✅ | Fixed `best_recipe` type mismatch |
| Error handling | ✅ | Added `ProfileLoadError`, `LLMConnectionError`, `MissingAPIKeyError` |
| Comprehensive crash prevention | ✅ | Try/catch for all failure modes |
| Test suite | ✅ | 80 tests covering all modules |
| Type annotations | ✅ | Full typing + `py.typed` marker |

### Phase 1: Real Compression Pipeline (Session 2)
| Component | Status | Description |
|-----------|--------|-------------|
| `compression/downloader.py` | ✅ | HuggingFace model download with caching |
| `compression/quantizer.py` | ✅ | GGUF conversion & quantization (Q2_K-Q8_0) |
| `compression/evaluator.py` | ✅ | Perplexity-based accuracy measurement |
| Worker integration | ✅ | Supports REAL and LEGACY modes |
| CLI flags | ✅ | `--model-id`, `--hf-token`, `--real-compression` |
| Dependencies | ✅ | Added `huggingface_hub`, `safetensors` |

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SINTRA WORKFLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│   │  ARCHITECT   │───▶│  BENCHMARKER │───▶│    CRITIC    │             │
│   │   (LLM)      │    │    (Lab)     │    │   (Judge)    │             │
│   └──────────────┘    └──────────────┘    └──────┬───────┘             │
│          ▲                                       │                      │
│          │            ┌──────────────┐           │                      │
│          └────────────│   REPORTER   │◀──────────┘                      │
│           (retry)     │  (Archivist) │     (converged)                  │
│                       └──────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                       COMPRESSION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   HuggingFace Model ID                                                  │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐ │
│   │ ModelDownloader  │────▶│  GGUFQuantizer   │────▶│ AccuracyEval   │ │
│   │                  │     │                  │     │                │ │
│   │ • HF Hub API     │     │ • convert_hf_to_ │     │ • Perplexity   │ │
│   │ • Caching        │     │   gguf.py        │     │ • Quick tests  │ │
│   │ • Gated models   │     │ • llama-quantize │     │                │ │
│   └──────────────────┘     │ • Q2_K → Q8_0    │     └────────────────┘ │
│                            └──────────────────┘                         │
│                                                                         │
│   Cache: ~/.cache/sintra/{downloads,gguf,quantized}/                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
sintra/
├── pyproject.toml              # Dependencies, build config
├── profiles/                   # Hardware target definitions
│   ├── raspberry_pi_5.yaml
│   └── mac_mini_m4.yaml
├── samples/
│   └── example_output.json     # Sample optimization result
├── src/sintra/
│   ├── __init__.py
│   ├── main.py                 # Entry point, workflow execution
│   ├── cli.py                  # Argument parsing
│   ├── agents/
│   │   ├── factory.py          # LLM provider factory
│   │   ├── nodes.py            # Workflow nodes (architect, benchmarker, critic, reporter)
│   │   ├── state.py            # TypedDict state definition
│   │   └── utils.py            # History formatting
│   ├── benchmarks/
│   │   ├── executor.py         # Mock and Standalone executors
│   │   └── worker/
│   │       └── runner.py       # Subprocess worker for benchmarking
│   ├── compression/            # NEW - Real compression pipeline
│   │   ├── downloader.py       # HuggingFace model download
│   │   ├── quantizer.py        # GGUF quantization
│   │   └── evaluator.py        # Accuracy measurement
│   ├── profiles/
│   │   ├── models.py           # Pydantic models
│   │   └── parser.py           # YAML profile loader
│   └── ui/
│       └── console.py          # Rich console output
└── tests/                      # 80 tests
    ├── test_compression.py
    ├── test_executor.py
    ├── test_models.py
    ├── test_nodes.py
    ├── test_parser.py
    ├── test_state.py
    └── test_utils.py
```

---

## How to Use

### Basic Usage (Debug Mode - No LLM Required)
```bash
uv run sintra profiles/raspberry_pi_5.yaml --debug
```

### With LLM Architect
```bash
# Ollama (local)
ollama serve  # Start Ollama first
uv run sintra profiles/raspberry_pi_5.yaml

# OpenAI
export OPENAI_API_KEY=sk-...
uv run sintra profiles/raspberry_pi_5.yaml --provider openai --model gpt-4o

# Anthropic
export ANTHROPIC_API_KEY=...
uv run sintra profiles/raspberry_pi_5.yaml --provider anthropic --model claude-3-5-sonnet-latest
```

### Real Compression Mode (Requires llama.cpp)
```bash
# Install llama.cpp first
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make llama-quantize

# Run with real compression
uv run sintra profiles/raspberry_pi_5.yaml \
    --real-compression \
    --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --provider openai --model gpt-4o
```

---

## What Still Needs Implementation

### Phase 2: Pruning & Layer Dropping (Priority: High)
| Task | Effort | Description |
|------|--------|-------------|
| Pre-conversion layer removal | 3 days | Remove transformer layers before GGUF conversion |
| `pruning_ratio` implementation | 2 days | Apply structured pruning to attention/FFN weights |
| Integration with Architect prompts | 1 day | Make Architect aware of pruning effects |

**Why needed:** Currently `pruning_ratio` and `layers_to_drop` in ModelRecipe are ignored.

### Phase 3: Advanced Accuracy Evaluation (Priority: Medium)
| Task | Effort | Description |
|------|--------|-------------|
| Full perplexity calculation | 2 days | Proper log-likelihood computation |
| MMLU subset evaluation | 2 days | Standard benchmark for reasoning |
| Custom task evaluation | 1 day | User-defined eval datasets |

**Why needed:** Current accuracy is estimated heuristically, not measured properly.

### Phase 4: Production Features (Priority: Low)
| Task | Effort | Description |
|------|--------|-------------|
| Docker isolation | 3 days | Sandboxed worker execution with resource limits |
| Multi-objective optimization | 5 days | Pareto frontier for TPS/accuracy tradeoffs |
| Model export | 2 days | Save optimized model to disk (not just recipe JSON) |
| CI/CD pipeline | 1 day | GitHub Actions for tests, linting, releases |
| PyPI publishing | 1 day | `pip install sintra` |

### Future Enhancements (Ideas)
| Enhancement | Description |
|-------------|-------------|
| **Auto-discovery mode** | "Find the best possible accuracy within my hardware limits" — no accuracy floor required. The agent explores the Pareto frontier and returns the optimal accuracy achievable given VRAM/TPS constraints. |
| **Presets for common scenarios** | Built-in profiles like `--preset chatbot` (high TPS, moderate accuracy) or `--preset batch` (low TPS, high accuracy) for users who don't want to define custom targets. |
| **Target guardrails** | Warn users if targets are unrealistic (e.g., 100 TPS on Raspberry Pi) before running the optimization loop. |

---

## Quantization Support Matrix

| Bits | Type | Size Reduction | Quality | Implemented |
|------|------|----------------|---------|-------------|
| 2 | Q2_K | ~87% | Poor | ✅ |
| 3 | Q3_K_M | ~81% | Fair | ✅ |
| 4 | Q4_K_M | ~75% | Good | ✅ (recommended) |
| 5 | Q5_K_M | ~69% | Very Good | ✅ |
| 6 | Q6_K | ~62% | Excellent | ✅ |
| 8 | Q8_0 | ~50% | Near-FP16 | ✅ |

---

## Test Coverage

```
tests/test_compression.py    21 tests  ✅
tests/test_executor.py        8 tests  ✅
tests/test_models.py         17 tests  ✅
tests/test_nodes.py          18 tests  ✅
tests/test_parser.py          8 tests  ✅
tests/test_state.py           3 tests  ✅
tests/test_utils.py           5 tests  ✅
─────────────────────────────────────────
TOTAL                        80 tests  ✅
```

---

## Dependencies

### Runtime
- `langgraph` - Agent workflow orchestration
- `langchain-*` - LLM provider integrations (OpenAI, Anthropic, Google, Ollama)
- `llama-cpp-python` - GGUF model loading and inference
- `huggingface_hub` - Model downloading
- `pydantic` - Data validation
- `rich` - Console UI
- `python-dotenv` - Environment variable management

### External (for real compression)
- `llama.cpp` - GGUF conversion and quantization binaries

---

## Git Branches

| Branch | Status | Description |
|--------|--------|-------------|
| `main` | ✅ Stable | Bug fixes, error handling, tests |
| `feature/real-compression` | ✅ Ready for merge | Compression pipeline implementation |

---

## Summary

| Aspect | Status |
|--------|--------|
| **Core Workflow** | ✅ Fully functional |
| **LLM Integration** | ✅ OpenAI, Anthropic, Google, Ollama |
| **Error Handling** | ✅ Comprehensive |
| **Test Coverage** | ✅ 80 tests passing |
| **Model Download** | ✅ HuggingFace Hub with caching |
| **Quantization** | ✅ Q2_K through Q8_0 |
| **Accuracy Eval** | 🟡 Basic (needs full perplexity) |
| **Pruning** | ❌ Not implemented |
| **Layer Dropping** | ❌ Not implemented |
| **Docker Isolation** | ❌ Not implemented |

---

## Estimated Remaining Work

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 2: Pruning/Layer Dropping | ~6 days | 🔴 High |
| Phase 3: Accuracy Evaluation | ~5 days | 🟡 Medium |
| Phase 4: Production Features | ~12 days | 🟢 Low |
| **Total** | **~23 days** | |

**Recommended Next Step:** Merge `feature/real-compression` to `main`, then implement pruning/layer dropping (Phase 2).

---

*Generated: January 12, 2026*
