# Sintra Code Health Report & Action Plan

**Generated:** January 12, 2026  
**Repository:** `/home/kkr/dev/sintra`  
**Version:** 0.1.0

---

## Executive Summary

Sintra is an autonomous agentic framework designed to optimize LLMs for edge hardware. The current implementation provides a **well-architected workflow prototype** but lacks core functionality to achieve its stated goals.

**Overall Health:** ⚠️ Prototype Stage — Workflow works, core features not implemented

---

## Part 1: Completed Fixes (This Session)

### Critical Bugs Fixed

| Issue | File | Fix Applied |
|-------|------|-------------|
| Entry point printed "Hello from sintra!" instead of running workflow | `src/sintra/__init__.py` | Now imports and calls `main` from `sintra.main` |
| Invalid router return `"continue"` | `src/sintra/agents/nodes.py` | Changed to valid route `"architect"` |
| Conflicting iteration limits (6 vs 10) | `src/sintra/agents/nodes.py` | Unified to `MAX_ITERATIONS = 10` |
| `best_recipe` type mismatch | `src/sintra/agents/state.py` | Fixed to `Dict[str, Union[ModelRecipe, ExperimentResult]]` |
| Missing initial state fields | `src/sintra/main.py` | Added `critic_feedback`, `best_recipe` |
| Pydantic models missing YAML fields | `src/sintra/profiles/models.py` | Added `cpu_arch`, `has_cuda`, `max_latency_ms`, `supported_quantizations` |
| No error handling in parser | `src/sintra/profiles/parser.py` | Added `ProfileLoadError` with proper exception chaining |
| Google provider dead code | `src/sintra/agents/factory.py` | Now actually prefixes model name |
| MockExecutor non-deterministic | `src/sintra/benchmarks/executor.py` | Added seed parameter for reproducibility |

### Architecture Improvements

| Improvement | Files |
|-------------|-------|
| Added `__init__.py` with exports to all subpackages | `agents/`, `benchmarks/`, `profiles/`, `ui/` |
| Created `BenchmarkExecutor` abstract base class | `src/sintra/benchmarks/executor.py` |
| Added type annotations throughout | Multiple files |
| Added `py.typed` marker | `src/sintra/py.typed` |
| Enhanced `pyproject.toml` with pytest, mypy, coverage configs | `pyproject.toml` |

### Test Suite Created

| Test File | Coverage |
|-----------|----------|
| `tests/test_parser.py` | Profile loading, error handling |
| `tests/test_models.py` | All Pydantic models, validators |
| `tests/test_nodes.py` | Agent node functions, routing |
| `tests/test_executor.py` | MockExecutor behavior |
| `tests/test_state.py` | TypedDict structure |
| `tests/test_utils.py` | History formatting |

---

## Part 2: Current State Assessment

### ✅ What Works

| Feature | Status | Notes |
|---------|--------|-------|
| LangGraph workflow orchestration | ✅ Working | Correct state machine with conditional routing |
| LLM-driven recipe generation | ✅ Working | Architect proposes compression strategies |
| CLI interface | ✅ Working | Proper argparse with provider/model options |
| TPS measurement | ✅ Working | Real tokens/second from llama-cpp |
| Hardware profile loading | ✅ Working | YAML parsing with Pydantic validation |
| Mock executor for testing | ✅ Working | Deterministic simulation of compression trade-offs |

### ❌ What Doesn't Work

| Feature | Status | Problem |
|---------|--------|---------|
| Dynamic quantization | ❌ Fake | Worker loads pre-quantized GGUF files, ignores `bits` parameter |
| Pruning | ❌ Not implemented | `pruning_ratio` and `layers_to_drop` are ignored |
| Accuracy measurement | ❌ Hardcoded | Returns `0.90` regardless of actual model performance |
| Model specification | ❌ Hardcoded | Only works with pre-downloaded TinyLlama GGUF files |
| Docker isolation | ❌ Not implemented | Mentioned in README, no code exists |
| RL Controller | ❌ Not implemented | Mentioned in README, no code exists |

### 🔴 Dead Code

| File | Issue |
|------|-------|
| `src/sintra/agents/runner.py` | Complete `SintraRunner` class with real bitsandbytes quantization — **never imported or used** |

---

## Part 3: Action Plan

### Phase 1: Foundation (Priority: Critical)

#### 1.1 Make Worker Actually Compress Models
**Effort:** 3-5 days  
**Files:** `src/sintra/benchmarks/worker/runner.py`

- [ ] Accept model ID as input (not hardcoded TinyLlama)
- [ ] Implement real quantization using `llama.cpp` quantize tool or `bitsandbytes`
- [ ] Implement layer dropping for `layers_to_drop` parameter
- [ ] Map `bits` parameter to actual quantization levels (Q2_K, Q3_K, Q4_K, Q5_K, Q8_0)

#### 1.2 Implement Real Accuracy Measurement
**Effort:** 2-3 days  
**Files:** `src/sintra/benchmarks/worker/runner.py`

- [ ] Add lightweight eval benchmark (subset of MMLU or custom reasoning tests)
- [ ] Return actual accuracy score based on model performance
- [ ] Cache baseline accuracy for comparison

#### 1.3 Integrate or Remove Dead Code
**Effort:** 1 day  
**Files:** `src/sintra/agents/runner.py`

- [ ] Either integrate `SintraRunner` into the workflow OR delete it
- [ ] Remove unused dependencies if not implementing features

### Phase 2: Usability (Priority: High)

#### 2.1 Model Input/Output
**Effort:** 2 days

- [ ] CLI flag to specify source model: `--model-id meta-llama/Llama-3-8B`
- [ ] Save compressed model to disk (not just JSON recipe)
- [ ] Add `--output-dir` flag for artifacts

#### 2.2 Improved Profiles
**Effort:** 1 day

- [ ] Add more hardware profiles (Jetson Nano, M1 Mac, etc.)
- [ ] Validate that `supported_quantizations` is actually used

### Phase 3: Advanced Features (Priority: Medium)

#### 3.1 Docker Isolation
**Effort:** 3-5 days

- [ ] Create Dockerfile for sandboxed execution
- [ ] Implement resource limits (memory, CPU)
- [ ] Add `--use-docker` flag

#### 3.2 Better Optimization Strategy
**Effort:** 5+ days

- [ ] Implement Optuna-based hyperparameter search
- [ ] Add multi-objective optimization (Pareto frontier)
- [ ] Implement early stopping based on constraint violations

### Phase 4: Production Readiness (Priority: Low)

- [ ] Add comprehensive logging with Python `logging` module
- [ ] CI/CD pipeline with GitHub Actions
- [ ] API documentation with Sphinx or mkdocs
- [ ] Publish to PyPI

---

## Appendix: Commands Reference

```bash
# Install dependencies
uv sync

# Run Sintra (debug mode)
uv run sintra profiles/raspberry_pi_5.yaml --debug

# Run Sintra (with OpenAI)
uv run sintra profiles/raspberry_pi_5.yaml --provider openai --model gpt-4o

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/sintra --cov-report=html

# Type checking
uv run mypy src/sintra

# Linting
uv run ruff check src/sintra
```

---

*Report generated by code review session*
