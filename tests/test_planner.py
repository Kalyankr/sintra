"""Tests for the planner agent."""

import pytest

from sintra.agents.planner import (
    OptimizationPlan,
    OptimizationStep,
    _create_default_plan,
    _create_rule_based_plan,
    _estimate_model_size,
    get_plan_guidance,
    planner_node,
)
from sintra.profiles.models import Constraints, HardwareProfile, Targets


@pytest.fixture
def mock_state():
    """Create a mock state for testing."""
    return {
        "target_model_id": "meta-llama/Llama-2-7b-hf",
        "profile": HardwareProfile(
            name="test_device",
            constraints=Constraints(
                vram_gb=8.0,
                max_model_size_mb=4096,
                has_cuda=True,
            ),
            targets=Targets(
                min_tokens_per_second=20,
                min_accuracy_score=0.8,
            ),
        ),
        "use_debug": False,
        "llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
    }


@pytest.fixture
def constrained_state():
    """Create a constrained state for testing aggressive planning."""
    return {
        "target_model_id": "meta-llama/Llama-2-7b-hf",
        "profile": HardwareProfile(
            name="raspberry_pi",
            constraints=Constraints(
                vram_gb=2.0,
                max_model_size_mb=1024,
                has_cuda=False,
            ),
            targets=Targets(
                min_tokens_per_second=5,
                min_accuracy_score=0.7,
            ),
        ),
        "use_debug": False,
        "llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
    }


@pytest.fixture
def high_speed_state():
    """Create a state requiring high speed."""
    return {
        "target_model_id": "meta-llama/Llama-2-7b-hf",
        "profile": HardwareProfile(
            name="gpu_server",
            constraints=Constraints(
                vram_gb=24.0,
                max_model_size_mb=16384,
                has_cuda=True,
            ),
            targets=Targets(
                min_tokens_per_second=60,
                min_accuracy_score=0.75,
            ),
        ),
        "use_debug": False,
        "llm_config": {"provider": "openai", "model": "gpt-4o-mini"},
    }


class TestOptimizationStep:
    """Tests for OptimizationStep model."""
    
    def test_create_step(self):
        """Test creating an optimization step."""
        step = OptimizationStep(
            step_number=1,
            strategy="explore",
            description="Try 4-bit quantization",
            target_bits=4,
            target_pruning=0.1,
            rationale="Good starting point",
        )
        assert step.step_number == 1
        assert step.strategy == "explore"
        assert step.target_bits == 4
        assert step.target_pruning == 0.1
    
    def test_step_optional_fields(self):
        """Test step with optional fields."""
        step = OptimizationStep(
            step_number=1,
            strategy="explore",
            description="General exploration",
            rationale="Try different options",
        )
        assert step.target_bits is None
        assert step.target_pruning is None


class TestOptimizationPlan:
    """Tests for OptimizationPlan model."""
    
    def test_create_plan(self):
        """Test creating an optimization plan."""
        plan = OptimizationPlan(
            model_id="test/model",
            hardware_name="test_device",
            overall_strategy="balanced",
            steps=[
                OptimizationStep(
                    step_number=1,
                    strategy="explore",
                    description="Test step",
                    rationale="Testing",
                ),
            ],
            max_iterations=5,
            early_stop_threshold=0.9,
            confidence=0.8,
        )
        assert plan.model_id == "test/model"
        assert plan.overall_strategy == "balanced"
        assert len(plan.steps) == 1
        assert plan.confidence == 0.8
    
    def test_plan_defaults(self):
        """Test plan default values."""
        plan = OptimizationPlan(
            model_id="test/model",
            hardware_name="test_device",
            overall_strategy="balanced",
        )
        assert plan.steps == []
        assert plan.max_iterations == 10
        assert plan.early_stop_threshold == 0.95
        assert plan.confidence == 0.5


class TestEstimateModelSize:
    """Tests for model size estimation."""
    
    def test_70b_model(self):
        """Test detecting 70B models."""
        assert _estimate_model_size("meta-llama/Llama-2-70b-chat") == "70B"
        assert _estimate_model_size("Qwen-72B-Chat") == "70B"
    
    def test_13b_model(self):
        """Test detecting 13B models."""
        assert _estimate_model_size("meta-llama/Llama-2-13b-hf") == "13B"
        assert _estimate_model_size("vicuna-14b") == "13B"
    
    def test_7b_model(self):
        """Test detecting 7B models."""
        assert _estimate_model_size("meta-llama/Llama-2-7b-hf") == "7B"
        assert _estimate_model_size("mistral-8b-instruct") == "7B"
    
    def test_3b_model(self):
        """Test detecting 3B models."""
        assert _estimate_model_size("phi-3b") == "3B"
    
    def test_1b_model(self):
        """Test detecting 1B models."""
        assert _estimate_model_size("TinyLlama-1.1b") == "1B"
        assert _estimate_model_size("pythia-1.3b") == "1B"
    
    def test_small_model_keywords(self):
        """Test detecting small models by keyword."""
        assert _estimate_model_size("llama-tiny") == "1B"
        assert _estimate_model_size("gpt2-small") == "1B"
        assert _estimate_model_size("mini-model") == "1B"
    
    def test_unknown_model(self):
        """Test unknown model size."""
        assert _estimate_model_size("custom-model") == "Unknown"


class TestRuleBasedPlan:
    """Tests for rule-based planning."""
    
    def test_constrained_hardware_aggressive(self, constrained_state):
        """Test that constrained hardware leads to aggressive strategy."""
        plan = _create_rule_based_plan(constrained_state, "", "7B")
        
        assert plan.overall_strategy == "aggressive"
        assert len(plan.steps) >= 2
        assert plan.steps[0].target_bits == 4  # Start aggressive
    
    def test_high_speed_target_aggressive(self, high_speed_state):
        """Test that high TPS target leads to aggressive strategy."""
        plan = _create_rule_based_plan(high_speed_state, "", "7B")
        
        assert plan.overall_strategy == "aggressive"
        assert plan.steps[0].target_bits <= 4  # Start with low bits
    
    def test_balanced_strategy(self, mock_state):
        """Test balanced strategy for normal conditions."""
        plan = _create_rule_based_plan(mock_state, "", "7B")
        
        assert plan.overall_strategy == "balanced"
        assert len(plan.steps) >= 3
        assert plan.steps[0].target_bits == 4  # Balanced start


class TestDefaultPlan:
    """Tests for default plan creation."""
    
    def test_debug_plan(self, mock_state):
        """Test creating a debug plan."""
        plan = _create_default_plan(mock_state)
        
        assert plan.overall_strategy == "balanced"
        assert len(plan.steps) == 1
        assert plan.max_iterations == 1
        assert plan.confidence == 1.0


class TestGetPlanGuidance:
    """Tests for getting plan guidance."""
    
    def test_get_first_step(self):
        """Test getting the first step."""
        plan = OptimizationPlan(
            model_id="test/model",
            hardware_name="test_device",
            overall_strategy="balanced",
            steps=[
                OptimizationStep(
                    step_number=1,
                    strategy="explore",
                    description="First step",
                    target_bits=4,
                    rationale="Start",
                ),
                OptimizationStep(
                    step_number=2,
                    strategy="exploit",
                    description="Second step",
                    target_bits=5,
                    rationale="Adjust",
                ),
            ],
        )
        
        step = get_plan_guidance(plan, 1)
        assert step is not None
        assert step.step_number == 1
        assert step.target_bits == 4
    
    def test_get_second_step(self):
        """Test getting the second step."""
        plan = OptimizationPlan(
            model_id="test/model",
            hardware_name="test_device",
            overall_strategy="balanced",
            steps=[
                OptimizationStep(
                    step_number=1,
                    strategy="explore",
                    description="First step",
                    rationale="Start",
                ),
                OptimizationStep(
                    step_number=2,
                    strategy="exploit",
                    description="Second step",
                    rationale="Adjust",
                ),
            ],
        )
        
        step = get_plan_guidance(plan, 2)
        assert step is not None
        assert step.step_number == 2
    
    def test_beyond_plan_length(self):
        """Test getting step beyond plan length."""
        plan = OptimizationPlan(
            model_id="test/model",
            hardware_name="test_device",
            overall_strategy="balanced",
            steps=[
                OptimizationStep(
                    step_number=1,
                    strategy="explore",
                    description="Only step",
                    rationale="Single",
                ),
            ],
        )
        
        step = get_plan_guidance(plan, 5)
        assert step is None


class TestPlannerNode:
    """Tests for the planner node function."""
    
    def test_debug_mode_skips_llm(self, mock_state):
        """Test that debug mode uses default plan."""
        mock_state["use_debug"] = True
        
        result = planner_node(mock_state)
        
        assert "optimization_plan" in result
        plan = result["optimization_plan"]
        assert plan.max_iterations == 1
        assert "Debug" in plan.steps[0].description
    
    def test_planner_creates_plan(self, mock_state):
        """Test that planner creates a valid plan."""
        # Use debug mode to avoid LLM call
        mock_state["use_debug"] = True
        
        result = planner_node(mock_state)
        
        assert "optimization_plan" in result
        plan = result["optimization_plan"]
        assert isinstance(plan, OptimizationPlan)
        assert plan.model_id == mock_state["target_model_id"]
        assert plan.hardware_name == mock_state["profile"].name
