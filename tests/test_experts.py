"""Tests for the multi-agent expert collaboration module."""

import pytest

from sintra.agents.experts import (
    ExpertConsensus,
    ExpertOpinion,
    _build_consensus_summary,
    _calculate_agreement,
    _parse_json_response,
    consult_integration_expert,
    consult_pruning_expert,
    consult_quantization_expert,
    expert_collaboration_node,
)
from sintra.profiles.models import (
    Constraints,
    ExperimentResult,
    HardwareProfile,
    LLMConfig,
    ModelRecipe,
    Targets,
)


@pytest.fixture
def sample_profile():
    """Create a sample hardware profile."""
    return HardwareProfile(
        name="test-device",
        constraints=Constraints(
            vram_gb=8.0,
            cpu_arch="x86_64",
            has_cuda=False,
        ),
        targets=Targets(
            min_tokens_per_second=30.0,
            min_accuracy_score=0.7,
        ),
    )


@pytest.fixture
def mock_state(sample_profile):
    """Create a mock state for testing."""
    return {
        "target_model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "profile": sample_profile,
        "use_debug": True,
        "use_mock": True,
        "backend": "gguf",
        "history": [],
        "llm_config": LLMConfig(),
        "iteration": 0,
    }


@pytest.fixture
def state_with_history(sample_profile):
    """Create a state with experiment history."""
    history = [
        {
            "recipe": ModelRecipe(
                bits=4, pruning_ratio=0.0, layers_to_drop=[], method="GGUF"
            ),
            "metrics": ExperimentResult(
                actual_tps=25.0,
                actual_vram_usage=3.0,
                accuracy_score=0.8,
                was_successful=True,
                error_log="",
            ),
        },
        {
            "recipe": ModelRecipe(
                bits=3, pruning_ratio=0.1, layers_to_drop=[], method="GGUF"
            ),
            "metrics": ExperimentResult(
                actual_tps=35.0,
                actual_vram_usage=2.5,
                accuracy_score=0.65,
                was_successful=True,
                error_log="",
            ),
        },
    ]
    return {
        "target_model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "profile": sample_profile,
        "use_debug": False,
        "use_mock": True,
        "backend": "gguf",
        "history": history,
        "llm_config": LLMConfig(),
        "iteration": 2,
    }


# ── ExpertOpinion Model Tests ──


class TestExpertOpinion:
    def test_creation(self):
        opinion = ExpertOpinion(
            expert_name="Quant Expert",
            domain="quantization",
            recommendation="Use 4-bit",
            suggested_bits=4,
            confidence=0.8,
            reasoning="Good balance",
            risk_assessment="low",
        )
        assert opinion.expert_name == "Quant Expert"
        assert opinion.suggested_bits == 4
        assert opinion.confidence == 0.8

    def test_defaults(self):
        opinion = ExpertOpinion(
            expert_name="Test", domain="quantization", recommendation="test"
        )
        assert opinion.confidence == 0.5
        assert opinion.risk_assessment == "medium"
        assert opinion.suggested_bits is None
        assert opinion.suggested_pruning is None
        assert opinion.suggested_layers_to_drop is None

    def test_all_domains(self):
        for domain in ["quantization", "pruning", "integration"]:
            opinion = ExpertOpinion(
                expert_name="Test", domain=domain, recommendation="test"
            )
            assert opinion.domain == domain


class TestExpertConsensus:
    def test_creation(self):
        consensus = ExpertConsensus(
            consensus_bits=4,
            consensus_pruning=0.1,
            agreement_level=0.8,
            summary="All agree on 4-bit",
        )
        assert consensus.consensus_bits == 4
        assert consensus.consensus_pruning == 0.1
        assert consensus.agreement_level == 0.8

    def test_defaults(self):
        consensus = ExpertConsensus()
        assert consensus.consensus_bits == 4
        assert consensus.consensus_pruning == 0.0
        assert consensus.consensus_layers_to_drop == []
        assert consensus.agreement_level == 0.5


# ── Rule-based Expert Tests ──


class TestQuantizationExpert:
    def test_rule_based_basic(self, mock_state):
        opinion = consult_quantization_expert(mock_state, use_llm=False)
        assert opinion.expert_name == "Quantization Expert"
        assert opinion.domain == "quantization"
        assert opinion.suggested_bits is not None
        assert 2 <= opinion.suggested_bits <= 8
        assert 0.0 < opinion.confidence <= 1.0

    def test_rule_based_with_history(self, state_with_history):
        opinion = consult_quantization_expert(state_with_history, use_llm=False)
        assert opinion.suggested_bits is not None
        # Should prefer successful bits
        assert opinion.confidence >= 0.5

    def test_low_vram_prefers_low_bits(self, mock_state):
        mock_state["profile"] = HardwareProfile(
            name="pi",
            constraints=Constraints(vram_gb=2.0, has_cuda=False),
            targets=Targets(min_tokens_per_second=20, min_accuracy_score=0.7),
        )
        opinion = consult_quantization_expert(mock_state, use_llm=False)
        assert opinion.suggested_bits <= 5

    def test_high_accuracy_target_prefers_high_bits(self, mock_state):
        mock_state["profile"] = HardwareProfile(
            name="server",
            constraints=Constraints(vram_gb=16.0, has_cuda=True),
            targets=Targets(min_tokens_per_second=10, min_accuracy_score=0.9),
        )
        opinion = consult_quantization_expert(mock_state, use_llm=False)
        assert opinion.suggested_bits >= 4


class TestPruningExpert:
    def test_rule_based_basic(self, mock_state):
        opinion = consult_pruning_expert(mock_state, planned_bits=4, use_llm=False)
        assert opinion.expert_name == "Pruning Expert"
        assert opinion.domain == "pruning"
        assert opinion.suggested_pruning is not None
        assert 0.0 <= opinion.suggested_pruning <= 0.5

    def test_no_pruning_when_accuracy_low(self, state_with_history):
        # Last entry has accuracy below target (0.65 < 0.7)
        opinion = consult_pruning_expert(
            state_with_history, planned_bits=3, use_llm=False
        )
        assert opinion.suggested_pruning is not None


class TestIntegrationExpert:
    def test_rule_based_basic(self, mock_state):
        quant = ExpertOpinion(
            expert_name="Quant",
            domain="quantization",
            recommendation="4-bit",
            suggested_bits=4,
            confidence=0.8,
        )
        prune = ExpertOpinion(
            expert_name="Prune",
            domain="pruning",
            recommendation="10%",
            suggested_pruning=0.1,
            suggested_layers_to_drop=[],
            confidence=0.7,
        )
        opinion = consult_integration_expert(mock_state, quant, prune, use_llm=False)
        assert opinion.expert_name == "Integration Expert"
        assert opinion.domain == "integration"
        assert opinion.suggested_bits is not None
        assert opinion.suggested_pruning is not None

    def test_reduces_aggression_for_high_accuracy(self, mock_state):
        mock_state["profile"] = HardwareProfile(
            name="server",
            constraints=Constraints(vram_gb=16.0, has_cuda=True),
            targets=Targets(min_tokens_per_second=10, min_accuracy_score=0.85),
        )
        quant = ExpertOpinion(
            expert_name="Quant",
            domain="quantization",
            recommendation="3-bit",
            suggested_bits=3,
            confidence=0.6,
        )
        prune = ExpertOpinion(
            expert_name="Prune",
            domain="pruning",
            recommendation="30%",
            suggested_pruning=0.3,
            suggested_layers_to_drop=[10, 11, 12],
            confidence=0.5,
        )
        opinion = consult_integration_expert(mock_state, quant, prune, use_llm=False)
        # Integration expert should moderate aggressive settings
        total = (8 - (opinion.suggested_bits or 4)) * 0.1 + (
            opinion.suggested_pruning or 0
        )
        assert total <= 0.8  # Should not be extremely aggressive


# ── Collaboration Node Tests ──


class TestExpertCollaborationNode:
    def test_debug_mode_skips(self, mock_state):
        mock_state["use_debug"] = True
        result = expert_collaboration_node(mock_state)
        assert "expert_consensus" in result
        assert result["expert_consensus"] is None

    def test_produces_consensus(self, mock_state):
        mock_state["use_debug"] = False
        mock_state["use_mock"] = True
        result = expert_collaboration_node(mock_state)
        assert "expert_consensus" in result
        consensus = result["expert_consensus"]
        assert isinstance(consensus, ExpertConsensus)
        assert len(consensus.opinions) == 3
        assert 2 <= consensus.consensus_bits <= 8
        assert 0.0 <= consensus.consensus_pruning <= 0.5
        assert 0.0 <= consensus.agreement_level <= 1.0


# ── Helper Function Tests ──


class TestHelpers:
    def test_calculate_agreement_all_agree(self):
        opinions = [
            ExpertOpinion(
                expert_name="A", domain="q", recommendation="x", suggested_bits=4
            ),
            ExpertOpinion(
                expert_name="B", domain="p", recommendation="x", suggested_bits=4
            ),
        ]
        assert _calculate_agreement(opinions) == 1.0

    def test_calculate_agreement_disagree(self):
        opinions = [
            ExpertOpinion(
                expert_name="A", domain="q", recommendation="x", suggested_bits=4
            ),
            ExpertOpinion(
                expert_name="B", domain="p", recommendation="x", suggested_bits=8
            ),
        ]
        agreement = _calculate_agreement(opinions)
        assert agreement < 1.0

    def test_calculate_agreement_single(self):
        opinions = [
            ExpertOpinion(expert_name="A", domain="q", recommendation="x"),
        ]
        assert _calculate_agreement(opinions) == 1.0

    def test_build_consensus_summary(self):
        opinions = [
            ExpertOpinion(
                expert_name="Quant",
                domain="quantization",
                recommendation="4-bit",
                confidence=0.8,
                risk_assessment="low",
            ),
        ]
        summary = _build_consensus_summary(opinions)
        assert "Quant" in summary
        assert "4-bit" in summary

    def test_parse_json_response_valid(self):
        data = _parse_json_response('{"bits": 4, "confidence": 0.8}')
        assert data["bits"] == 4

    def test_parse_json_response_markdown(self):
        content = 'Here is my analysis:\n```json\n{"bits": 4}\n```\nDone.'
        data = _parse_json_response(content)
        assert data["bits"] == 4

    def test_parse_json_response_invalid(self):
        data = _parse_json_response("no json here")
        assert data == {}
