"""Tests for MusiCritic dimension base classes."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.base import (
    BaseDimensionEvaluator,
    DimensionCategory,
    DimensionRegistry,
    DimensionResult,
    EvaluationConfig,
    EvaluationResult,
    QualityDimension,
    QualityResult,
    SafetyDecision,
    SafetyDimension,
    SafetyResult,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


class MockQualityEvaluator(BaseDimensionEvaluator):
    """Mock quality evaluator for testing."""

    dimension = QualityDimension.PROMPT_ADHERENCE
    category = DimensionCategory.QUALITY

    def _evaluate_impl(self, audio, sample_rate, prompt=None, **kwargs):
        return DimensionResult(
            dimension=self.dimension,
            score=0.75,
            confidence=0.9,
            explanation="Mock quality evaluation",
        )


class MockSafetyEvaluator(BaseDimensionEvaluator):
    """Mock safety evaluator for testing."""

    dimension = SafetyDimension.COPYRIGHT
    category = DimensionCategory.SAFETY

    def _evaluate_impl(self, audio, sample_rate, prompt=None, **kwargs):
        return DimensionResult(
            dimension=self.dimension,
            score=0.3,
            confidence=0.95,
            explanation="Mock safety evaluation",
        )


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    duration = 1.0
    sample_rate = 24000
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t), sample_rate


# -----------------------------------------------------------------------------
# DimensionResult Tests
# -----------------------------------------------------------------------------


class TestDimensionResult:
    """Tests for DimensionResult."""

    def test_basic_result(self):
        """Test creating a basic result."""
        result = DimensionResult(
            dimension=QualityDimension.PROMPT_ADHERENCE,
            score=0.8,
        )
        assert result.score == 0.8
        assert result.confidence == 1.0
        assert result.dimension == QualityDimension.PROMPT_ADHERENCE

    def test_result_with_all_fields(self):
        """Test result with all fields populated."""
        result = DimensionResult(
            dimension=QualityDimension.MUSICAL_COHERENCE,
            score=0.65,
            confidence=0.9,
            sub_scores={"harmony": 0.7, "rhythm": 0.6},
            metadata={"key": "C major"},
            explanation="Good coherence",
            timestamps=[(0.0, 0.5), (1.0, 0.7)],
        )
        assert result.sub_scores["harmony"] == 0.7
        assert result.metadata["key"] == "C major"
        assert len(result.timestamps) == 2

    def test_score_validation_too_high(self):
        """Test that score > 1.0 raises error."""
        with pytest.raises(ValueError, match="Score must be between"):
            DimensionResult(
                dimension=QualityDimension.PROMPT_ADHERENCE,
                score=1.5,
            )

    def test_score_validation_too_low(self):
        """Test that score < 0.0 raises error."""
        with pytest.raises(ValueError, match="Score must be between"):
            DimensionResult(
                dimension=QualityDimension.PROMPT_ADHERENCE,
                score=-0.1,
            )

    def test_confidence_validation(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            DimensionResult(
                dimension=QualityDimension.PROMPT_ADHERENCE,
                score=0.5,
                confidence=1.5,
            )

    def test_scaled_score(self):
        """Test scaled score property."""
        result = DimensionResult(
            dimension=QualityDimension.PROMPT_ADHERENCE,
            score=0.75,
        )
        assert result.scaled_score == 75.0

    def test_to_dict(self):
        """Test serialization to dict."""
        result = DimensionResult(
            dimension=QualityDimension.PROMPT_ADHERENCE,
            score=0.8,
            confidence=0.9,
            explanation="Test",
        )
        data = result.to_dict()
        assert data["dimension"] == "prompt_adherence"
        assert data["score"] == 0.8
        assert data["confidence"] == 0.9


# -----------------------------------------------------------------------------
# QualityResult Tests
# -----------------------------------------------------------------------------


class TestQualityResult:
    """Tests for QualityResult."""

    def test_quality_result_creation(self):
        """Test creating quality result."""
        dim_result = DimensionResult(
            dimension=QualityDimension.PROMPT_ADHERENCE,
            score=0.8,
        )
        result = QualityResult(
            overall_score=80.0,
            dimension_results={QualityDimension.PROMPT_ADHERENCE: dim_result},
            confidence=0.9,
            explanation="Good quality",
        )
        assert result.overall_score == 80.0
        assert len(result.dimension_results) == 1

    def test_quality_result_to_dict(self):
        """Test serialization."""
        dim_result = DimensionResult(
            dimension=QualityDimension.PROMPT_ADHERENCE,
            score=0.8,
        )
        result = QualityResult(
            overall_score=80.0,
            dimension_results={QualityDimension.PROMPT_ADHERENCE: dim_result},
        )
        data = result.to_dict()
        assert data["overall_score"] == 80.0
        assert "prompt_adherence" in data["dimensions"]


# -----------------------------------------------------------------------------
# SafetyResult Tests
# -----------------------------------------------------------------------------


class TestSafetyResult:
    """Tests for SafetyResult."""

    def test_safety_result_allow(self):
        """Test safety result with ALLOW decision."""
        dim_result = DimensionResult(
            dimension=SafetyDimension.COPYRIGHT,
            score=0.1,
        )
        result = SafetyResult(
            decision=SafetyDecision.ALLOW,
            dimension_results={SafetyDimension.COPYRIGHT: dim_result},
        )
        assert result.decision == SafetyDecision.ALLOW
        assert len(result.flags) == 0

    def test_safety_result_with_flags(self):
        """Test safety result with flags."""
        dim_result = DimensionResult(
            dimension=SafetyDimension.VOICE_CLONING,
            score=0.85,
        )
        result = SafetyResult(
            decision=SafetyDecision.FLAG,
            dimension_results={SafetyDimension.VOICE_CLONING: dim_result},
            flags=["Possible voice clone detected"],
            evidence={"matched_voice": "Artist A", "similarity": 0.85},
        )
        assert result.decision == SafetyDecision.FLAG
        assert len(result.flags) == 1

    def test_safety_result_to_dict(self):
        """Test serialization."""
        dim_result = DimensionResult(
            dimension=SafetyDimension.COPYRIGHT,
            score=0.1,
        )
        result = SafetyResult(
            decision=SafetyDecision.ALLOW,
            dimension_results={SafetyDimension.COPYRIGHT: dim_result},
        )
        data = result.to_dict()
        assert data["decision"] == "ALLOW"
        assert "copyright" in data["dimensions"]


# -----------------------------------------------------------------------------
# EvaluationResult Tests
# -----------------------------------------------------------------------------


class TestEvaluationResult:
    """Tests for EvaluationResult."""

    def test_full_evaluation_result(self):
        """Test creating full evaluation result."""
        quality = QualityResult(
            overall_score=75.0,
            dimension_results={},
        )
        safety = SafetyResult(
            decision=SafetyDecision.ALLOW,
            dimension_results={},
        )
        result = EvaluationResult(
            quality=quality,
            safety=safety,
            processing_time_ms=150.0,
            audio_duration_sec=30.0,
        )
        assert result.quality.overall_score == 75.0
        assert result.safety.decision == SafetyDecision.ALLOW
        assert result.processing_time_ms == 150.0

    def test_quality_only_result(self):
        """Test result with only quality evaluation."""
        quality = QualityResult(
            overall_score=80.0,
            dimension_results={},
        )
        result = EvaluationResult(
            quality=quality,
            safety=None,
        )
        assert result.quality is not None
        assert result.safety is None

    def test_to_dict(self):
        """Test serialization."""
        quality = QualityResult(
            overall_score=75.0,
            dimension_results={},
        )
        result = EvaluationResult(
            quality=quality,
            safety=None,
            processing_time_ms=100.0,
        )
        data = result.to_dict()
        assert data["quality"]["overall_score"] == 75.0
        assert data["safety"] is None
        assert data["processing_time_ms"] == 100.0


# -----------------------------------------------------------------------------
# EvaluationConfig Tests
# -----------------------------------------------------------------------------


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EvaluationConfig()
        assert len(config.enabled_quality_dimensions) == 4
        assert len(config.enabled_safety_dimensions) == 4
        assert config.cache_embeddings is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EvaluationConfig(
            enabled_quality_dimensions=frozenset(
                [QualityDimension.PROMPT_ADHERENCE]
            ),
            enabled_safety_dimensions=frozenset([SafetyDimension.COPYRIGHT]),
            cache_embeddings=False,
        )
        assert len(config.enabled_quality_dimensions) == 1
        assert len(config.enabled_safety_dimensions) == 1
        assert config.cache_embeddings is False

    def test_quality_weights_sum(self):
        """Test that default quality weights sum to 1.0."""
        config = EvaluationConfig()
        total = sum(config.quality_weights.values())
        assert abs(total - 1.0) < 0.001


# -----------------------------------------------------------------------------
# BaseDimensionEvaluator Tests
# -----------------------------------------------------------------------------


class TestBaseDimensionEvaluator:
    """Tests for BaseDimensionEvaluator."""

    def test_evaluate_mono_audio(self, sample_audio):
        """Test evaluation with mono audio."""
        audio, sr = sample_audio
        evaluator = MockQualityEvaluator()
        result = evaluator.evaluate(audio, sr)
        assert result.score == 0.75
        assert result.dimension == QualityDimension.PROMPT_ADHERENCE

    def test_evaluate_stereo_audio(self, sample_audio):
        """Test that stereo audio is converted to mono."""
        audio, sr = sample_audio
        stereo = np.stack([audio, audio])  # 2 x N
        evaluator = MockQualityEvaluator()
        result = evaluator.evaluate(stereo, sr)
        assert result.score == 0.75

    def test_evaluate_int16_audio(self, sample_audio):
        """Test that int16 audio is converted to float32."""
        audio, sr = sample_audio
        int_audio = (audio * 32767).astype(np.int16)
        evaluator = MockQualityEvaluator()
        # This will normalize the int audio
        result = evaluator.evaluate(int_audio.astype(np.float32), sr)
        assert result.score == 0.75

    def test_evaluate_unnormalized_audio(self, sample_audio):
        """Test that unnormalized audio is normalized."""
        audio, sr = sample_audio
        loud_audio = audio * 10  # Unnormalized
        evaluator = MockQualityEvaluator()
        result = evaluator.evaluate(loud_audio, sr)
        assert result.score == 0.75

    def test_evaluate_with_prompt(self, sample_audio):
        """Test evaluation with prompt."""
        audio, sr = sample_audio
        evaluator = MockQualityEvaluator()
        result = evaluator.evaluate(audio, sr, prompt="Test prompt")
        assert result.score == 0.75

    def test_evaluate_invalid_dimensions(self, sample_audio):
        """Test that 3D audio raises error."""
        audio, sr = sample_audio
        audio_3d = audio.reshape(1, 1, -1)
        evaluator = MockQualityEvaluator()
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            evaluator.evaluate(audio_3d, sr)


# -----------------------------------------------------------------------------
# DimensionRegistry Tests
# -----------------------------------------------------------------------------


class TestDimensionRegistry:
    """Tests for DimensionRegistry."""

    def test_empty_registry(self):
        """Test empty registry."""
        registry = DimensionRegistry()
        assert len(registry) == 0
        assert registry.list_registered() == []

    def test_register_quality_evaluator(self):
        """Test registering quality evaluator."""
        registry = DimensionRegistry()
        evaluator = MockQualityEvaluator()
        registry.register(evaluator)
        assert len(registry) == 1
        assert registry.get(QualityDimension.PROMPT_ADHERENCE) is evaluator

    def test_register_safety_evaluator(self):
        """Test registering safety evaluator."""
        registry = DimensionRegistry()
        evaluator = MockSafetyEvaluator()
        registry.register(evaluator)
        assert len(registry) == 1
        assert registry.get(SafetyDimension.COPYRIGHT) is evaluator

    def test_register_duplicate_raises(self):
        """Test that registering duplicate raises error."""
        registry = DimensionRegistry()
        evaluator1 = MockQualityEvaluator()
        evaluator2 = MockQualityEvaluator()
        registry.register(evaluator1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(evaluator2)

    def test_get_nonexistent_returns_none(self):
        """Test that getting nonexistent evaluator returns None."""
        registry = DimensionRegistry()
        assert registry.get(QualityDimension.PROMPT_ADHERENCE) is None

    def test_get_quality_evaluators(self):
        """Test getting all quality evaluators."""
        registry = DimensionRegistry()
        evaluator = MockQualityEvaluator()
        registry.register(evaluator)
        quality_evals = registry.get_quality_evaluators()
        assert len(quality_evals) == 1
        assert QualityDimension.PROMPT_ADHERENCE in quality_evals

    def test_get_safety_evaluators(self):
        """Test getting all safety evaluators."""
        registry = DimensionRegistry()
        evaluator = MockSafetyEvaluator()
        registry.register(evaluator)
        safety_evals = registry.get_safety_evaluators()
        assert len(safety_evals) == 1
        assert SafetyDimension.COPYRIGHT in safety_evals

    def test_list_registered(self):
        """Test listing registered dimensions."""
        registry = DimensionRegistry()
        registry.register(MockQualityEvaluator())
        registry.register(MockSafetyEvaluator())
        registered = registry.list_registered()
        assert len(registered) == 2
        assert QualityDimension.PROMPT_ADHERENCE in registered
        assert SafetyDimension.COPYRIGHT in registered
