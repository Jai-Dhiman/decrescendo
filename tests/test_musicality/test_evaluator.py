"""Tests for MusicalityEvaluator."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions import (
    DimensionCategory,
    DimensionResult,
    QualityDimension,
)
from decrescendo.musicritic.dimensions.musicality import (
    AudioTooShortError,
    ExpressionAnalyzer,
    MusicalityConfig,
    MusicalityEvaluator,
    TensionAnalyzer,
    TISAnalyzer,
)


class TestMusicalityEvaluatorAttributes:
    """Tests for MusicalityEvaluator class attributes."""

    def test_dimension_attribute(self) -> None:
        """Test that dimension is MUSICALITY."""
        assert MusicalityEvaluator.dimension == QualityDimension.MUSICALITY

    def test_category_attribute(self) -> None:
        """Test that category is QUALITY."""
        assert MusicalityEvaluator.category == DimensionCategory.QUALITY


class TestMusicalityEvaluatorInit:
    """Tests for MusicalityEvaluator initialization."""

    def test_default_config(self) -> None:
        """Test that default config is used when not provided."""
        evaluator = MusicalityEvaluator()
        assert isinstance(evaluator.config, MusicalityConfig)

    def test_custom_config(self) -> None:
        """Test that custom config is used."""
        config = MusicalityConfig(min_audio_duration=5.0)
        evaluator = MusicalityEvaluator(config=config)
        assert evaluator.config.min_audio_duration == 5.0

    def test_lazy_initialization(self) -> None:
        """Test that analyzers are lazily initialized."""
        evaluator = MusicalityEvaluator()

        # Private attributes should be None before first access
        assert evaluator._tis_analyzer is None
        assert evaluator._tension_analyzer is None
        assert evaluator._expression_analyzer is None

        # Accessing properties should initialize
        _ = evaluator.tis_analyzer
        _ = evaluator.tension_analyzer
        _ = evaluator.expression_analyzer

        # Now they should be initialized
        assert evaluator._tis_analyzer is not None
        assert evaluator._tension_analyzer is not None
        assert evaluator._expression_analyzer is not None

    def test_dependency_injection(self) -> None:
        """Test that pre-initialized analyzers are used."""
        tis = TISAnalyzer()
        tension = TensionAnalyzer()
        expression = ExpressionAnalyzer()

        evaluator = MusicalityEvaluator(
            tis_analyzer=tis,
            tension_analyzer=tension,
            expression_analyzer=expression,
        )

        assert evaluator.tis_analyzer is tis
        assert evaluator.tension_analyzer is tension
        assert evaluator.expression_analyzer is expression


class TestMusicalityEvaluatorValidation:
    """Tests for input validation."""

    def test_audio_too_short_raises(
        self, short_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that short audio raises AudioTooShortError."""
        audio, sr = short_audio
        evaluator = MusicalityEvaluator()

        with pytest.raises(AudioTooShortError) as exc_info:
            evaluator.evaluate(audio, sr)

        assert exc_info.value.actual_duration < evaluator.config.min_audio_duration
        assert exc_info.value.required_duration == evaluator.config.min_audio_duration

    def test_custom_min_duration(
        self, short_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that custom min_audio_duration is respected."""
        audio, sr = short_audio
        # Set very short minimum
        config = MusicalityConfig(min_audio_duration=0.5)
        evaluator = MusicalityEvaluator(config=config)

        # Should not raise now
        result = evaluator.evaluate(audio, sr)
        assert isinstance(result, DimensionResult)


class TestMusicalityEvaluatorEvaluation:
    """Tests for evaluation functionality."""

    def test_evaluate_returns_result(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that evaluate returns a DimensionResult."""
        audio, sr = tonal_audio
        evaluator = MusicalityEvaluator()

        result = evaluator.evaluate(audio, sr)

        assert isinstance(result, DimensionResult)

    def test_evaluate_score_range(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that score is in valid range."""
        audio, sr = tonal_audio
        evaluator = MusicalityEvaluator()

        result = evaluator.evaluate(audio, sr)

        assert 0.0 <= result.score <= 1.0

    def test_evaluate_confidence_range(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that confidence is in valid range."""
        audio, sr = tonal_audio
        evaluator = MusicalityEvaluator()

        result = evaluator.evaluate(audio, sr)

        assert 0.0 <= result.confidence <= 1.0

    def test_evaluate_sub_scores(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that sub_scores contains expected keys."""
        audio, sr = tonal_audio
        evaluator = MusicalityEvaluator()

        result = evaluator.evaluate(audio, sr)

        assert "tis" in result.sub_scores
        assert "tension" in result.sub_scores
        assert "expression" in result.sub_scores

        # All sub-scores should be in valid range
        for score in result.sub_scores.values():
            assert 0.0 <= score <= 1.0

    def test_evaluate_metadata(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that metadata contains expected keys."""
        audio, sr = tonal_audio
        evaluator = MusicalityEvaluator()

        result = evaluator.evaluate(audio, sr)

        assert "musicality_level" in result.metadata
        assert "cloud_diameter" in result.metadata
        assert "cloud_momentum" in result.metadata
        assert "tensile_strain" in result.metadata
        assert "tonal_center" in result.metadata
        assert "resolution_count" in result.metadata
        assert "arc_quality" in result.metadata
        assert "dynamic_range_db" in result.metadata
        assert "dynamic_variation" in result.metadata

    def test_evaluate_explanation(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that explanation is generated."""
        audio, sr = tonal_audio
        evaluator = MusicalityEvaluator()

        result = evaluator.evaluate(audio, sr)

        assert result.explanation
        assert len(result.explanation) > 0
        assert "musicality" in result.explanation.lower() or "score" in result.explanation.lower()


class TestMusicalityEvaluatorClassification:
    """Tests for musicality level classification."""

    def test_classify_excellent(self) -> None:
        """Test classification of excellent scores."""
        evaluator = MusicalityEvaluator()

        level = evaluator._classify_musicality(0.85)
        assert level == "excellent"

    def test_classify_good(self) -> None:
        """Test classification of good scores."""
        evaluator = MusicalityEvaluator()

        level = evaluator._classify_musicality(0.70)
        assert level == "good"

    def test_classify_moderate(self) -> None:
        """Test classification of moderate scores."""
        evaluator = MusicalityEvaluator()

        level = evaluator._classify_musicality(0.50)
        assert level == "moderate"

    def test_classify_poor(self) -> None:
        """Test classification of poor scores."""
        evaluator = MusicalityEvaluator()

        level = evaluator._classify_musicality(0.30)
        assert level == "poor"


class TestMusicalityEvaluatorWeighting:
    """Tests for sub-score weighting."""

    def test_weighted_combination(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that final score is weighted combination of sub-scores."""
        audio, sr = tonal_audio
        evaluator = MusicalityEvaluator()

        result = evaluator.evaluate(audio, sr)

        # Compute expected weighted score
        expected = (
            evaluator.config.tis_weight * result.sub_scores["tis"]
            + evaluator.config.tension_weight * result.sub_scores["tension"]
            + evaluator.config.expression_weight * result.sub_scores["expression"]
        )

        # Should be approximately equal (might differ slightly due to clipping)
        assert abs(result.score - expected) < 0.01

    def test_custom_weights(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that custom weights are used."""
        audio, sr = tonal_audio

        # Heavy weight on expression
        config = MusicalityConfig(
            tis_weight=0.1,
            tension_weight=0.1,
            expression_weight=0.8,
        )
        evaluator = MusicalityEvaluator(config=config)

        result = evaluator.evaluate(audio, sr)

        # Expression should dominate the score
        expected = (
            0.1 * result.sub_scores["tis"]
            + 0.1 * result.sub_scores["tension"]
            + 0.8 * result.sub_scores["expression"]
        )

        assert abs(result.score - expected) < 0.01


class TestMusicalityEvaluatorDifferentInputs:
    """Tests for different input types."""

    def test_dynamic_audio(
        self, dynamic_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test evaluation of dynamic audio."""
        audio, sr = dynamic_audio
        evaluator = MusicalityEvaluator()

        result = evaluator.evaluate(audio, sr)

        assert isinstance(result, DimensionResult)
        assert 0.0 <= result.score <= 1.0

    def test_tension_resolution_audio(
        self, tension_resolution_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test evaluation of tension-resolution audio."""
        audio, sr = tension_resolution_audio
        evaluator = MusicalityEvaluator()

        result = evaluator.evaluate(audio, sr)

        assert isinstance(result, DimensionResult)
        assert 0.0 <= result.score <= 1.0

    def test_atonal_audio(
        self, atonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test evaluation of atonal audio."""
        audio, sr = atonal_audio
        evaluator = MusicalityEvaluator()

        result = evaluator.evaluate(audio, sr)

        assert isinstance(result, DimensionResult)
        assert 0.0 <= result.score <= 1.0

    def test_stereo_converted_to_mono(self, sample_rate: int) -> None:
        """Test that stereo audio is handled correctly."""
        # Create stereo audio (2, n_samples) - channels first
        duration = 4.0
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, dtype=np.float32)
        mono = 0.5 * np.sin(2 * np.pi * 440 * t)
        stereo = np.vstack([mono, mono])  # (2, n_samples)

        evaluator = MusicalityEvaluator()
        result = evaluator.evaluate(stereo, sample_rate)

        assert isinstance(result, DimensionResult)
