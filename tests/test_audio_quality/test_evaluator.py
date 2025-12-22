"""Tests for AudioQualityEvaluator."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.base import (
    DimensionCategory,
    DimensionResult,
    QualityDimension,
)
from decrescendo.musicritic.dimensions.audio_quality import (
    AudioQualityConfig,
    AudioQualityEvaluator,
)
from decrescendo.musicritic.dimensions.audio_quality.exceptions import (
    AudioTooShortError,
)


class TestAudioQualityEvaluatorAttributes:
    """Tests for evaluator class attributes."""

    def test_dimension_attribute(self):
        """Test that dimension is AUDIO_QUALITY."""
        assert AudioQualityEvaluator.dimension == QualityDimension.AUDIO_QUALITY

    def test_category_attribute(self):
        """Test that category is QUALITY."""
        assert AudioQualityEvaluator.category == DimensionCategory.QUALITY


class TestAudioQualityEvaluatorInit:
    """Tests for evaluator initialization."""

    def test_default_init(self):
        """Test initialization with defaults."""
        evaluator = AudioQualityEvaluator()
        assert evaluator.config is not None
        assert evaluator._artifact_detector is None
        assert evaluator._loudness_analyzer is None
        assert evaluator._perceptual_analyzer is None

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = AudioQualityConfig(artifact_weight=0.5)
        evaluator = AudioQualityEvaluator(config=config)
        assert evaluator.config.artifact_weight == 0.5

    def test_lazy_initialization(self):
        """Test that analyzers are lazily initialized."""
        evaluator = AudioQualityEvaluator()

        # Access properties to trigger initialization
        _ = evaluator.artifact_detector
        _ = evaluator.loudness_analyzer
        _ = evaluator.perceptual_analyzer

        assert evaluator._artifact_detector is not None
        assert evaluator._loudness_analyzer is not None
        assert evaluator._perceptual_analyzer is not None


class TestAudioQualityEvaluatorValidation:
    """Tests for input validation."""

    def test_short_audio_raises(self):
        """Test that too-short audio raises AudioTooShortError."""
        evaluator = AudioQualityEvaluator()
        sr = 44100
        audio = np.zeros(int(sr * 0.3), dtype=np.float32)  # 0.3 seconds

        with pytest.raises(AudioTooShortError):
            evaluator.evaluate(audio, sr)


class TestAudioQualityEvaluatorEvaluation:
    """Tests for evaluation functionality."""

    def test_evaluate_returns_result(self, sample_audio_44k):
        """Test that evaluate returns a DimensionResult."""
        audio, sr = sample_audio_44k
        evaluator = AudioQualityEvaluator()
        result = evaluator.evaluate(audio, sr)

        assert isinstance(result, DimensionResult)
        assert result.dimension == QualityDimension.AUDIO_QUALITY

    def test_evaluate_score_in_range(self, sample_audio_44k):
        """Test that score is in valid range."""
        audio, sr = sample_audio_44k
        evaluator = AudioQualityEvaluator()
        result = evaluator.evaluate(audio, sr)

        assert 0.0 <= result.score <= 1.0

    def test_evaluate_sub_scores(self, sample_audio_44k):
        """Test that sub_scores are present and valid."""
        audio, sr = sample_audio_44k
        evaluator = AudioQualityEvaluator()
        result = evaluator.evaluate(audio, sr)

        assert "artifacts" in result.sub_scores
        assert "loudness" in result.sub_scores
        assert "perceptual" in result.sub_scores

        for score in result.sub_scores.values():
            assert 0.0 <= score <= 1.0

    def test_evaluate_metadata(self, sample_audio_44k):
        """Test that metadata contains expected fields."""
        audio, sr = sample_audio_44k
        evaluator = AudioQualityEvaluator()
        result = evaluator.evaluate(audio, sr)

        assert "quality_level" in result.metadata
        assert "streaming_compliant" in result.metadata
        assert "integrated_lufs" in result.metadata
        assert "true_peak_dbtp" in result.metadata
        assert "click_count" in result.metadata

    def test_evaluate_confidence(self, sample_audio_44k):
        """Test that confidence is calculated."""
        audio, sr = sample_audio_44k
        evaluator = AudioQualityEvaluator()
        result = evaluator.evaluate(audio, sr)

        assert 0.5 <= result.confidence <= 1.0

    def test_evaluate_explanation(self, sample_audio_44k):
        """Test that explanation is generated."""
        audio, sr = sample_audio_44k
        evaluator = AudioQualityEvaluator()
        result = evaluator.evaluate(audio, sr)

        assert result.explanation is not None
        assert len(result.explanation) > 0

    def test_evaluate_stereo_handled(self, sample_stereo_audio):
        """Test that stereo audio is handled by base class."""
        audio, sr = sample_stereo_audio
        evaluator = AudioQualityEvaluator()
        result = evaluator.evaluate(audio, sr)

        # Should successfully evaluate stereo (base class converts to mono)
        assert isinstance(result, DimensionResult)


class TestAudioQualityEvaluatorClassification:
    """Tests for quality level classification."""

    def test_classify_excellent(self):
        """Test excellent classification."""
        evaluator = AudioQualityEvaluator()
        level = evaluator._classify_quality(0.90)
        assert level == "excellent"

    def test_classify_good(self):
        """Test good classification."""
        evaluator = AudioQualityEvaluator()
        level = evaluator._classify_quality(0.75)
        assert level == "good"

    def test_classify_acceptable(self):
        """Test acceptable classification."""
        evaluator = AudioQualityEvaluator()
        level = evaluator._classify_quality(0.55)
        assert level == "acceptable"

    def test_classify_poor(self):
        """Test poor classification."""
        evaluator = AudioQualityEvaluator()
        level = evaluator._classify_quality(0.30)
        assert level == "poor"


class TestAudioQualityEvaluatorScaledScore:
    """Tests for scaled score (0-100)."""

    def test_scaled_score(self, sample_audio_44k):
        """Test that scaled_score property works."""
        audio, sr = sample_audio_44k
        evaluator = AudioQualityEvaluator()
        result = evaluator.evaluate(audio, sr)

        # scaled_score should be score * 100
        assert abs(result.scaled_score - result.score * 100) < 0.01


class TestAudioQualityComparison:
    """Tests comparing different audio quality levels."""

    def test_clean_vs_clipped(self, sample_audio_44k, clipped_audio):
        """Test that clean audio scores higher than clipped."""
        evaluator = AudioQualityEvaluator()

        clean, sr = sample_audio_44k
        clipped, _ = clipped_audio

        clean_result = evaluator.evaluate(clean, sr)
        clipped_result = evaluator.evaluate(clipped, sr)

        assert clean_result.score >= clipped_result.score

    def test_normal_vs_loud(self, sample_audio_44k, loud_audio):
        """Test that properly leveled audio scores higher than too-loud audio."""
        evaluator = AudioQualityEvaluator()

        normal, sr = sample_audio_44k
        loud, _ = loud_audio

        normal_result = evaluator.evaluate(normal, sr)
        loud_result = evaluator.evaluate(loud, sr)

        # Normal audio should have better streaming compliance
        assert normal_result.metadata["true_peak_compliant"]
        # Loud audio should fail True Peak
        assert not loud_result.metadata["true_peak_compliant"]


class TestAudioQualityEdgeCases:
    """Tests for edge cases."""

    def test_silence(self, silence):
        """Test evaluation of silence."""
        audio, sr = silence
        evaluator = AudioQualityEvaluator()
        result = evaluator.evaluate(audio, sr)

        # Should evaluate without error
        assert isinstance(result, DimensionResult)

    def test_white_noise(self, white_noise):
        """Test evaluation of white noise."""
        audio, sr = white_noise
        evaluator = AudioQualityEvaluator()
        result = evaluator.evaluate(audio, sr)

        # Should evaluate without error
        assert isinstance(result, DimensionResult)
