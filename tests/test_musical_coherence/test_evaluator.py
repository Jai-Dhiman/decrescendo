"""Tests for MusicalCoherenceEvaluator."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.base import DimensionResult, QualityDimension
from decrescendo.musicritic.dimensions.musical_coherence.config import (
    MusicalCoherenceConfig,
)
from decrescendo.musicritic.dimensions.musical_coherence.evaluator import (
    MusicalCoherenceEvaluator,
)
from decrescendo.musicritic.dimensions.musical_coherence.exceptions import (
    AudioTooShortError,
)


class TestMusicalCoherenceEvaluator:
    """Test MusicalCoherenceEvaluator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        evaluator = MusicalCoherenceEvaluator()
        assert evaluator.config is not None
        assert evaluator.dimension == QualityDimension.MUSICAL_COHERENCE

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = MusicalCoherenceConfig(structure_weight=0.4)
        evaluator = MusicalCoherenceEvaluator(config=config)
        assert evaluator.config.structure_weight == 0.4

    def test_lazy_initialization_structure(self):
        """Test lazy initialization of structure analyzer."""
        evaluator = MusicalCoherenceEvaluator()
        assert evaluator._structure_analyzer is None
        analyzer = evaluator.structure_analyzer
        assert analyzer is not None
        assert evaluator._structure_analyzer is not None

    def test_lazy_initialization_harmony(self):
        """Test lazy initialization of harmony analyzer."""
        evaluator = MusicalCoherenceEvaluator()
        assert evaluator._harmony_analyzer is None
        analyzer = evaluator.harmony_analyzer
        assert analyzer is not None
        assert evaluator._harmony_analyzer is not None

    def test_lazy_initialization_rhythm(self):
        """Test lazy initialization of rhythm analyzer."""
        evaluator = MusicalCoherenceEvaluator()
        assert evaluator._rhythm_analyzer is None
        analyzer = evaluator.rhythm_analyzer
        assert analyzer is not None
        assert evaluator._rhythm_analyzer is not None

    def test_lazy_initialization_melody(self):
        """Test lazy initialization of melody analyzer."""
        evaluator = MusicalCoherenceEvaluator()
        assert evaluator._melody_analyzer is None
        analyzer = evaluator.melody_analyzer
        assert analyzer is not None
        assert evaluator._melody_analyzer is not None

    def test_evaluate_sample_audio(self, sample_audio):
        """Test evaluation on sample audio."""
        audio, sample_rate = sample_audio
        evaluator = MusicalCoherenceEvaluator()

        result = evaluator.evaluate(audio, sample_rate)

        assert isinstance(result, DimensionResult)
        assert result.dimension == QualityDimension.MUSICAL_COHERENCE
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_evaluate_returns_sub_scores(self, sample_audio):
        """Test that evaluation returns all sub-scores."""
        audio, sample_rate = sample_audio
        evaluator = MusicalCoherenceEvaluator()

        result = evaluator.evaluate(audio, sample_rate)

        assert "structure" in result.sub_scores
        assert "harmony" in result.sub_scores
        assert "rhythm" in result.sub_scores
        assert "melody" in result.sub_scores

        for score in result.sub_scores.values():
            assert 0.0 <= score <= 1.0

    def test_evaluate_returns_metadata(self, sample_audio):
        """Test that evaluation returns expected metadata."""
        audio, sample_rate = sample_audio
        evaluator = MusicalCoherenceEvaluator()

        result = evaluator.evaluate(audio, sample_rate)

        assert "coherence_level" in result.metadata
        assert "detected_key" in result.metadata
        assert "tempo_bpm" in result.metadata
        assert "section_count" in result.metadata
        assert "voiced_ratio" in result.metadata

    def test_evaluate_returns_explanation(self, sample_audio):
        """Test that evaluation returns explanation."""
        audio, sample_rate = sample_audio
        evaluator = MusicalCoherenceEvaluator()

        result = evaluator.evaluate(audio, sample_rate)

        assert result.explanation != ""
        assert "coherence" in result.explanation.lower()

    def test_evaluate_structured_audio(self, structured_audio):
        """Test evaluation on structured audio."""
        audio, sample_rate = structured_audio
        evaluator = MusicalCoherenceEvaluator()

        result = evaluator.evaluate(audio, sample_rate)

        assert isinstance(result, DimensionResult)
        # Structured audio should have some coherence

    def test_evaluate_chord_progression(self, chord_progression_audio):
        """Test evaluation on chord progression audio."""
        audio, sample_rate = chord_progression_audio
        evaluator = MusicalCoherenceEvaluator()

        result = evaluator.evaluate(audio, sample_rate)

        assert isinstance(result, DimensionResult)
        # Chord progression should have decent harmony score
        assert result.sub_scores["harmony"] >= 0.0

    def test_evaluate_rhythmic_audio(self, rhythmic_audio):
        """Test evaluation on rhythmic audio."""
        audio, sample_rate = rhythmic_audio
        evaluator = MusicalCoherenceEvaluator()

        result = evaluator.evaluate(audio, sample_rate)

        assert isinstance(result, DimensionResult)
        # Rhythmic audio should have some rhythm score
        assert result.sub_scores["rhythm"] >= 0.0

    def test_evaluate_melodic_audio(self, melodic_audio):
        """Test evaluation on melodic audio."""
        audio, sample_rate = melodic_audio
        evaluator = MusicalCoherenceEvaluator()

        result = evaluator.evaluate(audio, sample_rate)

        assert isinstance(result, DimensionResult)
        # Melodic audio should have some melody score
        assert result.sub_scores["melody"] >= 0.0

    def test_evaluate_short_audio_raises_error(self, short_audio):
        """Test that short audio raises AudioTooShortError."""
        audio, sample_rate = short_audio
        config = MusicalCoherenceConfig(min_audio_duration=3.0)
        evaluator = MusicalCoherenceEvaluator(config=config)

        with pytest.raises(AudioTooShortError) as exc_info:
            evaluator.evaluate(audio, sample_rate)

        assert exc_info.value.duration < exc_info.value.min_duration

    def test_evaluate_silence(self, silence):
        """Test evaluation on silence."""
        audio, sample_rate = silence
        evaluator = MusicalCoherenceEvaluator()

        result = evaluator.evaluate(audio, sample_rate)

        assert isinstance(result, DimensionResult)
        # Silence should have low coherence
        assert result.score < 0.5

    def test_evaluate_white_noise(self, white_noise):
        """Test evaluation on white noise."""
        audio, sample_rate = white_noise
        evaluator = MusicalCoherenceEvaluator()

        result = evaluator.evaluate(audio, sample_rate)

        assert isinstance(result, DimensionResult)
        # Noise should have low coherence
        assert result.score < 0.7

    def test_classify_coherence_excellent(self):
        """Test coherence classification for excellent."""
        evaluator = MusicalCoherenceEvaluator()

        level = evaluator._classify_coherence(0.85)

        assert level == "excellent"

    def test_classify_coherence_good(self):
        """Test coherence classification for good."""
        evaluator = MusicalCoherenceEvaluator()

        level = evaluator._classify_coherence(0.70)

        assert level == "good"

    def test_classify_coherence_moderate(self):
        """Test coherence classification for moderate."""
        evaluator = MusicalCoherenceEvaluator()

        level = evaluator._classify_coherence(0.50)

        assert level == "moderate"

    def test_classify_coherence_poor(self):
        """Test coherence classification for poor."""
        evaluator = MusicalCoherenceEvaluator()

        level = evaluator._classify_coherence(0.30)

        assert level == "poor"

    def test_compute_confidence_high(self):
        """Test confidence computation with high agreement."""
        evaluator = MusicalCoherenceEvaluator()

        # Long duration, similar scores = high confidence
        confidence = evaluator._compute_confidence(
            duration=30.0,
            structure_score=0.7,
            harmony_score=0.7,
            rhythm_score=0.7,
            melody_score=0.7,
        )

        assert confidence > 0.8

    def test_compute_confidence_low(self):
        """Test confidence computation with low agreement."""
        evaluator = MusicalCoherenceEvaluator()

        # Short duration, varied scores = lower confidence
        confidence = evaluator._compute_confidence(
            duration=2.0,
            structure_score=0.1,
            harmony_score=0.9,
            rhythm_score=0.3,
            melody_score=0.7,
        )

        # Still above minimum (0.5)
        assert 0.5 <= confidence < 0.8

    def test_compute_confidence_bounds(self):
        """Test confidence is always in valid range."""
        evaluator = MusicalCoherenceEvaluator()

        test_cases = [
            (1.0, 0.0, 0.0, 0.0, 0.0),  # Very short
            (60.0, 1.0, 1.0, 1.0, 1.0),  # Long, perfect scores
            (10.0, 0.5, 0.5, 0.5, 0.5),  # Medium
        ]

        for duration, s, h, r, m in test_cases:
            confidence = evaluator._compute_confidence(duration, s, h, r, m)
            assert 0.5 <= confidence <= 1.0

    def test_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        config = MusicalCoherenceConfig()
        total = (
            config.structure_weight
            + config.harmony_weight
            + config.rhythm_weight
            + config.melody_weight
        )
        assert abs(total - 1.0) < 0.001

    def test_custom_weights(self, sample_audio):
        """Test evaluation with custom weights."""
        audio, sample_rate = sample_audio
        config = MusicalCoherenceConfig(
            structure_weight=0.10,
            harmony_weight=0.40,
            rhythm_weight=0.40,
            melody_weight=0.10,
        )
        evaluator = MusicalCoherenceEvaluator(config=config)

        result = evaluator.evaluate(audio, sample_rate)

        assert isinstance(result, DimensionResult)
        # Result should still be valid
        assert 0.0 <= result.score <= 1.0

    def test_stereo_audio_converted(self, sample_rate):
        """Test that stereo audio is converted to mono."""
        duration = 4.0
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, dtype=np.float32)

        # Create stereo audio (2 x n_samples)
        left = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = 0.5 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
        stereo_audio = np.stack([left, right])

        evaluator = MusicalCoherenceEvaluator()
        result = evaluator.evaluate(stereo_audio, sample_rate)

        assert isinstance(result, DimensionResult)
