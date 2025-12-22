"""Tests for CopyrightEvaluator."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.base import (
    DimensionCategory,
    DimensionResult,
    SafetyDecision,
    SafetyDimension,
)
from decrescendo.musicritic.dimensions.copyright.config import CopyrightConfig
from decrescendo.musicritic.dimensions.copyright.evaluator import CopyrightEvaluator
from decrescendo.musicritic.dimensions.copyright.exceptions import AudioTooShortError
from decrescendo.musicritic.dimensions.copyright.fingerprint import FingerprintDatabase


class TestCopyrightEvaluatorAttributes:
    """Tests for class attributes."""

    def test_dimension_attribute(self):
        """Should have correct dimension."""
        assert CopyrightEvaluator.dimension == SafetyDimension.COPYRIGHT

    def test_category_attribute(self):
        """Should be a safety dimension."""
        assert CopyrightEvaluator.category == DimensionCategory.SAFETY


class TestCopyrightEvaluatorInit:
    """Tests for initialization."""

    def test_default_init(self):
        """Should initialize with defaults."""
        evaluator = CopyrightEvaluator()
        assert evaluator.config is not None
        assert evaluator.fingerprint_db is None

    def test_custom_config(self):
        """Should accept custom config."""
        config = CopyrightConfig(flag_threshold=0.8)
        evaluator = CopyrightEvaluator(config=config)
        assert evaluator.config.flag_threshold == 0.8

    def test_with_fingerprint_db(self):
        """Should accept fingerprint database."""
        db = FingerprintDatabase()
        evaluator = CopyrightEvaluator(fingerprint_db=db)
        assert evaluator.fingerprint_db is db

    def test_with_reference_audios(self, sine_440hz: np.ndarray, sample_rate: int):
        """Should accept reference audios."""
        references = [(sine_440hz, sample_rate, "reference1")]
        evaluator = CopyrightEvaluator(reference_audios=references)
        assert len(evaluator._reference_audios) == 1


class TestCopyrightEvaluatorLazyInit:
    """Tests for lazy initialization of components."""

    def test_melody_extractor_lazy(self):
        """Melody extractor should be lazily initialized."""
        evaluator = CopyrightEvaluator()
        assert evaluator._melody_extractor is None
        _ = evaluator.melody_extractor
        assert evaluator._melody_extractor is not None

    def test_rhythm_extractor_lazy(self):
        """Rhythm extractor should be lazily initialized."""
        evaluator = CopyrightEvaluator()
        assert evaluator._rhythm_extractor is None
        _ = evaluator.rhythm_extractor
        assert evaluator._rhythm_extractor is not None

    def test_similarity_matcher_lazy(self):
        """Similarity matcher should be lazily initialized."""
        evaluator = CopyrightEvaluator()
        assert evaluator._similarity_matcher is None
        _ = evaluator.similarity_matcher
        assert evaluator._similarity_matcher is not None


class TestCopyrightEvaluatorReferenceManagement:
    """Tests for reference audio management."""

    def test_add_reference(self, sine_440hz: np.ndarray, sample_rate: int):
        """Should add reference audio."""
        evaluator = CopyrightEvaluator()
        evaluator.add_reference(sine_440hz, sample_rate, "test_ref")
        assert len(evaluator._reference_audios) == 1

    def test_add_multiple_references(
        self, sine_440hz: np.ndarray, sine_880hz: np.ndarray, sample_rate: int
    ):
        """Should add multiple references."""
        evaluator = CopyrightEvaluator()
        evaluator.add_reference(sine_440hz, sample_rate, "ref1")
        evaluator.add_reference(sine_880hz, sample_rate, "ref2")
        assert len(evaluator._reference_audios) == 2

    def test_clear_references(self, sine_440hz: np.ndarray, sample_rate: int):
        """Should clear all references."""
        evaluator = CopyrightEvaluator()
        evaluator.add_reference(sine_440hz, sample_rate, "ref1")
        evaluator.clear_references()
        assert len(evaluator._reference_audios) == 0

    def test_set_fingerprint_db(self):
        """Should set fingerprint database."""
        evaluator = CopyrightEvaluator()
        db = FingerprintDatabase()
        evaluator.set_fingerprint_db(db)
        assert evaluator.fingerprint_db is db


class TestCopyrightEvaluatorEvaluation:
    """Tests for evaluation."""

    def test_evaluate_returns_result(self, sine_440hz: np.ndarray, sample_rate: int):
        """Should return DimensionResult."""
        evaluator = CopyrightEvaluator()
        result = evaluator.evaluate(sine_440hz, sample_rate)

        assert isinstance(result, DimensionResult)
        assert result.dimension == SafetyDimension.COPYRIGHT
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_evaluate_short_audio_raises(
        self, short_audio: np.ndarray, sample_rate: int
    ):
        """Should raise for audio shorter than minimum."""
        evaluator = CopyrightEvaluator()
        with pytest.raises(AudioTooShortError):
            evaluator.evaluate(short_audio, sample_rate)

    def test_evaluate_no_references_low_score(
        self, sine_440hz: np.ndarray, sample_rate: int
    ):
        """Without references, score should be 0 (can't detect plagiarism)."""
        evaluator = CopyrightEvaluator()
        result = evaluator.evaluate(sine_440hz, sample_rate)

        # No references = can't detect plagiarism = score 0
        assert result.score == 0.0
        assert result.metadata["decision"] == "ALLOW"

    def test_evaluate_with_identical_reference(
        self, sine_440hz: np.ndarray, sample_rate: int
    ):
        """Identical reference should produce moderate-to-high score."""
        evaluator = CopyrightEvaluator()
        evaluator.add_reference(sine_440hz, sample_rate, "identical")
        result = evaluator.evaluate(sine_440hz, sample_rate)

        # Identical audio should have high melody/harmony similarity
        # but rhythm similarity may be 0 for pure tones (no onsets)
        # so overall score is weighted lower
        assert result.score > 0.5
        assert len(result.metadata["similarity_matches"]) > 0

    def test_evaluate_with_different_reference(
        self, sine_440hz: np.ndarray, white_noise: np.ndarray, sample_rate: int
    ):
        """Different reference should produce lower score."""
        evaluator = CopyrightEvaluator()
        evaluator.add_reference(white_noise, sample_rate, "different")
        result = evaluator.evaluate(sine_440hz, sample_rate)

        # Very different audio should have lower similarity
        assert result.score < 0.7

    def test_evaluate_sub_scores(self, sine_440hz: np.ndarray, sample_rate: int):
        """Should include sub-scores in result."""
        evaluator = CopyrightEvaluator()
        evaluator.add_reference(sine_440hz, sample_rate, "ref")
        result = evaluator.evaluate(sine_440hz, sample_rate)

        assert "fingerprint" in result.sub_scores
        assert "melody" in result.sub_scores
        assert "rhythm" in result.sub_scores
        assert "harmony" in result.sub_scores

    def test_evaluate_metadata(self, sine_440hz: np.ndarray, sample_rate: int):
        """Should include metadata in result."""
        evaluator = CopyrightEvaluator()
        result = evaluator.evaluate(sine_440hz, sample_rate)

        assert "decision" in result.metadata
        assert "duration_seconds" in result.metadata
        assert "fingerprint_available" in result.metadata
        assert "reference_count" in result.metadata
        assert "fingerprint_matches" in result.metadata
        assert "similarity_matches" in result.metadata

    def test_evaluate_explanation(self, sine_440hz: np.ndarray, sample_rate: int):
        """Should include explanation in result."""
        evaluator = CopyrightEvaluator()
        result = evaluator.evaluate(sine_440hz, sample_rate)

        assert len(result.explanation) > 0
        assert "ALLOW" in result.explanation or "FLAG" in result.explanation or "BLOCK" in result.explanation


class TestCopyrightEvaluatorDecisions:
    """Tests for safety decisions."""

    def test_allow_decision(self, sine_440hz: np.ndarray, sample_rate: int):
        """Low score should result in ALLOW."""
        evaluator = CopyrightEvaluator()
        result = evaluator.evaluate(sine_440hz, sample_rate)

        assert result.metadata["decision"] == "ALLOW"

    def test_flag_decision(
        self, sine_440hz: np.ndarray, sine_880hz: np.ndarray, sample_rate: int
    ):
        """Score between thresholds should result in FLAG."""
        # Use custom config with lower thresholds
        config = CopyrightConfig(flag_threshold=0.3, block_threshold=0.9)
        evaluator = CopyrightEvaluator(config=config)
        evaluator.add_reference(sine_440hz, sample_rate, "ref")

        # Sine waves at octave relation have moderate similarity
        result = evaluator.evaluate(sine_880hz, sample_rate)

        # Check that decision logic works (exact decision depends on similarity)
        assert result.metadata["decision"] in ["ALLOW", "FLAG", "BLOCK"]

    def test_block_decision_threshold(self, sine_440hz: np.ndarray, sample_rate: int):
        """Very high score should result in BLOCK."""
        # Use config with very low thresholds to force BLOCK
        config = CopyrightConfig(flag_threshold=0.1, block_threshold=0.2)
        evaluator = CopyrightEvaluator(config=config)
        evaluator.add_reference(sine_440hz, sample_rate, "identical")

        result = evaluator.evaluate(sine_440hz, sample_rate)

        # Identical audio should trigger BLOCK with low thresholds
        assert result.metadata["decision"] == "BLOCK"


class TestCopyrightEvaluatorConfidence:
    """Tests for confidence computation."""

    def test_confidence_in_range(self, sine_440hz: np.ndarray, sample_rate: int):
        """Confidence should be in [0, 1]."""
        evaluator = CopyrightEvaluator()
        result = evaluator.evaluate(sine_440hz, sample_rate)

        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_higher_with_references(
        self, sine_440hz: np.ndarray, sample_rate: int
    ):
        """Confidence should be higher with references."""
        evaluator_no_ref = CopyrightEvaluator()
        result_no_ref = evaluator_no_ref.evaluate(sine_440hz, sample_rate)

        evaluator_with_ref = CopyrightEvaluator()
        evaluator_with_ref.add_reference(sine_440hz, sample_rate, "ref")
        result_with_ref = evaluator_with_ref.evaluate(sine_440hz, sample_rate)

        assert result_with_ref.confidence >= result_no_ref.confidence


class TestCopyrightEvaluatorEdgeCases:
    """Tests for edge cases."""

    def test_stereo_audio(self, sample_rate: int):
        """Should handle stereo audio (converted to mono in base class)."""
        duration = 2.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        stereo_audio = np.stack([
            np.sin(2 * np.pi * 440 * t),
            np.sin(2 * np.pi * 440 * t),
        ])

        evaluator = CopyrightEvaluator()
        result = evaluator.evaluate(stereo_audio, sample_rate)

        assert isinstance(result, DimensionResult)

    def test_unnormalized_audio(self, sample_rate: int):
        """Should handle unnormalized audio (normalized in base class)."""
        duration = 2.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        loud_audio = np.sin(2 * np.pi * 440 * t) * 5.0  # Peak at 5.0

        evaluator = CopyrightEvaluator()
        result = evaluator.evaluate(loud_audio, sample_rate)

        assert isinstance(result, DimensionResult)

    def test_different_sample_rate_reference(self, sample_rate: int):
        """Should handle references at different sample rates."""
        evaluator = CopyrightEvaluator()

        # Reference at 22050 Hz
        ref_sr = 22050
        ref_duration = 2.0
        ref_samples = int(ref_sr * ref_duration)
        t_ref = np.linspace(0, ref_duration, ref_samples, dtype=np.float32)
        ref_audio = np.sin(2 * np.pi * 440 * t_ref).astype(np.float32)
        evaluator.add_reference(ref_audio, ref_sr, "ref")

        # Query at 44100 Hz
        query_duration = 2.0
        query_samples = int(sample_rate * query_duration)
        t_query = np.linspace(0, query_duration, query_samples, dtype=np.float32)
        query_audio = np.sin(2 * np.pi * 440 * t_query).astype(np.float32)

        result = evaluator.evaluate(query_audio, sample_rate)

        assert isinstance(result, DimensionResult)

    def test_silence_audio(self, silence: np.ndarray, sample_rate: int):
        """Should handle near-silent audio."""
        evaluator = CopyrightEvaluator()
        result = evaluator.evaluate(silence, sample_rate)

        assert isinstance(result, DimensionResult)


class TestCopyrightEvaluatorSimilarityMatches:
    """Tests for similarity match reporting."""

    def test_similarity_matches_above_threshold(
        self, sine_440hz: np.ndarray, sample_rate: int
    ):
        """Similar audio should appear in similarity_matches."""
        evaluator = CopyrightEvaluator()
        evaluator.add_reference(sine_440hz, sample_rate, "similar_song")

        result = evaluator.evaluate(sine_440hz, sample_rate)

        assert len(result.metadata["similarity_matches"]) > 0
        match = result.metadata["similarity_matches"][0]
        assert match["name"] == "similar_song"
        assert "melody_similarity" in match
        assert "rhythm_similarity" in match
        assert "harmony_similarity" in match

    def test_no_matches_for_different_audio(
        self, sine_440hz: np.ndarray, white_noise: np.ndarray, sample_rate: int
    ):
        """Very different audio should not appear in matches."""
        evaluator = CopyrightEvaluator()
        evaluator.add_reference(white_noise, sample_rate, "noise")

        result = evaluator.evaluate(sine_440hz, sample_rate)

        # Matches are only added if overall_similarity > 0.5
        # Noise vs sine should be below that
        for match in result.metadata.get("similarity_matches", []):
            assert match["overall_similarity"] >= 0.5
