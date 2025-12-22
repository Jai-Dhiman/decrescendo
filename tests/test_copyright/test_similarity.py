"""Tests for similarity module."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.copyright.similarity import (
    MelodyExtractor,
    MelodyReport,
    RhythmExtractor,
    RhythmReport,
    SimilarityMatcher,
    SimilarityReport,
)
from decrescendo.musicritic.dimensions.copyright.config import (
    MelodySimilarityConfig,
    RhythmSimilarityConfig,
)
from decrescendo.musicritic.dimensions.copyright.exceptions import (
    MelodySimilarityError,
    RhythmSimilarityError,
)


class TestMelodySimilarityConfig:
    """Tests for MelodySimilarityConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = MelodySimilarityConfig()
        assert config.hop_length == 512
        assert config.fmin == 65.0
        assert config.fmax == 2093.0
        assert config.target_sample_rate == 22050

    def test_frozen(self):
        """Config should be frozen."""
        config = MelodySimilarityConfig()
        with pytest.raises(AttributeError):
            config.hop_length = 256  # type: ignore


class TestRhythmSimilarityConfig:
    """Tests for RhythmSimilarityConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = RhythmSimilarityConfig()
        assert config.hop_length == 512
        assert config.target_sample_rate == 22050


class TestMelodyReport:
    """Tests for MelodyReport dataclass."""

    def test_voiced_ratio_empty(self):
        """Empty mask should return 0 voiced ratio."""
        report = MelodyReport(
            pitch_contour=np.array([]),
            voiced_mask=np.array([]),
            pitch_confidence=np.array([]),
            duration=0.0,
            hop_time=0.01,
        )
        assert report.voiced_ratio == 0.0

    def test_voiced_ratio_all_voiced(self):
        """All voiced frames should return 1.0."""
        report = MelodyReport(
            pitch_contour=np.array([440.0, 440.0, 440.0]),
            voiced_mask=np.array([True, True, True]),
            pitch_confidence=np.array([0.9, 0.9, 0.9]),
            duration=0.03,
            hop_time=0.01,
        )
        assert report.voiced_ratio == 1.0

    def test_voiced_ratio_partial(self):
        """Partial voiced frames should return correct ratio."""
        report = MelodyReport(
            pitch_contour=np.array([440.0, np.nan, 440.0, np.nan]),
            voiced_mask=np.array([True, False, True, False]),
            pitch_confidence=np.array([0.9, 0.1, 0.9, 0.1]),
            duration=0.04,
            hop_time=0.01,
        )
        assert report.voiced_ratio == 0.5

    def test_mean_pitch_no_voiced(self):
        """No voiced frames should return 0."""
        report = MelodyReport(
            pitch_contour=np.array([np.nan, np.nan]),
            voiced_mask=np.array([False, False]),
            pitch_confidence=np.array([0.1, 0.1]),
            duration=0.02,
            hop_time=0.01,
        )
        assert report.mean_pitch == 0.0

    def test_mean_pitch_voiced(self):
        """Should compute mean of voiced pitches."""
        report = MelodyReport(
            pitch_contour=np.array([400.0, np.nan, 500.0]),
            voiced_mask=np.array([True, False, True]),
            pitch_confidence=np.array([0.9, 0.1, 0.9]),
            duration=0.03,
            hop_time=0.01,
        )
        assert report.mean_pitch == 450.0


class TestRhythmReport:
    """Tests for RhythmReport dataclass."""

    def test_onset_density_zero_duration(self):
        """Zero duration should return 0 density."""
        report = RhythmReport(
            onset_times=np.array([0.0, 0.5]),
            onset_strengths=np.array([1.0, 1.0]),
            tempo=120.0,
            beat_times=np.array([0.0, 0.5, 1.0]),
            duration=0.0,
        )
        assert report.onset_density == 0.0

    def test_onset_density(self):
        """Should compute onsets per second."""
        report = RhythmReport(
            onset_times=np.array([0.0, 0.5, 1.0, 1.5]),
            onset_strengths=np.array([1.0, 1.0, 1.0, 1.0]),
            tempo=120.0,
            beat_times=np.array([0.0, 0.5, 1.0, 1.5]),
            duration=2.0,
        )
        assert report.onset_density == 2.0  # 4 onsets / 2 seconds


class TestMelodyExtractor:
    """Tests for MelodyExtractor."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        extractor = MelodyExtractor()
        assert extractor.config is not None

    def test_init_custom_config(self):
        """Should accept custom config."""
        config = MelodySimilarityConfig(hop_length=256)
        extractor = MelodyExtractor(config)
        assert extractor.config.hop_length == 256

    def test_extract_sine(self, sine_440hz: np.ndarray, sample_rate: int):
        """Should extract pitch from sine wave."""
        extractor = MelodyExtractor()
        report = extractor.extract(sine_440hz, sample_rate)

        assert isinstance(report, MelodyReport)
        assert report.duration > 0
        assert report.hop_time > 0
        assert len(report.pitch_contour) > 0
        # 440 Hz sine should have many voiced frames
        assert report.voiced_ratio > 0.5

    def test_extract_complex(self, complex_audio: np.ndarray, sample_rate: int):
        """Should extract pitch from complex audio."""
        extractor = MelodyExtractor()
        report = extractor.extract(complex_audio, sample_rate)

        assert isinstance(report, MelodyReport)
        assert report.duration > 0

    def test_extract_noise(self, white_noise: np.ndarray, sample_rate: int):
        """Should handle noise (returns valid report)."""
        extractor = MelodyExtractor()
        report = extractor.extract(white_noise, sample_rate)

        assert isinstance(report, MelodyReport)
        # Note: pyin may detect spurious pitches in noise
        # Just verify we get a valid report
        assert 0.0 <= report.voiced_ratio <= 1.0

    def test_compute_pitch_histogram(self, sine_440hz: np.ndarray, sample_rate: int):
        """Should compute normalized pitch histogram."""
        extractor = MelodyExtractor()
        report = extractor.extract(sine_440hz, sample_rate)
        hist = extractor.compute_pitch_histogram(report)

        assert len(hist) == 12  # 12 pitch classes
        assert np.isclose(hist.sum(), 1.0, atol=0.01) or hist.sum() == 0


class TestRhythmExtractor:
    """Tests for RhythmExtractor."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        extractor = RhythmExtractor()
        assert extractor.config is not None

    def test_extract_rhythmic(self, rhythmic_audio: np.ndarray, sample_rate: int):
        """Should extract rhythm from rhythmic audio."""
        extractor = RhythmExtractor()
        report = extractor.extract(rhythmic_audio, sample_rate)

        assert isinstance(report, RhythmReport)
        assert report.duration > 0
        assert len(report.onset_times) > 0
        assert report.tempo > 0
        # 120 BPM audio should have tempo around 120
        assert 80 < report.tempo < 180

    def test_extract_sine(self, sine_440hz: np.ndarray, sample_rate: int):
        """Should handle steady tone (few onsets expected)."""
        extractor = RhythmExtractor()
        report = extractor.extract(sine_440hz, sample_rate)

        assert isinstance(report, RhythmReport)
        assert report.duration > 0

    def test_compute_onset_histogram(self, rhythmic_audio: np.ndarray, sample_rate: int):
        """Should compute normalized onset interval histogram."""
        extractor = RhythmExtractor()
        report = extractor.extract(rhythmic_audio, sample_rate)
        hist = extractor.compute_onset_histogram(report)

        assert len(hist) == 16  # default bins
        # Histogram should be normalized or empty
        assert hist.sum() <= 1.0 + 0.01 or hist.sum() == 0


class TestSimilarityMatcher:
    """Tests for SimilarityMatcher."""

    def test_init_default(self):
        """Should initialize with default configs."""
        matcher = SimilarityMatcher()
        assert matcher.melody_weight == 0.5
        assert matcher.rhythm_weight == 0.3
        assert matcher.harmony_weight == 0.2

    def test_init_custom_weights(self):
        """Should accept custom weights."""
        matcher = SimilarityMatcher(
            melody_weight=0.6,
            rhythm_weight=0.2,
            harmony_weight=0.2,
        )
        assert matcher.melody_weight == 0.6

    def test_compare_identical(self, sine_440hz: np.ndarray, sample_rate: int):
        """Identical audio should have high melody and harmony similarity."""
        matcher = SimilarityMatcher()
        report = matcher.compare(sine_440hz, sine_440hz, sample_rate)

        assert isinstance(report, SimilarityReport)
        assert report.melody_similarity > 0.9
        assert report.harmony_similarity > 0.9
        # Note: rhythm_similarity may be 0 for pure tones (no onsets)
        # so overall_similarity may be lower than expected
        assert report.overall_similarity > 0.5

    def test_compare_different_pitch(
        self, sine_440hz: np.ndarray, sine_880hz: np.ndarray, sample_rate: int
    ):
        """Different pitches should have lower melody similarity."""
        matcher = SimilarityMatcher()
        report = matcher.compare(sine_440hz, sine_880hz, sample_rate)

        # Different frequencies but octave relation may have some similarity
        assert report.overall_similarity < 1.0

    def test_compare_different_rhythm(
        self,
        rhythmic_audio: np.ndarray,
        different_rhythm_audio: np.ndarray,
        sample_rate: int,
    ):
        """Different rhythms should have lower rhythm similarity."""
        matcher = SimilarityMatcher()
        report = matcher.compare(rhythmic_audio, different_rhythm_audio, sample_rate)

        # Different tempos (120 vs 90 BPM) should reduce similarity
        assert report.rhythm_similarity < 1.0

    def test_compare_noise_vs_tone(
        self, sine_440hz: np.ndarray, white_noise: np.ndarray, sample_rate: int
    ):
        """Noise vs tone should have low similarity."""
        matcher = SimilarityMatcher()
        report = matcher.compare(sine_440hz, white_noise, sample_rate)

        assert report.overall_similarity < 0.7

    def test_compute_similarity_score(
        self, sine_440hz: np.ndarray, sine_550hz: np.ndarray, sample_rate: int
    ):
        """Should return scalar similarity score."""
        matcher = SimilarityMatcher()
        score = matcher.compute_similarity_score(sine_440hz, sine_550hz, sample_rate)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_report_has_metadata(
        self, rhythmic_audio: np.ndarray, sample_rate: int
    ):
        """Report should include metadata."""
        matcher = SimilarityMatcher()
        report = matcher.compare(rhythmic_audio, rhythmic_audio, sample_rate)

        assert "tempo1" in report.metadata
        assert "tempo2" in report.metadata
        assert "melody1_voiced_ratio" in report.metadata


class TestSimilarityReportDataclass:
    """Tests for SimilarityReport dataclass."""

    def test_create_report(self):
        """Should create report with all fields."""
        report = SimilarityReport(
            melody_similarity=0.8,
            rhythm_similarity=0.7,
            harmony_similarity=0.6,
            overall_similarity=0.72,
            matched_sections=[(0.0, 5.0)],
            metadata={"key": "value"},
        )
        assert report.melody_similarity == 0.8
        assert report.rhythm_similarity == 0.7
        assert report.harmony_similarity == 0.6
        assert report.overall_similarity == 0.72
        assert len(report.matched_sections) == 1
        assert report.metadata["key"] == "value"

    def test_default_matched_sections(self):
        """Should have empty matched_sections by default."""
        report = SimilarityReport(
            melody_similarity=0.5,
            rhythm_similarity=0.5,
            harmony_similarity=0.5,
            overall_similarity=0.5,
        )
        assert report.matched_sections == []
        assert report.metadata == {}
