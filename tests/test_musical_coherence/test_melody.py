"""Tests for MelodyAnalyzer."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.musical_coherence.config import MelodyConfig
from decrescendo.musicritic.dimensions.musical_coherence.exceptions import (
    MelodyAnalysisError,
)
from decrescendo.musicritic.dimensions.musical_coherence.melody import (
    MelodyAnalyzer,
    MelodyReport,
)


class TestMelodyConfig:
    """Test MelodyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MelodyConfig()
        assert config.fmin == 65.0  # C2
        assert config.fmax == 2093.0  # C7
        assert config.frame_length == 2048
        assert config.hop_length == 512
        assert config.min_voiced_ratio == 0.1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MelodyConfig(
            fmin=100.0,
            fmax=1000.0,
            min_voiced_ratio=0.2,
        )
        assert config.fmin == 100.0
        assert config.fmax == 1000.0
        assert config.min_voiced_ratio == 0.2


class TestMelodyReport:
    """Test MelodyReport dataclass."""

    def test_default_report(self):
        """Test default report values."""
        report = MelodyReport()
        assert len(report.pitch_contour) == 0
        assert len(report.pitch_timestamps) == 0
        assert report.voiced_ratio == 0.0
        assert report.pitch_range_hz == 0.0
        assert report.phrase_count == 0
        assert report.contour_complexity == 0.0

    def test_custom_report(self):
        """Test custom report values."""
        report = MelodyReport(
            pitch_contour=np.array([440.0, 450.0, 460.0]),
            voiced_ratio=0.8,
            pitch_range_hz=100.0,
            phrase_count=2,
            contour_complexity=0.6,
        )
        assert len(report.pitch_contour) == 3
        assert report.voiced_ratio == 0.8
        assert report.phrase_count == 2


class TestMelodyAnalyzer:
    """Test MelodyAnalyzer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        analyzer = MelodyAnalyzer()
        assert analyzer.config is not None
        assert analyzer.config.fmin == 65.0

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = MelodyConfig(fmin=100.0)
        analyzer = MelodyAnalyzer(config=config)
        assert analyzer.config.fmin == 100.0

    def test_extract_pitch(self, melodic_audio):
        """Test pitch extraction from melodic audio."""
        audio, sample_rate = melodic_audio
        analyzer = MelodyAnalyzer()

        f0, voiced_flag, voiced_probs = analyzer.extract_pitch(audio, sample_rate)

        assert len(f0) > 0
        assert len(voiced_flag) == len(f0)
        assert len(voiced_probs) == len(f0)
        # Should detect some voiced content
        assert np.any(voiced_flag)

    def test_extract_pitch_silence(self, silence):
        """Test pitch extraction from silence."""
        audio, sample_rate = silence
        analyzer = MelodyAnalyzer()

        f0, voiced_flag, voiced_probs = analyzer.extract_pitch(audio, sample_rate)

        # Silence should have very low voiced ratio
        voiced_ratio = np.mean(voiced_flag)
        assert voiced_ratio < 0.1

    def test_compute_timestamps(self, sample_rate):
        """Test timestamp computation."""
        analyzer = MelodyAnalyzer()
        n_frames = 100

        timestamps = analyzer.compute_timestamps(n_frames, sample_rate)

        assert len(timestamps) == n_frames
        assert timestamps[0] == 0.0
        assert timestamps[-1] > 0

    def test_detect_phrase_boundaries(self):
        """Test phrase boundary detection."""
        analyzer = MelodyAnalyzer()

        # Create f0 with a gap (phrase break)
        f0 = np.array([440.0, 440.0, 440.0, np.nan, np.nan, np.nan,
                       np.nan, np.nan, 550.0, 550.0, 550.0])
        voiced_flag = np.array([True, True, True, False, False, False,
                                False, False, True, True, True])
        timestamps = np.linspace(0, 1, len(f0))

        boundaries = analyzer.detect_phrase_boundaries(f0, voiced_flag, timestamps)

        # Should detect at least one boundary (the gap)
        assert len(boundaries) >= 1

    def test_detect_phrase_boundaries_no_melody(self):
        """Test phrase boundary detection with no melody."""
        analyzer = MelodyAnalyzer()

        f0 = np.array([np.nan, np.nan, np.nan])
        voiced_flag = np.array([False, False, False])
        timestamps = np.array([0.0, 0.5, 1.0])

        boundaries = analyzer.detect_phrase_boundaries(f0, voiced_flag, timestamps)

        assert boundaries == []

    def test_compute_contour_complexity_varied(self, melodic_audio):
        """Test contour complexity for varied melody."""
        audio, sample_rate = melodic_audio
        analyzer = MelodyAnalyzer()

        f0, voiced_flag, _ = analyzer.extract_pitch(audio, sample_rate)
        complexity = analyzer.compute_contour_complexity(f0, voiced_flag)

        # C major scale should have some complexity
        assert complexity > 0.0
        assert complexity <= 1.0

    def test_compute_contour_complexity_monotone(self, sample_rate):
        """Test contour complexity for monotone (single pitch)."""
        analyzer = MelodyAnalyzer()

        # Single pitch repeated
        f0 = np.full(100, 440.0)
        voiced_flag = np.ones(100, dtype=bool)

        complexity = analyzer.compute_contour_complexity(f0, voiced_flag)

        # Monotone should have low complexity
        assert complexity < 0.3

    def test_compute_contour_complexity_no_voiced(self):
        """Test contour complexity with no voiced content."""
        analyzer = MelodyAnalyzer()

        f0 = np.full(100, np.nan)
        voiced_flag = np.zeros(100, dtype=bool)

        complexity = analyzer.compute_contour_complexity(f0, voiced_flag)

        assert complexity == 0.0

    def test_compute_pitch_stability_stable(self):
        """Test pitch stability for stable melody."""
        analyzer = MelodyAnalyzer()

        # Very stable pitches (small variations)
        f0 = np.array([440.0, 441.0, 440.5, 440.2, 440.8])
        voiced_flag = np.ones(len(f0), dtype=bool)

        stability = analyzer.compute_pitch_stability(f0, voiced_flag)

        # Should be very stable
        assert stability > 0.8

    def test_compute_pitch_stability_unstable(self):
        """Test pitch stability for unstable melody."""
        analyzer = MelodyAnalyzer()

        # Wild pitch jumps
        f0 = np.array([200.0, 800.0, 300.0, 600.0, 250.0])
        voiced_flag = np.ones(len(f0), dtype=bool)

        stability = analyzer.compute_pitch_stability(f0, voiced_flag)

        # Should be unstable
        assert stability < 0.5

    def test_analyze_melodic_audio(self, melodic_audio):
        """Test full analysis on melodic audio."""
        audio, sample_rate = melodic_audio
        analyzer = MelodyAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, MelodyReport)
        assert len(report.pitch_contour) > 0
        assert len(report.pitch_timestamps) > 0
        assert report.voiced_ratio > 0
        assert 0.0 <= report.contour_complexity <= 1.0
        assert 0.0 <= report.pitch_stability <= 1.0

    def test_analyze_harmonic_audio(self, harmonic_audio):
        """Test analysis on harmonic (chord) audio."""
        audio, sample_rate = harmonic_audio
        analyzer = MelodyAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, MelodyReport)
        # Chord may or may not be detected as voiced

    def test_analyze_silence(self, silence):
        """Test analysis on silence."""
        audio, sample_rate = silence
        analyzer = MelodyAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, MelodyReport)
        assert report.voiced_ratio < 0.1
        assert report.phrase_count == 0

    def test_analyze_white_noise(self, white_noise):
        """Test analysis on white noise."""
        audio, sample_rate = white_noise
        analyzer = MelodyAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, MelodyReport)
        # Noise should have low stability if voiced
        if report.voiced_ratio > 0.1:
            assert report.pitch_stability < 0.7

    def test_compute_score_excellent(self):
        """Test score computation for excellent melody."""
        analyzer = MelodyAnalyzer()
        report = MelodyReport(
            voiced_ratio=0.6,
            contour_complexity=0.7,
            pitch_stability=0.9,
        )

        score = analyzer.compute_score(report)

        assert score > 0.7

    def test_compute_score_no_melody(self):
        """Test score computation for no melody."""
        analyzer = MelodyAnalyzer()
        report = MelodyReport(
            voiced_ratio=0.0,
            contour_complexity=0.0,
            pitch_stability=0.0,
        )

        score = analyzer.compute_score(report)

        assert score == 0.0

    def test_compute_score_moderate(self):
        """Test score computation for moderate melody."""
        analyzer = MelodyAnalyzer()
        report = MelodyReport(
            voiced_ratio=0.3,
            contour_complexity=0.4,
            pitch_stability=0.6,
        )

        score = analyzer.compute_score(report)

        assert 0.3 < score < 0.8

    def test_compute_score_below_min_voiced(self):
        """Test score when voiced ratio is below minimum."""
        config = MelodyConfig(min_voiced_ratio=0.2)
        analyzer = MelodyAnalyzer(config=config)
        report = MelodyReport(
            voiced_ratio=0.05,  # Below minimum
            contour_complexity=0.5,
            pitch_stability=0.5,
        )

        score = analyzer.compute_score(report)

        # Voiced score should be 0, but complexity and stability contribute
        assert score < 0.5

    def test_compute_score_bounds(self):
        """Test that score is always in valid range."""
        analyzer = MelodyAnalyzer()

        reports = [
            MelodyReport(voiced_ratio=1.0, contour_complexity=1.0, pitch_stability=1.0),
            MelodyReport(voiced_ratio=0.0, contour_complexity=0.0, pitch_stability=0.0),
            MelodyReport(voiced_ratio=0.5, contour_complexity=0.5, pitch_stability=0.5),
        ]

        for report in reports:
            score = analyzer.compute_score(report)
            assert 0.0 <= score <= 1.0
