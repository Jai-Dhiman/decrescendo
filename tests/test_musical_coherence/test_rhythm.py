"""Tests for RhythmAnalyzer."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.musical_coherence.config import RhythmConfig
from decrescendo.musicritic.dimensions.musical_coherence.exceptions import (
    DependencyNotAvailableError,
    RhythmAnalysisError,
)
from decrescendo.musicritic.dimensions.musical_coherence.rhythm import (
    RhythmAnalyzer,
    RhythmReport,
)


class TestRhythmConfig:
    """Test RhythmConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RhythmConfig()
        assert config.use_madmom is False
        assert config.min_tempo == 40.0
        assert config.max_tempo == 240.0
        assert config.tempo_stability_window == 8
        assert config.hop_length == 512

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RhythmConfig(
            use_madmom=True,
            min_tempo=60.0,
            max_tempo=180.0,
        )
        assert config.use_madmom is True
        assert config.min_tempo == 60.0
        assert config.max_tempo == 180.0


class TestRhythmReport:
    """Test RhythmReport dataclass."""

    def test_default_report(self):
        """Test default report values."""
        report = RhythmReport()
        assert report.tempo_bpm == 0.0
        assert report.tempo_confidence == 0.0
        assert report.beat_timestamps == []
        assert report.beat_count == 0
        assert report.tempo_stability == 0.0
        assert report.beat_strength == 0.0

    def test_custom_report(self):
        """Test custom report values."""
        report = RhythmReport(
            tempo_bpm=120.0,
            tempo_confidence=0.9,
            beat_timestamps=[0.5, 1.0, 1.5, 2.0],
            beat_count=4,
            tempo_stability=0.85,
            beat_strength=0.7,
        )
        assert report.tempo_bpm == 120.0
        assert report.beat_count == 4
        assert len(report.beat_timestamps) == 4


class TestRhythmAnalyzer:
    """Test RhythmAnalyzer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        analyzer = RhythmAnalyzer()
        assert analyzer.config is not None
        assert analyzer.config.use_madmom is False

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = RhythmConfig(min_tempo=80.0)
        analyzer = RhythmAnalyzer(config=config)
        assert analyzer.config.min_tempo == 80.0

    def test_madmom_availability_check(self):
        """Test that madmom availability is checked."""
        analyzer = RhythmAnalyzer()
        # Should return True or False without error
        available = analyzer.madmom_available
        assert isinstance(available, bool)

    def test_detect_beats_librosa(self, rhythmic_audio):
        """Test beat detection with librosa."""
        audio, sample_rate = rhythmic_audio
        analyzer = RhythmAnalyzer()

        beat_times, tempo = analyzer.detect_beats_librosa(audio, sample_rate)

        assert len(beat_times) > 0
        assert tempo > 0
        # Expected tempo is around 120 BPM
        assert 80 < tempo < 160

    def test_detect_beats_uses_config(self, rhythmic_audio):
        """Test that detect_beats respects config."""
        audio, sample_rate = rhythmic_audio
        config = RhythmConfig(use_madmom=False)
        analyzer = RhythmAnalyzer(config=config)

        beat_times, tempo = analyzer.detect_beats(audio, sample_rate)

        assert len(beat_times) > 0
        assert tempo > 0

    def test_compute_tempo_stability_steady(self):
        """Test tempo stability for steady tempo."""
        analyzer = RhythmAnalyzer()

        # Perfectly steady 120 BPM (0.5s intervals)
        beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        stability = analyzer.compute_tempo_stability(beats)

        # Should be very high stability
        assert stability > 0.9

    def test_compute_tempo_stability_varying(self, varying_tempo_audio):
        """Test tempo stability for varying tempo."""
        audio, sample_rate = varying_tempo_audio
        analyzer = RhythmAnalyzer()

        # Create varying intervals
        beats = np.array([0.0, 0.6, 1.1, 1.5, 1.8, 2.0, 2.15])
        stability = analyzer.compute_tempo_stability(beats)

        # Should be lower stability
        assert stability < 0.7

    def test_compute_tempo_stability_few_beats(self):
        """Test tempo stability with too few beats."""
        analyzer = RhythmAnalyzer()

        # Only 2 beats - not enough for stability
        beats = np.array([0.0, 0.5])
        stability = analyzer.compute_tempo_stability(beats)

        assert stability == 0.0

    def test_compute_beat_strength(self, rhythmic_audio):
        """Test beat strength computation."""
        audio, sample_rate = rhythmic_audio
        analyzer = RhythmAnalyzer()

        beat_times, _ = analyzer.detect_beats_librosa(audio, sample_rate)
        strength = analyzer.compute_beat_strength(audio, sample_rate, beat_times)

        assert 0.0 <= strength <= 1.0

    def test_compute_beat_strength_empty(self, sample_rate):
        """Test beat strength with no beats."""
        analyzer = RhythmAnalyzer()
        duration = 2.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

        strength = analyzer.compute_beat_strength(
            audio, sample_rate, np.array([])
        )

        assert strength == 0.0

    def test_analyze_rhythmic_audio(self, rhythmic_audio):
        """Test full analysis on rhythmic audio."""
        audio, sample_rate = rhythmic_audio
        analyzer = RhythmAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, RhythmReport)
        assert report.tempo_bpm > 0
        assert report.beat_count > 0
        assert len(report.beat_timestamps) == report.beat_count
        assert 0.0 <= report.tempo_stability <= 1.0
        assert 0.0 <= report.beat_strength <= 1.0

    def test_analyze_silence(self, silence):
        """Test analysis on silence."""
        audio, sample_rate = silence
        analyzer = RhythmAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, RhythmReport)
        # Silence may still detect some spurious beats

    def test_analyze_white_noise(self, white_noise):
        """Test analysis on white noise."""
        audio, sample_rate = white_noise
        analyzer = RhythmAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, RhythmReport)
        # White noise should have low tempo stability
        # (beats detected are not consistent)

    def test_compute_score_excellent(self):
        """Test score computation for excellent rhythm."""
        analyzer = RhythmAnalyzer()
        report = RhythmReport(
            tempo_bpm=120.0,
            tempo_confidence=0.95,
            tempo_stability=0.95,
            beat_strength=0.85,
        )

        score = analyzer.compute_score(report)

        assert score > 0.7

    def test_compute_score_poor(self):
        """Test score computation for poor rhythm."""
        analyzer = RhythmAnalyzer()
        report = RhythmReport(
            tempo_bpm=0.0,
            tempo_confidence=0.0,
            tempo_stability=0.0,
            beat_strength=0.0,
        )

        score = analyzer.compute_score(report)

        assert score == 0.0

    def test_compute_score_moderate(self):
        """Test score computation for moderate rhythm."""
        analyzer = RhythmAnalyzer()
        report = RhythmReport(
            tempo_bpm=120.0,
            tempo_confidence=0.7,
            tempo_stability=0.6,
            beat_strength=0.5,
        )

        score = analyzer.compute_score(report)

        assert 0.3 < score < 0.8

    def test_compute_score_bounds(self):
        """Test that score is always in valid range."""
        analyzer = RhythmAnalyzer()

        # Test edge cases
        reports = [
            RhythmReport(tempo_stability=1.0, beat_strength=1.0, tempo_bpm=120, tempo_confidence=1.0),
            RhythmReport(tempo_stability=0.0, beat_strength=0.0, tempo_bpm=0, tempo_confidence=0.0),
            RhythmReport(tempo_stability=0.5, beat_strength=0.5, tempo_bpm=300, tempo_confidence=0.5),
        ]

        for report in reports:
            score = analyzer.compute_score(report)
            assert 0.0 <= score <= 1.0

    def test_madmom_not_available_raises_error(self):
        """Test that requesting madmom when unavailable raises error."""
        config = RhythmConfig(use_madmom=True)
        analyzer = RhythmAnalyzer(config=config)

        if not analyzer.madmom_available:
            # Generate some audio
            sample_rate = 22050
            duration = 2.0
            audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

            with pytest.raises(DependencyNotAvailableError) as exc_info:
                analyzer.detect_beats(audio, sample_rate)

            assert "madmom" in str(exc_info.value)
