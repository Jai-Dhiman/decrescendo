"""Tests for HarmonyAnalyzer."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.musical_coherence.config import HarmonyConfig
from decrescendo.musicritic.dimensions.musical_coherence.exceptions import (
    DependencyNotAvailableError,
    HarmonyAnalysisError,
)
from decrescendo.musicritic.dimensions.musical_coherence.harmony import (
    HarmonyAnalyzer,
    HarmonyReport,
    PITCH_CLASSES,
)


class TestHarmonyConfig:
    """Test HarmonyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HarmonyConfig()
        assert config.use_essentia is False
        assert config.key_profile == "krumhansl"
        assert config.min_chord_duration == 0.25
        assert config.hop_length == 2048

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HarmonyConfig(
            use_essentia=True,
            min_chord_duration=0.5,
        )
        assert config.use_essentia is True
        assert config.min_chord_duration == 0.5


class TestHarmonyReport:
    """Test HarmonyReport dataclass."""

    def test_default_report(self):
        """Test default report values."""
        report = HarmonyReport()
        assert report.detected_key == ""
        assert report.key_confidence == 0.0
        assert report.chord_sequence == []
        assert report.chord_count == 0
        assert report.key_consistency == 0.0
        assert len(report.chroma_features) == 12

    def test_custom_report(self):
        """Test custom report values."""
        report = HarmonyReport(
            detected_key="C major",
            key_confidence=0.9,
            chord_sequence=[(0.0, 1.0, "C"), (1.0, 2.0, "G")],
            chord_count=2,
            unique_chord_count=2,
            key_consistency=0.85,
        )
        assert report.detected_key == "C major"
        assert report.chord_count == 2


class TestHarmonyAnalyzer:
    """Test HarmonyAnalyzer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        analyzer = HarmonyAnalyzer()
        assert analyzer.config is not None
        assert analyzer.config.use_essentia is False

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = HarmonyConfig(min_chord_duration=0.5)
        analyzer = HarmonyAnalyzer(config=config)
        assert analyzer.config.min_chord_duration == 0.5

    def test_essentia_availability_check(self):
        """Test that Essentia availability is checked."""
        analyzer = HarmonyAnalyzer()
        available = analyzer.essentia_available
        assert isinstance(available, bool)

    def test_compute_chroma(self, harmonic_audio):
        """Test chroma computation."""
        audio, sample_rate = harmonic_audio
        analyzer = HarmonyAnalyzer()

        chroma = analyzer.compute_chroma(audio, sample_rate)

        assert chroma.shape[0] == 12  # 12 pitch classes
        assert chroma.shape[1] > 0  # Some frames

    def test_detect_key_major(self, harmonic_audio):
        """Test key detection on C major chord."""
        audio, sample_rate = harmonic_audio
        analyzer = HarmonyAnalyzer()

        key, confidence = analyzer.detect_key(audio, sample_rate)

        assert "major" in key or "minor" in key
        assert 0.0 <= confidence <= 1.0
        # C major chord should detect C major or A minor (relative)
        assert any(pitch in key for pitch in ["C", "A"])

    def test_detect_key_confidence(self, harmonic_audio):
        """Test key detection confidence."""
        audio, sample_rate = harmonic_audio
        analyzer = HarmonyAnalyzer()

        key, confidence = analyzer.detect_key(audio, sample_rate)

        # Clean harmonic content should have reasonable confidence
        assert confidence > 0.3

    def test_detect_chords_chroma(self, chord_progression_audio):
        """Test chord detection using chroma method."""
        audio, sample_rate = chord_progression_audio
        analyzer = HarmonyAnalyzer()

        chords = analyzer.detect_chords_chroma(audio, sample_rate)

        assert len(chords) > 0
        # Each chord should have (start, end, name)
        for start, end, name in chords:
            assert start < end
            assert isinstance(name, str)

    def test_detect_chords_respects_config(self, harmonic_audio):
        """Test that detect_chords respects config."""
        audio, sample_rate = harmonic_audio
        config = HarmonyConfig(use_essentia=False)
        analyzer = HarmonyAnalyzer(config=config)

        chords = analyzer.detect_chords(audio, sample_rate)

        assert isinstance(chords, list)

    def test_compute_key_consistency(self, harmonic_audio):
        """Test key consistency computation."""
        audio, sample_rate = harmonic_audio
        analyzer = HarmonyAnalyzer()

        chroma = analyzer.compute_chroma(audio, sample_rate)
        consistency = analyzer.compute_key_consistency(chroma, "C major")

        assert 0.0 <= consistency <= 1.0

    def test_compute_key_consistency_empty(self):
        """Test key consistency with empty chroma."""
        analyzer = HarmonyAnalyzer()
        chroma = np.zeros((12, 0))

        consistency = analyzer.compute_key_consistency(chroma, "C major")

        assert consistency == 0.0

    def test_compute_key_consistency_invalid_key(self):
        """Test key consistency with invalid key string."""
        analyzer = HarmonyAnalyzer()
        chroma = np.random.rand(12, 10)

        consistency = analyzer.compute_key_consistency(chroma, "invalid")

        assert consistency == 0.0

    def test_compute_progression_quality_good(self):
        """Test progression quality for good progression."""
        analyzer = HarmonyAnalyzer()

        # I-IV-V-I progression in C
        chords = [
            (0.0, 1.0, "C"),
            (1.0, 2.0, "F"),
            (2.0, 3.0, "G"),
            (3.0, 4.0, "C"),
        ]

        quality = analyzer.compute_progression_quality(chords)

        # Should be good quality
        assert quality > 0.5

    def test_compute_progression_quality_random(self):
        """Test progression quality for random progression."""
        analyzer = HarmonyAnalyzer()

        # Random chromatic chords
        chords = [
            (0.0, 1.0, "C"),
            (1.0, 2.0, "C#"),
            (2.0, 3.0, "F#"),
            (3.0, 4.0, "A#"),
        ]

        quality = analyzer.compute_progression_quality(chords)

        # Quality should still be in range
        assert 0.0 <= quality <= 1.0

    def test_compute_progression_quality_single_chord(self):
        """Test progression quality for single chord."""
        analyzer = HarmonyAnalyzer()

        chords = [(0.0, 4.0, "C")]

        quality = analyzer.compute_progression_quality(chords)

        # Neutral for single chord
        assert quality == 0.5

    def test_compute_progression_quality_no_chord(self):
        """Test progression quality for no chord."""
        analyzer = HarmonyAnalyzer()

        chords = [(0.0, 4.0, "N")]

        quality = analyzer.compute_progression_quality(chords)

        assert 0.0 <= quality <= 1.0

    def test_analyze_harmonic_audio(self, harmonic_audio):
        """Test full analysis on harmonic audio."""
        audio, sample_rate = harmonic_audio
        analyzer = HarmonyAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, HarmonyReport)
        assert report.detected_key != ""
        assert 0.0 <= report.key_confidence <= 1.0
        assert 0.0 <= report.key_consistency <= 1.0
        assert len(report.chroma_features) == 12

    def test_analyze_chord_progression(self, chord_progression_audio):
        """Test analysis on chord progression."""
        audio, sample_rate = chord_progression_audio
        analyzer = HarmonyAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, HarmonyReport)
        assert report.chord_count > 0

    def test_analyze_silence(self, silence):
        """Test analysis on silence."""
        audio, sample_rate = silence
        analyzer = HarmonyAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, HarmonyReport)

    def test_analyze_white_noise(self, white_noise):
        """Test analysis on white noise."""
        audio, sample_rate = white_noise
        analyzer = HarmonyAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, HarmonyReport)
        # Noise should have low key confidence
        # (may still detect something due to random patterns)

    def test_compute_score_excellent(self):
        """Test score computation for excellent harmony."""
        analyzer = HarmonyAnalyzer()
        report = HarmonyReport(
            key_confidence=0.95,
            key_consistency=0.9,
            progression_quality=0.85,
        )

        score = analyzer.compute_score(report)

        assert score > 0.8

    def test_compute_score_poor(self):
        """Test score computation for poor harmony."""
        analyzer = HarmonyAnalyzer()
        report = HarmonyReport(
            key_confidence=0.1,
            key_consistency=0.2,
            progression_quality=0.1,
        )

        score = analyzer.compute_score(report)

        assert score < 0.3

    def test_compute_score_moderate(self):
        """Test score computation for moderate harmony."""
        analyzer = HarmonyAnalyzer()
        report = HarmonyReport(
            key_confidence=0.6,
            key_consistency=0.5,
            progression_quality=0.5,
        )

        score = analyzer.compute_score(report)

        assert 0.4 < score < 0.7

    def test_compute_score_bounds(self):
        """Test that score is always in valid range."""
        analyzer = HarmonyAnalyzer()

        reports = [
            HarmonyReport(key_confidence=1.0, key_consistency=1.0, progression_quality=1.0),
            HarmonyReport(key_confidence=0.0, key_consistency=0.0, progression_quality=0.0),
            HarmonyReport(key_confidence=0.5, key_consistency=0.5, progression_quality=0.5),
        ]

        for report in reports:
            score = analyzer.compute_score(report)
            assert 0.0 <= score <= 1.0

    def test_essentia_not_available_raises_error(self, harmonic_audio):
        """Test that requesting Essentia when unavailable raises error."""
        audio, sample_rate = harmonic_audio
        config = HarmonyConfig(use_essentia=True)
        analyzer = HarmonyAnalyzer(config=config)

        if not analyzer.essentia_available:
            with pytest.raises(DependencyNotAvailableError) as exc_info:
                analyzer.detect_chords(audio, sample_rate)

            assert "essentia" in str(exc_info.value)


class TestPitchClasses:
    """Test pitch class constants."""

    def test_pitch_classes_length(self):
        """Test that there are 12 pitch classes."""
        assert len(PITCH_CLASSES) == 12

    def test_pitch_classes_content(self):
        """Test pitch class names."""
        assert PITCH_CLASSES[0] == "C"
        assert PITCH_CLASSES[9] == "A"
        assert "C#" in PITCH_CLASSES
        assert "D#" in PITCH_CLASSES
