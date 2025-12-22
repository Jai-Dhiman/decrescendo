"""Tests for loudness analysis."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.audio_quality import (
    LoudnessAnalyzer,
    LoudnessConfig,
    LoudnessReport,
)
from decrescendo.musicritic.dimensions.audio_quality.exceptions import (
    AudioTooShortError,
    LoudnessAnalysisError,
)


class TestLoudnessAnalyzerInit:
    """Tests for LoudnessAnalyzer initialization."""

    def test_default_config(self):
        """Test initialization with default config."""
        analyzer = LoudnessAnalyzer()
        assert analyzer.config.target_lufs == -14.0
        assert analyzer.config.max_true_peak_dbtp == -1.0

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = LoudnessConfig(target_lufs=-16.0)
        analyzer = LoudnessAnalyzer(config=config)
        assert analyzer.config.target_lufs == -16.0


class TestIntegratedLoudness:
    """Tests for integrated loudness measurement."""

    def test_integrated_loudness_value(self, sample_audio_44k):
        """Test that integrated loudness returns reasonable value."""
        audio, sr = sample_audio_44k
        analyzer = LoudnessAnalyzer()
        lufs = analyzer.measure_integrated_loudness(audio, sr)
        # A -6 dBFS sine wave should be around -9 to -10 LUFS
        assert -30 < lufs < 0

    def test_quiet_audio_lower_lufs(self, quiet_audio, sample_audio_44k):
        """Test that quiet audio has lower LUFS."""
        analyzer = LoudnessAnalyzer()
        quiet, sr = quiet_audio
        normal, _ = sample_audio_44k

        quiet_lufs = analyzer.measure_integrated_loudness(quiet, sr)
        normal_lufs = analyzer.measure_integrated_loudness(normal, sr)

        assert quiet_lufs < normal_lufs

    def test_silence_returns_low_value(self, silence):
        """Test that silence returns a very low LUFS value."""
        audio, sr = silence
        analyzer = LoudnessAnalyzer()
        lufs = analyzer.measure_integrated_loudness(audio, sr)
        assert lufs <= -70.0


class TestTruePeak:
    """Tests for True Peak measurement."""

    def test_true_peak_value(self, sample_audio_44k):
        """Test that True Peak returns reasonable value."""
        audio, sr = sample_audio_44k
        analyzer = LoudnessAnalyzer()
        peak = analyzer.measure_true_peak(audio, sr)
        # Audio at 0.5 amplitude should be around -6 dBTP
        assert -20 < peak < 0

    def test_loud_audio_higher_peak(self, loud_audio, quiet_audio):
        """Test that loud audio has higher True Peak."""
        analyzer = LoudnessAnalyzer()
        loud, sr = loud_audio
        quiet, _ = quiet_audio

        loud_peak = analyzer.measure_true_peak(loud, sr)
        quiet_peak = analyzer.measure_true_peak(quiet, sr)

        assert loud_peak > quiet_peak

    def test_silence_true_peak(self, silence):
        """Test True Peak for silence."""
        audio, sr = silence
        analyzer = LoudnessAnalyzer()
        peak = analyzer.measure_true_peak(audio, sr)
        assert peak <= -70.0


class TestLoudnessRange:
    """Tests for Loudness Range measurement."""

    def test_loudness_range_value(self, sample_audio_44k):
        """Test that LRA returns a value."""
        audio, sr = sample_audio_44k
        analyzer = LoudnessAnalyzer()
        lra = analyzer.measure_loudness_range(audio, sr)
        # Constant amplitude sine wave should have low LRA
        assert lra >= 0

    def test_short_audio_returns_zero(self, sample_audio_short):
        """Test that very short audio returns 0 LRA."""
        audio, sr = sample_audio_short
        analyzer = LoudnessAnalyzer()
        lra = analyzer.measure_loudness_range(audio, sr)
        assert lra == 0.0


class TestLoudnessAnalyze:
    """Tests for full loudness analysis."""

    def test_analyze_returns_report(self, sample_audio_44k):
        """Test that analyze returns a LoudnessReport."""
        audio, sr = sample_audio_44k
        analyzer = LoudnessAnalyzer()
        report = analyzer.analyze(audio, sr)

        assert isinstance(report, LoudnessReport)
        assert isinstance(report.integrated_lufs, float)
        assert isinstance(report.loudness_range_lu, float)
        assert isinstance(report.true_peak_dbtp, float)

    def test_analyze_too_short_raises(self):
        """Test that very short audio raises AudioTooShortError."""
        sr = 44100
        audio = np.zeros(int(sr * 0.1), dtype=np.float32)  # 0.1 seconds
        analyzer = LoudnessAnalyzer()

        with pytest.raises(AudioTooShortError) as exc_info:
            analyzer.analyze(audio, sr)

        assert exc_info.value.duration < exc_info.value.min_duration

    def test_streaming_compliance_true(self, sample_audio_44k):
        """Test streaming compliance detection."""
        audio, sr = sample_audio_44k
        analyzer = LoudnessAnalyzer()
        report = analyzer.analyze(audio, sr)

        # Our sample is a moderate level, should check compliance flags
        assert isinstance(report.streaming_compliant, bool)
        assert isinstance(report.true_peak_compliant, bool)

    def test_loud_audio_not_peak_compliant(self, loud_audio):
        """Test that very loud audio fails True Peak compliance."""
        audio, sr = loud_audio
        analyzer = LoudnessAnalyzer()
        report = analyzer.analyze(audio, sr)

        # At 0.95 amplitude, True Peak should exceed -1 dBTP
        assert report.true_peak_dbtp > -1.0
        assert not report.true_peak_compliant


class TestLoudnessScore:
    """Tests for loudness scoring."""

    def test_score_range(self, sample_audio_44k):
        """Test that score is in valid range."""
        audio, sr = sample_audio_44k
        analyzer = LoudnessAnalyzer()
        report = analyzer.analyze(audio, sr)
        score = analyzer.compute_score(report)

        assert 0.0 <= score <= 1.0

    def test_compliant_audio_higher_score(self, sample_audio_44k, loud_audio):
        """Test that compliant audio gets higher score than non-compliant."""
        analyzer = LoudnessAnalyzer()

        normal_audio, sr = sample_audio_44k
        loud, _ = loud_audio

        normal_report = analyzer.analyze(normal_audio, sr)
        loud_report = analyzer.analyze(loud, sr)

        normal_score = analyzer.compute_score(normal_report)
        loud_score = analyzer.compute_score(loud_report)

        # Normal audio should score higher (lower True Peak)
        assert normal_score >= loud_score
