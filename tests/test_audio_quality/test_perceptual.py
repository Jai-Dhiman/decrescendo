"""Tests for perceptual quality analysis."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.audio_quality import (
    PerceptualAnalyzer,
    PerceptualConfig,
    PerceptualReport,
)


class TestPerceptualAnalyzerInit:
    """Tests for PerceptualAnalyzer initialization."""

    def test_default_config(self):
        """Test initialization with default config."""
        analyzer = PerceptualAnalyzer()
        assert analyzer.config.target_sample_rate == 44100
        assert len(analyzer.config.frequency_bands) == 4

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = PerceptualConfig(min_centroid_hz=300.0)
        analyzer = PerceptualAnalyzer(config=config)
        assert analyzer.config.min_centroid_hz == 300.0


class TestSpectralCentroid:
    """Tests for spectral centroid computation."""

    def test_centroid_returns_values(self, sample_audio_44k):
        """Test that spectral centroid returns mean and std."""
        audio, sr = sample_audio_44k
        analyzer = PerceptualAnalyzer()
        mean, std = analyzer.compute_spectral_centroid(audio, sr)

        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert mean > 0
        assert std >= 0

    def test_sine_wave_centroid_near_frequency(self, sample_audio_44k):
        """Test that sine wave centroid is near its frequency."""
        audio, sr = sample_audio_44k
        analyzer = PerceptualAnalyzer()
        mean, _ = analyzer.compute_spectral_centroid(audio, sr)

        # A 440 Hz sine wave should have centroid near 440 Hz
        assert 400 <= mean <= 500

    def test_low_frequency_lower_centroid(self, bass_heavy_audio, sample_audio_44k):
        """Test that bass-heavy audio has lower centroid."""
        analyzer = PerceptualAnalyzer()

        bass, sr = bass_heavy_audio
        normal, _ = sample_audio_44k

        bass_centroid, _ = analyzer.compute_spectral_centroid(bass, sr)
        normal_centroid, _ = analyzer.compute_spectral_centroid(normal, sr)

        assert bass_centroid < normal_centroid


class TestSpectralFlatness:
    """Tests for spectral flatness computation."""

    def test_flatness_returns_values(self, sample_audio_44k):
        """Test that spectral flatness returns mean and std."""
        audio, sr = sample_audio_44k
        analyzer = PerceptualAnalyzer()
        mean, std = analyzer.compute_spectral_flatness(audio, sr)

        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert 0 <= mean <= 1
        assert std >= 0

    def test_sine_wave_low_flatness(self, sample_audio_44k):
        """Test that pure sine wave has low flatness (tonal)."""
        audio, sr = sample_audio_44k
        analyzer = PerceptualAnalyzer()
        mean, _ = analyzer.compute_spectral_flatness(audio, sr)

        # Pure tone should have very low flatness
        assert mean < 0.2

    def test_noise_higher_flatness(self, white_noise, sample_audio_44k):
        """Test that noise has higher flatness than sine wave."""
        analyzer = PerceptualAnalyzer()

        noise, sr = white_noise
        sine, _ = sample_audio_44k

        noise_flatness, _ = analyzer.compute_spectral_flatness(noise, sr)
        sine_flatness, _ = analyzer.compute_spectral_flatness(sine, sr)

        assert noise_flatness > sine_flatness


class TestFrequencyBalance:
    """Tests for frequency balance computation."""

    def test_balance_returns_dict(self, sample_audio_44k):
        """Test that frequency balance returns expected dictionary."""
        audio, sr = sample_audio_44k
        analyzer = PerceptualAnalyzer()
        balance = analyzer.compute_frequency_balance(audio, sr)

        assert isinstance(balance, dict)
        assert "bass" in balance
        assert "mids" in balance
        assert "upper_mids" in balance
        assert "highs" in balance

    def test_balance_sums_to_one(self, sample_audio_44k):
        """Test that frequency balance ratios sum to 1.0."""
        audio, sr = sample_audio_44k
        analyzer = PerceptualAnalyzer()
        balance = analyzer.compute_frequency_balance(audio, sr)

        total = sum(balance.values())
        assert abs(total - 1.0) < 0.01

    def test_bass_heavy_audio_more_bass(self, bass_heavy_audio, sample_audio_44k):
        """Test that bass-heavy audio has more energy in bass band."""
        analyzer = PerceptualAnalyzer()

        bass, sr = bass_heavy_audio
        normal, _ = sample_audio_44k

        bass_balance = analyzer.compute_frequency_balance(bass, sr)
        normal_balance = analyzer.compute_frequency_balance(normal, sr)

        assert bass_balance["bass"] > normal_balance["bass"]


class TestBandwidthUtilization:
    """Tests for bandwidth utilization computation."""

    def test_utilization_in_range(self, sample_audio_44k):
        """Test that bandwidth utilization is in valid range."""
        audio, sr = sample_audio_44k
        analyzer = PerceptualAnalyzer()
        utilization = analyzer.compute_bandwidth_utilization(audio, sr)

        assert 0.0 <= utilization <= 1.0

    def test_noise_higher_utilization(self, white_noise, sample_audio_44k):
        """Test that noise has higher bandwidth utilization than sine wave."""
        analyzer = PerceptualAnalyzer()

        noise, sr = white_noise
        sine, _ = sample_audio_44k

        noise_util = analyzer.compute_bandwidth_utilization(noise, sr)
        sine_util = analyzer.compute_bandwidth_utilization(sine, sr)

        # White noise should use more bandwidth
        assert noise_util > sine_util


class TestPerceptualAnalyze:
    """Tests for full perceptual analysis."""

    def test_analyze_returns_report(self, sample_audio_44k):
        """Test that analyze returns a PerceptualReport."""
        audio, sr = sample_audio_44k
        analyzer = PerceptualAnalyzer()
        report = analyzer.analyze(audio, sr)

        assert isinstance(report, PerceptualReport)
        assert report.spectral_centroid_mean > 0
        assert 0 <= report.spectral_flatness_mean <= 1
        assert len(report.frequency_balance) == 4
        assert 0 <= report.balance_deviation <= 1
        assert 0 <= report.bandwidth_utilization <= 1


class TestPerceptualScore:
    """Tests for perceptual scoring."""

    def test_score_range(self, sample_audio_44k):
        """Test that score is in valid range."""
        audio, sr = sample_audio_44k
        analyzer = PerceptualAnalyzer()
        report = analyzer.analyze(audio, sr)
        score = analyzer.compute_score(report)

        assert 0.0 <= score <= 1.0

    def test_balanced_audio_higher_score(self, sample_audio_44k, bass_heavy_audio):
        """Test that more balanced audio gets higher score."""
        analyzer = PerceptualAnalyzer()

        normal, sr = sample_audio_44k
        bass, _ = bass_heavy_audio

        normal_report = analyzer.analyze(normal, sr)
        bass_report = analyzer.analyze(bass, sr)

        normal_score = analyzer.compute_score(normal_report)
        bass_score = analyzer.compute_score(bass_report)

        # Bass-heavy audio should have higher balance deviation
        assert bass_report.balance_deviation > normal_report.balance_deviation
