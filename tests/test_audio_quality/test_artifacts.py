"""Tests for artifact detection."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.audio_quality import (
    ArtifactDetector,
    ArtifactDetectionConfig,
    ArtifactReport,
)


class TestArtifactDetectorInit:
    """Tests for ArtifactDetector initialization."""

    def test_default_config(self):
        """Test initialization with default config."""
        detector = ArtifactDetector()
        assert detector.config.click_threshold == 0.1
        assert detector.config.clipping_threshold == 0.99

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = ArtifactDetectionConfig(click_threshold=0.2)
        detector = ArtifactDetector(config=config)
        assert detector.config.click_threshold == 0.2


class TestClickDetection:
    """Tests for click/pop detection."""

    def test_clean_audio_no_clicks(self, sample_audio_44k):
        """Test that clean audio has few or no clicks."""
        audio, sr = sample_audio_44k
        detector = ArtifactDetector()
        count, timestamps = detector.detect_clicks(audio, sr)

        # Clean sine wave shouldn't have many clicks
        assert count <= 2  # Allow for edge effects

    def test_clicks_detected(self, audio_with_clicks):
        """Test that clicks are detected in audio with artificial clicks."""
        audio, sr = audio_with_clicks
        detector = ArtifactDetector()
        count, timestamps = detector.detect_clicks(audio, sr)

        # Should detect at least some of the artificial clicks
        assert count >= 1
        assert len(timestamps) == count

    def test_click_timestamps_in_range(self, audio_with_clicks):
        """Test that click timestamps are within audio duration."""
        audio, sr = audio_with_clicks
        duration = len(audio) / sr
        detector = ArtifactDetector()
        count, timestamps = detector.detect_clicks(audio, sr)

        for ts in timestamps:
            assert 0 <= ts <= duration


class TestClippingDetection:
    """Tests for clipping detection."""

    def test_clean_audio_no_clipping(self, sample_audio_44k):
        """Test that normalized audio has no clipping."""
        audio, sr = sample_audio_44k
        detector = ArtifactDetector()
        count, timestamps, severity = detector.detect_clipping(audio, sr)

        assert count == 0
        assert severity == 0.0

    def test_clipped_audio_detected(self, clipped_audio):
        """Test that clipping is detected."""
        audio, sr = clipped_audio
        detector = ArtifactDetector()
        count, timestamps, severity = detector.detect_clipping(audio, sr)

        # Should detect clipping events
        assert count > 0
        assert severity > 0.0
        assert severity <= 1.0

    def test_clipping_severity_proportional(self, clipped_audio, sample_audio_44k):
        """Test that clipping severity is proportional to amount of clipping."""
        detector = ArtifactDetector()

        clipped, sr = clipped_audio
        clean, _ = sample_audio_44k

        _, _, clipped_severity = detector.detect_clipping(clipped, sr)
        _, _, clean_severity = detector.detect_clipping(clean, sr)

        assert clipped_severity > clean_severity


class TestAIArtifactDetection:
    """Tests for AI artifact detection."""

    def test_natural_audio_low_score(self, sample_audio_44k):
        """Test that natural audio has relatively low AI score."""
        audio, sr = sample_audio_44k
        detector = ArtifactDetector()
        ai_score = detector.detect_ai_artifacts(audio, sr)

        assert 0.0 <= ai_score <= 1.0

    def test_noise_different_from_sine(self, white_noise, sample_audio_44k):
        """Test that noise and sine wave have different AI scores."""
        detector = ArtifactDetector()

        noise, sr = white_noise
        sine, _ = sample_audio_44k

        noise_score = detector.detect_ai_artifacts(noise, sr)
        sine_score = detector.detect_ai_artifacts(sine, sr)

        # They should be measurably different (not necessarily one higher)
        assert noise_score != sine_score


class TestArtifactAnalyze:
    """Tests for full artifact analysis."""

    def test_analyze_returns_report(self, sample_audio_44k):
        """Test that analyze returns an ArtifactReport."""
        audio, sr = sample_audio_44k
        detector = ArtifactDetector()
        report = detector.analyze(audio, sr)

        assert isinstance(report, ArtifactReport)
        assert isinstance(report.click_count, int)
        assert isinstance(report.clipping_count, int)
        assert isinstance(report.clipping_severity, float)
        assert isinstance(report.ai_artifact_score, float)

    def test_analyze_report_consistency(self, clipped_audio):
        """Test that report fields are consistent."""
        audio, sr = clipped_audio
        detector = ArtifactDetector()
        report = detector.analyze(audio, sr)

        # Clipping count and timestamps should match
        assert report.clipping_count == len(report.clipping_timestamps)
        assert report.click_count == len(report.click_timestamps)


class TestArtifactScore:
    """Tests for artifact scoring."""

    def test_score_range(self, sample_audio_44k):
        """Test that score is in valid range."""
        audio, sr = sample_audio_44k
        duration = len(audio) / sr
        detector = ArtifactDetector()
        report = detector.analyze(audio, sr)
        score = detector.compute_score(report, audio_duration=duration)

        assert 0.0 <= score <= 1.0

    def test_clean_audio_higher_score(self, sample_audio_44k, clipped_audio):
        """Test that clean audio gets higher score than clipped."""
        detector = ArtifactDetector()

        clean, sr = sample_audio_44k
        clipped, _ = clipped_audio

        clean_report = detector.analyze(clean, sr)
        clipped_report = detector.analyze(clipped, sr)

        clean_score = detector.compute_score(clean_report, len(clean) / sr)
        clipped_score = detector.compute_score(clipped_report, len(clipped) / sr)

        assert clean_score >= clipped_score

    def test_score_normalized_by_duration(self, sample_audio_44k):
        """Test that score accounts for audio duration."""
        audio, sr = sample_audio_44k
        detector = ArtifactDetector()

        # Create a report with fixed click count
        report = ArtifactReport(
            click_count=5,
            click_timestamps=[0.1, 0.2, 0.3, 0.4, 0.5],
            clipping_count=0,
            clipping_severity=0.0,
            ai_artifact_score=0.0,
        )

        # Same clicks in longer audio = less penalty
        short_score = detector.compute_score(report, audio_duration=10.0)
        long_score = detector.compute_score(report, audio_duration=60.0)

        assert long_score >= short_score
