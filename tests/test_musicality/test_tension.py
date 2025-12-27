"""Tests for Tension analyzer."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.musicality import (
    TensionAnalyzer,
    TensionConfig,
    TensionReport,
    TensionAnalysisError,
    TISAnalyzer,
    TISReport,
)


class TestTensionAnalyzerAttributes:
    """Tests for TensionAnalyzer attributes and initialization."""

    def test_default_config(self) -> None:
        """Test that default config is used when not provided."""
        analyzer = TensionAnalyzer()
        assert isinstance(analyzer.config, TensionConfig)

    def test_custom_config(self) -> None:
        """Test that custom config is used."""
        config = TensionConfig(resolution_threshold=0.25)
        analyzer = TensionAnalyzer(config=config)
        assert analyzer.config.resolution_threshold == 0.25


class TestTensionCurve:
    """Tests for tension curve computation."""

    def test_compute_tension_curve_shape(
        self, tension_resolution_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that tension curve has correct shape."""
        audio, sr = tension_resolution_audio
        tis_analyzer = TISAnalyzer()
        tension_analyzer = TensionAnalyzer()

        tis_report = tis_analyzer.analyze(audio, sr)
        tension_curve = tension_analyzer.compute_tension_curve(tis_report)

        assert len(tension_curve) > 0
        assert len(tension_curve) <= len(tis_report.tensile_strain_curve)

    def test_compute_tension_curve_range(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that tension curve values are in valid range."""
        audio, sr = tonal_audio
        tis_analyzer = TISAnalyzer()
        tension_analyzer = TensionAnalyzer()

        tis_report = tis_analyzer.analyze(audio, sr)
        tension_curve = tension_analyzer.compute_tension_curve(tis_report)

        assert np.all(tension_curve >= 0.0)
        assert np.all(tension_curve <= 1.0)


class TestResolutionDetection:
    """Tests for resolution detection."""

    def test_detect_resolutions_returns_lists(
        self, tension_resolution_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that resolution detection returns correct types."""
        audio, sr = tension_resolution_audio
        tis_analyzer = TISAnalyzer()
        tension_analyzer = TensionAnalyzer()

        tis_report = tis_analyzer.analyze(audio, sr)
        tension_curve = tension_analyzer.compute_tension_curve(tis_report)
        times, strengths = tension_analyzer.detect_resolutions(
            tension_curve, tis_report.frame_rate
        )

        assert isinstance(times, list)
        assert isinstance(strengths, list)
        assert len(times) == len(strengths)

    def test_resolution_times_positive(
        self, tension_resolution_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that resolution times are positive."""
        audio, sr = tension_resolution_audio
        tis_analyzer = TISAnalyzer()
        tension_analyzer = TensionAnalyzer()

        tis_report = tis_analyzer.analyze(audio, sr)
        tension_curve = tension_analyzer.compute_tension_curve(tis_report)
        times, _ = tension_analyzer.detect_resolutions(
            tension_curve, tis_report.frame_rate
        )

        for t in times:
            assert t >= 0.0


class TestArcQuality:
    """Tests for arc quality computation."""

    def test_arc_quality_range(
        self, tension_resolution_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that arc quality is in valid range."""
        audio, sr = tension_resolution_audio
        tis_analyzer = TISAnalyzer()
        tension_analyzer = TensionAnalyzer()

        tis_report = tis_analyzer.analyze(audio, sr)
        tension_curve = tension_analyzer.compute_tension_curve(tis_report)
        times, _ = tension_analyzer.detect_resolutions(
            tension_curve, tis_report.frame_rate
        )

        duration = len(audio) / sr
        arc_quality = tension_analyzer.compute_arc_quality(
            tension_curve, times, duration
        )

        assert 0.0 <= arc_quality <= 1.0

    def test_flat_audio_lower_arc_quality(
        self, flat_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that flat audio has lower arc quality."""
        audio, sr = flat_audio
        tis_analyzer = TISAnalyzer()
        tension_analyzer = TensionAnalyzer()

        tis_report = tis_analyzer.analyze(audio, sr)
        tension_curve = tension_analyzer.compute_tension_curve(tis_report)
        times, _ = tension_analyzer.detect_resolutions(
            tension_curve, tis_report.frame_rate
        )

        duration = len(audio) / sr
        arc_quality = tension_analyzer.compute_arc_quality(
            tension_curve, times, duration
        )

        # Flat audio should have lower arc quality due to less variation
        assert arc_quality <= 0.8


class TestTensionAnalyze:
    """Tests for full tension analysis."""

    def test_analyze_returns_report(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that analyze returns a TensionReport."""
        audio, sr = tonal_audio
        tis_analyzer = TISAnalyzer()
        tension_analyzer = TensionAnalyzer()

        tis_report = tis_analyzer.analyze(audio, sr)
        report = tension_analyzer.analyze(audio, sr, tis_report)

        assert isinstance(report, TensionReport)

    def test_analyze_report_fields(
        self, tension_resolution_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that report has all expected fields."""
        audio, sr = tension_resolution_audio
        tis_analyzer = TISAnalyzer()
        tension_analyzer = TensionAnalyzer()

        tis_report = tis_analyzer.analyze(audio, sr)
        report = tension_analyzer.analyze(audio, sr, tis_report)

        assert len(report.tension_curve) > 0
        assert isinstance(report.resolution_points, list)
        assert report.resolution_count >= 0
        assert 0.0 <= report.resolution_strength <= 1.0
        assert 0.0 <= report.arc_quality <= 1.0
        assert 0.0 <= report.average_tension <= 1.0
        assert report.tension_variance >= 0.0
        assert report.frame_rate > 0


class TestTensionScore:
    """Tests for tension score computation."""

    def test_compute_score_range(
        self, tension_resolution_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that score is in valid range."""
        audio, sr = tension_resolution_audio
        tis_analyzer = TISAnalyzer()
        tension_analyzer = TensionAnalyzer()

        tis_report = tis_analyzer.analyze(audio, sr)
        report = tension_analyzer.analyze(audio, sr, tis_report)
        duration = len(audio) / sr

        score = tension_analyzer.compute_score(report, duration)

        assert 0.0 <= score <= 1.0

    def test_compute_score_empty_report(self) -> None:
        """Test score computation with empty report."""
        analyzer = TensionAnalyzer()
        report = TensionReport()

        score = analyzer.compute_score(report, duration=5.0)

        assert 0.0 <= score <= 1.0
