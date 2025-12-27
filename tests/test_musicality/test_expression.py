"""Tests for Expression analyzer."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.musicality import (
    ExpressionAnalyzer,
    ExpressionConfig,
    ExpressionReport,
    ExpressionAnalysisError,
)


class TestExpressionAnalyzerAttributes:
    """Tests for ExpressionAnalyzer attributes and initialization."""

    def test_default_config(self) -> None:
        """Test that default config is used when not provided."""
        analyzer = ExpressionAnalyzer()
        assert isinstance(analyzer.config, ExpressionConfig)

    def test_custom_config(self) -> None:
        """Test that custom config is used."""
        config = ExpressionConfig(rms_frame_length=1024)
        analyzer = ExpressionAnalyzer(config=config)
        assert analyzer.config.rms_frame_length == 1024


class TestLoudnessCurve:
    """Tests for loudness curve computation."""

    def test_compute_loudness_curve_shape(
        self, dynamic_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that loudness curve has correct shape."""
        audio, sr = dynamic_audio
        analyzer = ExpressionAnalyzer()

        loudness_db, frame_rate = analyzer.compute_loudness_curve(audio, sr)

        assert len(loudness_db) > 0
        assert frame_rate > 0

    def test_compute_loudness_curve_silent(
        self, silence_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test loudness curve for silent audio."""
        audio, sr = silence_audio
        analyzer = ExpressionAnalyzer()

        loudness_db, _ = analyzer.compute_loudness_curve(audio, sr)

        # Silent audio should have very low dB values
        assert np.all(loudness_db < -50)


class TestDynamicRange:
    """Tests for dynamic range computation."""

    def test_dynamic_range_positive(
        self, dynamic_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that dynamic range is non-negative."""
        audio, sr = dynamic_audio
        analyzer = ExpressionAnalyzer()

        loudness_db, _ = analyzer.compute_loudness_curve(audio, sr)
        dr = analyzer.compute_dynamic_range(loudness_db)

        assert dr >= 0.0

    def test_flat_audio_low_range(
        self, flat_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that flat audio has low dynamic range."""
        audio, sr = flat_audio
        analyzer = ExpressionAnalyzer()

        loudness_db, _ = analyzer.compute_loudness_curve(audio, sr)
        dr = analyzer.compute_dynamic_range(loudness_db)

        # Flat audio should have low dynamic range
        assert dr < 10.0

    def test_dynamic_audio_higher_range(
        self,
        dynamic_audio: tuple[np.ndarray, int],
        flat_audio: tuple[np.ndarray, int],
    ) -> None:
        """Test that dynamic audio has higher range than flat."""
        dynamic, sr_dyn = dynamic_audio
        flat, sr_flat = flat_audio
        analyzer = ExpressionAnalyzer()

        dyn_loudness, _ = analyzer.compute_loudness_curve(dynamic, sr_dyn)
        flat_loudness, _ = analyzer.compute_loudness_curve(flat, sr_flat)

        dyn_dr = analyzer.compute_dynamic_range(dyn_loudness)
        flat_dr = analyzer.compute_dynamic_range(flat_loudness)

        assert dyn_dr > flat_dr


class TestDynamicVariation:
    """Tests for dynamic variation computation."""

    def test_dynamic_variation_range(
        self, dynamic_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that dynamic variation is in valid range."""
        audio, sr = dynamic_audio
        analyzer = ExpressionAnalyzer()

        loudness_db, _ = analyzer.compute_loudness_curve(audio, sr)
        variation = analyzer.compute_dynamic_variation(loudness_db)

        assert 0.0 <= variation <= 1.0

    def test_flat_audio_low_variation(
        self, flat_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that flat audio has low dynamic variation."""
        audio, sr = flat_audio
        analyzer = ExpressionAnalyzer()

        loudness_db, _ = analyzer.compute_loudness_curve(audio, sr)
        variation = analyzer.compute_dynamic_variation(loudness_db)

        # Flat audio should have low variation
        assert variation < 0.3


class TestDynamicsEvents:
    """Tests for dynamics events detection."""

    def test_detect_events_returns_counts(
        self, dynamic_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that dynamics event detection returns counts."""
        audio, sr = dynamic_audio
        analyzer = ExpressionAnalyzer()

        loudness_db, frame_rate = analyzer.compute_loudness_curve(audio, sr)
        crescendo, decrescendo = analyzer.detect_dynamics_events(
            loudness_db, frame_rate
        )

        assert isinstance(crescendo, int)
        assert isinstance(decrescendo, int)
        assert crescendo >= 0
        assert decrescendo >= 0


class TestExpressionAnalyze:
    """Tests for full expression analysis."""

    def test_analyze_returns_report(
        self, dynamic_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that analyze returns an ExpressionReport."""
        audio, sr = dynamic_audio
        analyzer = ExpressionAnalyzer()

        report = analyzer.analyze(audio, sr)

        assert isinstance(report, ExpressionReport)

    def test_analyze_report_fields(
        self, dynamic_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that report has all expected fields."""
        audio, sr = dynamic_audio
        analyzer = ExpressionAnalyzer()

        report = analyzer.analyze(audio, sr)

        assert report.dynamic_range_db >= 0.0
        assert 0.0 <= report.dynamic_variation <= 1.0
        assert len(report.loudness_curve) > 0
        assert report.crescendo_count >= 0
        assert report.decrescendo_count >= 0
        assert report.frame_rate > 0


class TestExpressionScore:
    """Tests for expression score computation."""

    def test_compute_score_range(
        self, dynamic_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that score is in valid range."""
        audio, sr = dynamic_audio
        analyzer = ExpressionAnalyzer()

        report = analyzer.analyze(audio, sr)
        score = analyzer.compute_score(report)

        assert 0.0 <= score <= 1.0

    def test_dynamic_higher_score_than_flat(
        self,
        dynamic_audio: tuple[np.ndarray, int],
        flat_audio: tuple[np.ndarray, int],
    ) -> None:
        """Test that dynamic audio scores higher than flat."""
        dynamic, sr_dyn = dynamic_audio
        flat, sr_flat = flat_audio
        analyzer = ExpressionAnalyzer()

        dyn_report = analyzer.analyze(dynamic, sr_dyn)
        flat_report = analyzer.analyze(flat, sr_flat)

        dyn_score = analyzer.compute_score(dyn_report)
        flat_score = analyzer.compute_score(flat_report)

        # Dynamic audio should score higher for expression
        assert dyn_score >= flat_score

    def test_compute_score_empty_report(self) -> None:
        """Test score computation with empty report."""
        analyzer = ExpressionAnalyzer()
        report = ExpressionReport()

        score = analyzer.compute_score(report)

        assert 0.0 <= score <= 1.0
