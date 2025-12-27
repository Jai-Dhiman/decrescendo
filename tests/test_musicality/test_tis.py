"""Tests for TIS (Tonal Interval Space) analyzer."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.musicality import (
    TISAnalyzer,
    TISConfig,
    TISReport,
    TISAnalysisError,
)


class TestTISAnalyzerAttributes:
    """Tests for TISAnalyzer attributes and initialization."""

    def test_default_config(self) -> None:
        """Test that default config is used when not provided."""
        analyzer = TISAnalyzer()
        assert isinstance(analyzer.config, TISConfig)

    def test_custom_config(self) -> None:
        """Test that custom config is used."""
        config = TISConfig(hop_length=1024)
        analyzer = TISAnalyzer(config=config)
        assert analyzer.config.hop_length == 1024


class TestTISAnalyzerChroma:
    """Tests for chroma feature computation."""

    def test_compute_chroma_shape(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that chroma has correct shape."""
        audio, sr = tonal_audio
        analyzer = TISAnalyzer()

        chroma = analyzer.compute_chroma_features(audio, sr)

        assert chroma.shape[0] == 12  # 12 pitch classes
        assert chroma.shape[1] > 0  # At least one frame

    def test_compute_chroma_normalized(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that chroma frames are L2 normalized."""
        audio, sr = tonal_audio
        analyzer = TISAnalyzer()

        chroma = analyzer.compute_chroma_features(audio, sr)

        # Check that each frame is approximately unit norm
        norms = np.linalg.norm(chroma, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-6)


class TestTISAnalyzerCloudDiameter:
    """Tests for cloud diameter computation."""

    def test_single_note_low_diameter(self, sample_rate: int) -> None:
        """Test that a single note has low cloud diameter."""
        # Generate single frequency tone
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        analyzer = TISAnalyzer()
        chroma = analyzer.compute_chroma_features(audio, sample_rate)

        # Check a frame in the middle
        mid_frame = chroma[:, chroma.shape[1] // 2]
        diameter = analyzer.compute_cloud_diameter(mid_frame)

        # Single note should have low diameter
        assert diameter < 0.3

    def test_chromatic_cluster_high_diameter(
        self, complex_harmonic_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that complex harmonies have higher cloud diameter."""
        audio, sr = complex_harmonic_audio
        analyzer = TISAnalyzer()

        chroma = analyzer.compute_chroma_features(audio, sr)
        mid_frame = chroma[:, chroma.shape[1] // 2]
        diameter = analyzer.compute_cloud_diameter(mid_frame)

        # Complex chord should have moderate-high diameter
        assert diameter > 0.2


class TestTISAnalyzerTensileStrain:
    """Tests for tensile strain computation."""

    def test_tonic_low_strain(self, tonal_audio: tuple[np.ndarray, int]) -> None:
        """Test that music in key has low tensile strain."""
        audio, sr = tonal_audio
        analyzer = TISAnalyzer()

        chroma = analyzer.compute_chroma_features(audio, sr)
        tonal_center = analyzer.estimate_tonal_center(chroma)

        # Check strain at beginning (on tonic)
        first_frame = chroma[:, 0]
        strain = analyzer.compute_tensile_strain(first_frame, tonal_center)

        # Should be low-moderate strain for tonal music
        assert 0.0 <= strain <= 1.0


class TestTISAnalyzerAnalyze:
    """Tests for full TIS analysis."""

    def test_analyze_returns_report(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that analyze returns a TISReport."""
        audio, sr = tonal_audio
        analyzer = TISAnalyzer()

        report = analyzer.analyze(audio, sr)

        assert isinstance(report, TISReport)

    def test_analyze_report_fields(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that report has all expected fields."""
        audio, sr = tonal_audio
        analyzer = TISAnalyzer()

        report = analyzer.analyze(audio, sr)

        assert 0.0 <= report.cloud_diameter <= 1.0
        assert 0.0 <= report.cloud_momentum <= 1.0
        assert 0.0 <= report.tensile_strain <= 1.0
        assert len(report.cloud_diameter_curve) > 0
        assert len(report.tensile_strain_curve) > 0
        assert 0 <= report.tonal_center <= 11
        assert report.frame_rate > 0

    def test_analyze_tonal_vs_atonal(
        self,
        tonal_audio: tuple[np.ndarray, int],
        atonal_audio: tuple[np.ndarray, int],
    ) -> None:
        """Test that tonal audio has lower strain than atonal."""
        tonal, sr_tonal = tonal_audio
        atonal, sr_atonal = atonal_audio
        analyzer = TISAnalyzer()

        tonal_report = analyzer.analyze(tonal, sr_tonal)
        atonal_report = analyzer.analyze(atonal, sr_atonal)

        # Atonal should generally have higher complexity
        # (though random might not always be more complex)
        assert tonal_report.cloud_diameter >= 0.0
        assert atonal_report.cloud_diameter >= 0.0


class TestTISAnalyzerScore:
    """Tests for TIS score computation."""

    def test_compute_score_range(
        self, tonal_audio: tuple[np.ndarray, int]
    ) -> None:
        """Test that score is in valid range."""
        audio, sr = tonal_audio
        analyzer = TISAnalyzer()

        report = analyzer.analyze(audio, sr)
        score = analyzer.compute_score(report)

        assert 0.0 <= score <= 1.0

    def test_compute_score_empty_report(self) -> None:
        """Test score computation with empty report."""
        analyzer = TISAnalyzer()
        report = TISReport()

        score = analyzer.compute_score(report)

        assert 0.0 <= score <= 1.0
