"""Tests for StructureAnalyzer."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.musical_coherence.config import StructureConfig
from decrescendo.musicritic.dimensions.musical_coherence.exceptions import (
    StructureAnalysisError,
)
from decrescendo.musicritic.dimensions.musical_coherence.structure import (
    StructureAnalyzer,
    StructureReport,
)


class TestStructureConfig:
    """Test StructureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StructureConfig()
        assert config.min_section_duration == 4.0
        assert config.novelty_threshold == 0.3
        assert config.hop_length == 512
        assert config.n_fft == 2048

    def test_custom_config(self):
        """Test custom configuration values."""
        config = StructureConfig(
            min_section_duration=8.0,
            novelty_threshold=0.5,
        )
        assert config.min_section_duration == 8.0
        assert config.novelty_threshold == 0.5


class TestStructureReport:
    """Test StructureReport dataclass."""

    def test_default_report(self):
        """Test default report values."""
        report = StructureReport()
        assert report.section_count == 0
        assert report.sections == []
        assert report.boundary_timestamps == []
        assert report.repetition_ratio == 0.0
        assert report.structure_clarity == 0.0

    def test_custom_report(self):
        """Test custom report values."""
        report = StructureReport(
            section_count=3,
            sections=[(0.0, 4.0, "A"), (4.0, 8.0, "B"), (8.0, 12.0, "A")],
            boundary_timestamps=[4.0, 8.0],
            repetition_ratio=0.33,
            structure_clarity=0.8,
        )
        assert report.section_count == 3
        assert len(report.sections) == 3
        assert len(report.boundary_timestamps) == 2


class TestStructureAnalyzer:
    """Test StructureAnalyzer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        analyzer = StructureAnalyzer()
        assert analyzer.config is not None
        assert analyzer.config.min_section_duration == 4.0

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = StructureConfig(min_section_duration=8.0)
        analyzer = StructureAnalyzer(config=config)
        assert analyzer.config.min_section_duration == 8.0

    def test_compute_features(self, sample_audio):
        """Test feature computation."""
        audio, sample_rate = sample_audio
        analyzer = StructureAnalyzer()

        features = analyzer.compute_features(audio, sample_rate)

        assert features.shape[0] == 25  # 13 MFCCs + 12 chroma
        assert features.shape[1] > 0  # Some frames

    def test_compute_self_similarity(self, sample_audio):
        """Test self-similarity computation."""
        audio, sample_rate = sample_audio
        analyzer = StructureAnalyzer()

        features = analyzer.compute_features(audio, sample_rate)
        similarity = analyzer.compute_self_similarity(features)

        # Should be square matrix
        assert similarity.shape[0] == similarity.shape[1]
        assert similarity.shape[0] == features.shape[1]

        # Diagonal should be 1.0 (self-similarity)
        np.testing.assert_array_almost_equal(
            np.diag(similarity), np.ones(similarity.shape[0])
        )

        # Should be symmetric
        np.testing.assert_array_almost_equal(similarity, similarity.T)

    def test_detect_boundaries(self, structured_audio):
        """Test boundary detection on structured audio."""
        audio, sample_rate = structured_audio
        analyzer = StructureAnalyzer()

        features = analyzer.compute_features(audio, sample_rate)
        similarity = analyzer.compute_self_similarity(features)
        boundaries = analyzer.detect_boundaries(similarity, sample_rate)

        assert isinstance(boundaries, list)
        # Structured audio should have some boundaries
        # (though exact number depends on detection sensitivity)

    def test_detect_boundaries_short_audio(self, short_audio):
        """Test boundary detection on short audio."""
        audio, sample_rate = short_audio
        analyzer = StructureAnalyzer()

        features = analyzer.compute_features(audio, sample_rate)
        similarity = analyzer.compute_self_similarity(features)
        boundaries = analyzer.detect_boundaries(similarity, sample_rate)

        # Short audio may not have detectable boundaries
        assert isinstance(boundaries, list)

    def test_label_sections(self, structured_audio):
        """Test section labeling."""
        audio, sample_rate = structured_audio
        audio_duration = len(audio) / sample_rate
        analyzer = StructureAnalyzer()

        features = analyzer.compute_features(audio, sample_rate)
        similarity = analyzer.compute_self_similarity(features)
        boundaries = [4.0, 8.0, 12.0]  # Known boundaries

        sections = analyzer.label_sections(
            similarity, boundaries, sample_rate, audio_duration
        )

        assert len(sections) == 4  # 3 boundaries = 4 sections
        for start, end, label in sections:
            assert start < end
            assert isinstance(label, str)
            assert len(label) == 1  # Single letter label

    def test_label_sections_no_boundaries(self, sample_audio):
        """Test section labeling with no boundaries."""
        audio, sample_rate = sample_audio
        audio_duration = len(audio) / sample_rate
        analyzer = StructureAnalyzer()

        features = analyzer.compute_features(audio, sample_rate)
        similarity = analyzer.compute_self_similarity(features)

        sections = analyzer.label_sections(
            similarity, [], sample_rate, audio_duration
        )

        assert len(sections) == 1
        assert sections[0][0] == 0.0
        assert sections[0][1] == audio_duration
        assert sections[0][2] == "A"

    def test_compute_repetition_ratio_repeated(self):
        """Test repetition ratio for repeated sections."""
        analyzer = StructureAnalyzer()

        # ABAB pattern
        sections = [
            (0.0, 4.0, "A"),
            (4.0, 8.0, "B"),
            (8.0, 12.0, "A"),
            (12.0, 16.0, "B"),
        ]

        ratio = analyzer.compute_repetition_ratio(sections)

        # 4 sections, 2 unique labels = 0.5 repetition
        assert ratio == 0.5

    def test_compute_repetition_ratio_no_repetition(self):
        """Test repetition ratio with no repetition."""
        analyzer = StructureAnalyzer()

        # ABCD pattern - all unique
        sections = [
            (0.0, 4.0, "A"),
            (4.0, 8.0, "B"),
            (8.0, 12.0, "C"),
            (12.0, 16.0, "D"),
        ]

        ratio = analyzer.compute_repetition_ratio(sections)

        # 4 sections, 4 unique = 0 repetition
        assert ratio == 0.0

    def test_compute_repetition_ratio_all_same(self):
        """Test repetition ratio when all sections are same."""
        analyzer = StructureAnalyzer()

        # AAAA pattern
        sections = [
            (0.0, 4.0, "A"),
            (4.0, 8.0, "A"),
            (8.0, 12.0, "A"),
            (12.0, 16.0, "A"),
        ]

        ratio = analyzer.compute_repetition_ratio(sections)

        # 4 sections, 1 unique = 0.75 repetition
        assert ratio == 0.75

    def test_compute_repetition_ratio_single_section(self):
        """Test repetition ratio for single section."""
        analyzer = StructureAnalyzer()

        sections = [(0.0, 4.0, "A")]

        ratio = analyzer.compute_repetition_ratio(sections)

        assert ratio == 0.0

    def test_compute_structure_clarity(self, structured_audio):
        """Test structure clarity computation."""
        audio, sample_rate = structured_audio
        analyzer = StructureAnalyzer()

        features = analyzer.compute_features(audio, sample_rate)
        similarity = analyzer.compute_self_similarity(features)
        boundaries = analyzer.detect_boundaries(similarity, sample_rate)

        clarity = analyzer.compute_structure_clarity(
            similarity, boundaries, sample_rate
        )

        assert 0.0 <= clarity <= 1.0

    def test_compute_structure_clarity_no_boundaries(self, sample_audio):
        """Test structure clarity with no boundaries."""
        audio, sample_rate = sample_audio
        analyzer = StructureAnalyzer()

        features = analyzer.compute_features(audio, sample_rate)
        similarity = analyzer.compute_self_similarity(features)

        clarity = analyzer.compute_structure_clarity(similarity, [], sample_rate)

        # Should have some value based on overall coherence
        assert 0.0 <= clarity <= 1.0

    def test_analyze_structured_audio(self, structured_audio):
        """Test full analysis on structured audio."""
        audio, sample_rate = structured_audio
        analyzer = StructureAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, StructureReport)
        assert report.section_count >= 1
        assert len(report.sections) == report.section_count
        assert 0.0 <= report.repetition_ratio <= 1.0
        assert 0.0 <= report.structure_clarity <= 1.0
        assert report.self_similarity_matrix.shape[0] > 0

    def test_analyze_sample_audio(self, sample_audio):
        """Test analysis on simple audio."""
        audio, sample_rate = sample_audio
        analyzer = StructureAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, StructureReport)
        assert report.section_count >= 1

    def test_analyze_silence(self, silence):
        """Test analysis on silence."""
        audio, sample_rate = silence
        analyzer = StructureAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, StructureReport)

    def test_analyze_white_noise(self, white_noise):
        """Test analysis on white noise."""
        audio, sample_rate = white_noise
        analyzer = StructureAnalyzer()

        report = analyzer.analyze(audio, sample_rate)

        assert isinstance(report, StructureReport)
        # Noise should have low structure clarity

    def test_compute_score_excellent(self):
        """Test score computation for excellent structure."""
        analyzer = StructureAnalyzer()
        report = StructureReport(
            section_count=4,
            sections=[
                (0.0, 8.0, "A"),
                (8.0, 16.0, "B"),
                (16.0, 24.0, "A"),
                (24.0, 32.0, "B"),
            ],
            repetition_ratio=0.5,
            structure_clarity=0.9,
        )

        score = analyzer.compute_score(report, audio_duration=32.0)

        assert score > 0.7

    def test_compute_score_poor(self):
        """Test score computation for poor structure."""
        analyzer = StructureAnalyzer()
        report = StructureReport(
            section_count=1,
            sections=[(0.0, 30.0, "A")],
            repetition_ratio=0.0,
            structure_clarity=0.1,
        )

        score = analyzer.compute_score(report, audio_duration=30.0)

        # Low clarity and no repetition = lower score
        assert score < 0.6

    def test_compute_score_moderate(self):
        """Test score computation for moderate structure."""
        analyzer = StructureAnalyzer()
        report = StructureReport(
            section_count=2,
            sections=[(0.0, 15.0, "A"), (15.0, 30.0, "B")],
            repetition_ratio=0.0,
            structure_clarity=0.5,
        )

        score = analyzer.compute_score(report, audio_duration=30.0)

        assert 0.3 < score < 0.8

    def test_compute_score_too_many_sections(self):
        """Test score with too many sections."""
        analyzer = StructureAnalyzer()
        # 10 sections in 30 seconds = too fragmented
        sections = [(i * 3.0, (i + 1) * 3.0, chr(65 + i % 26)) for i in range(10)]
        report = StructureReport(
            section_count=10,
            sections=sections,
            repetition_ratio=0.0,
            structure_clarity=0.8,
        )

        score = analyzer.compute_score(report, audio_duration=30.0)

        # High clarity but too fragmented
        assert score < 0.9

    def test_compute_score_bounds(self):
        """Test that score is always in valid range."""
        analyzer = StructureAnalyzer()

        reports = [
            StructureReport(section_count=4, repetition_ratio=0.5, structure_clarity=1.0),
            StructureReport(section_count=0, repetition_ratio=0.0, structure_clarity=0.0),
            StructureReport(section_count=2, repetition_ratio=0.5, structure_clarity=0.5),
        ]

        for report in reports:
            score = analyzer.compute_score(report, audio_duration=30.0)
            assert 0.0 <= score <= 1.0
