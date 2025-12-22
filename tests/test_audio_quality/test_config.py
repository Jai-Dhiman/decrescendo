"""Tests for Audio Quality configuration."""

import pytest
from dataclasses import FrozenInstanceError

from decrescendo.musicritic.dimensions.audio_quality import (
    ArtifactDetectionConfig,
    AudioQualityConfig,
    LoudnessConfig,
    PerceptualConfig,
)


class TestArtifactDetectionConfig:
    """Tests for ArtifactDetectionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ArtifactDetectionConfig()
        assert config.click_threshold == 0.1
        assert config.clipping_threshold == 0.99
        assert config.min_clipping_samples == 3
        assert config.spectral_flux_threshold == 2.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ArtifactDetectionConfig(
            click_threshold=0.2,
            clipping_threshold=0.95,
            min_clipping_samples=5,
        )
        assert config.click_threshold == 0.2
        assert config.clipping_threshold == 0.95
        assert config.min_clipping_samples == 5

    def test_immutability(self):
        """Test that config is frozen."""
        config = ArtifactDetectionConfig()
        with pytest.raises(FrozenInstanceError):
            config.click_threshold = 0.5


class TestLoudnessConfig:
    """Tests for LoudnessConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoudnessConfig()
        assert config.target_lufs == -14.0
        assert config.max_true_peak_dbtp == -1.0
        assert config.min_lra == 4.0
        assert config.max_lra == 20.0
        assert config.block_size == 0.4

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LoudnessConfig(
            target_lufs=-16.0,
            max_true_peak_dbtp=-2.0,
        )
        assert config.target_lufs == -16.0
        assert config.max_true_peak_dbtp == -2.0


class TestPerceptualConfig:
    """Tests for PerceptualConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PerceptualConfig()
        assert config.target_sample_rate == 44100
        assert len(config.frequency_bands) == 4
        assert config.frequency_bands[0] == (20, 250)
        assert len(config.ideal_balance) == 4
        assert sum(config.ideal_balance) == 1.0

    def test_centroid_range(self):
        """Test centroid range configuration."""
        config = PerceptualConfig()
        assert config.min_centroid_hz == 500.0
        assert config.max_centroid_hz == 4000.0


class TestAudioQualityConfig:
    """Tests for AudioQualityConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AudioQualityConfig()
        assert isinstance(config.artifact_config, ArtifactDetectionConfig)
        assert isinstance(config.loudness_config, LoudnessConfig)
        assert isinstance(config.perceptual_config, PerceptualConfig)

    def test_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        config = AudioQualityConfig()
        total = (
            config.artifact_weight
            + config.loudness_weight
            + config.perceptual_weight
        )
        assert abs(total - 1.0) < 1e-6

    def test_quality_thresholds(self):
        """Test quality level thresholds."""
        config = AudioQualityConfig()
        assert config.excellent_threshold == 0.85
        assert config.good_threshold == 0.70
        assert config.acceptable_threshold == 0.50
        # Ensure ordering
        assert config.excellent_threshold > config.good_threshold
        assert config.good_threshold > config.acceptable_threshold

    def test_min_audio_duration(self):
        """Test minimum audio duration setting."""
        config = AudioQualityConfig()
        assert config.min_audio_duration == 0.5

    def test_nested_configs_immutable(self):
        """Test that nested configs are also frozen."""
        config = AudioQualityConfig()
        with pytest.raises(FrozenInstanceError):
            config.artifact_config.click_threshold = 0.5
