"""Tests for Musicality configuration."""

import pytest
from dataclasses import FrozenInstanceError

from decrescendo.musicritic.dimensions.musicality import (
    ExpressionConfig,
    MusicalityConfig,
    TensionConfig,
    TISConfig,
)


class TestTISConfig:
    """Tests for TISConfig."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = TISConfig()

        assert config.hop_length == 2048
        assert config.smoothing_window == 5
        assert config.min_chroma_energy == 0.01

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = TISConfig()

        with pytest.raises(FrozenInstanceError):
            config.hop_length = 1024  # type: ignore


class TestTensionConfig:
    """Tests for TensionConfig."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = TensionConfig()

        assert config.resolution_threshold == 0.15
        assert config.cadence_window_sec == 2.0
        assert config.min_tension_drop == 0.2
        assert config.smoothing_window == 5

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = TensionConfig()

        with pytest.raises(FrozenInstanceError):
            config.resolution_threshold = 0.5  # type: ignore


class TestExpressionConfig:
    """Tests for ExpressionConfig."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = ExpressionConfig()

        assert config.rms_frame_length == 2048
        assert config.rms_hop_length == 512
        assert config.min_dynamic_range_db == 6.0
        assert config.max_dynamic_range_db == 40.0

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = ExpressionConfig()

        with pytest.raises(FrozenInstanceError):
            config.rms_frame_length = 1024  # type: ignore


class TestMusicalityConfig:
    """Tests for MusicalityConfig."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = MusicalityConfig()

        assert config.tis_weight == 0.40
        assert config.tension_weight == 0.35
        assert config.expression_weight == 0.25
        assert config.excellent_threshold == 0.80
        assert config.good_threshold == 0.65
        assert config.moderate_threshold == 0.45
        assert config.min_audio_duration == 3.0

    def test_weights_sum_to_one(self) -> None:
        """Test that default weights sum to 1.0."""
        config = MusicalityConfig()

        total_weight = config.tis_weight + config.tension_weight + config.expression_weight
        assert abs(total_weight - 1.0) < 1e-6

    def test_nested_configs(self) -> None:
        """Test that nested configs are created correctly."""
        config = MusicalityConfig()

        assert isinstance(config.tis_config, TISConfig)
        assert isinstance(config.tension_config, TensionConfig)
        assert isinstance(config.expression_config, ExpressionConfig)

    def test_custom_nested_config(self) -> None:
        """Test custom nested config values."""
        custom_tis = TISConfig(hop_length=1024)
        config = MusicalityConfig(tis_config=custom_tis)

        assert config.tis_config.hop_length == 1024

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = MusicalityConfig()

        with pytest.raises(FrozenInstanceError):
            config.tis_weight = 0.5  # type: ignore
