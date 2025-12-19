"""Tests for Prompt Adherence configuration."""

import pytest

from decrescendo.musicritic.dimensions.prompt_adherence import (
    CLAPEncoderConfig,
    PromptAdherenceConfig,
)


class TestCLAPEncoderConfig:
    """Tests for CLAPEncoderConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CLAPEncoderConfig()

        assert config.model_name == "laion/larger_clap_music"
        assert config.sample_rate == 48000
        assert config.embedding_dim == 512

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = CLAPEncoderConfig(
            model_name="laion/clap-htsat-unfused",
            sample_rate=44100,
            embedding_dim=256,
        )

        assert config.model_name == "laion/clap-htsat-unfused"
        assert config.sample_rate == 44100
        assert config.embedding_dim == 256


class TestPromptAdherenceConfig:
    """Tests for PromptAdherenceConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PromptAdherenceConfig()

        assert config.strong_adherence_threshold == 0.7
        assert config.moderate_adherence_threshold == 0.5
        assert config.cache_text_embeddings is True
        assert isinstance(config.encoder_config, CLAPEncoderConfig)

    def test_custom_thresholds(self):
        """Test configuration with custom thresholds."""
        config = PromptAdherenceConfig(
            strong_adherence_threshold=0.8,
            moderate_adherence_threshold=0.6,
            cache_text_embeddings=False,
        )

        assert config.strong_adherence_threshold == 0.8
        assert config.moderate_adherence_threshold == 0.6
        assert config.cache_text_embeddings is False

    def test_config_immutability(self):
        """Test that frozen dataclasses prevent modification."""
        config = CLAPEncoderConfig()

        with pytest.raises(AttributeError):
            config.model_name = "new_model.pt"

        config2 = PromptAdherenceConfig()
        with pytest.raises(AttributeError):
            config2.strong_adherence_threshold = 0.9
