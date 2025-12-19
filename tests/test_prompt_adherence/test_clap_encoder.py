"""Tests for CLAP encoder wrapper."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.prompt_adherence import (
    CLAPEncoder,
    CLAPEncoderConfig,
)


class TestCLAPEncoderLazyLoading:
    """Tests for lazy loading behavior."""

    def test_encoder_not_loaded_on_init(self):
        """Test that model is not loaded at instantiation."""
        encoder = CLAPEncoder()

        # Model should not be loaded yet
        assert encoder._model is None
        assert encoder.is_loaded is False

    def test_encoder_loaded_on_first_use(self, clap_encoder, sample_prompt):
        """Test that model is loaded when first used."""
        # Accessing model property should trigger loading
        _ = clap_encoder.encode_text(sample_prompt)

        assert clap_encoder.is_loaded is True


class TestCLAPEncoderTextEncoding:
    """Tests for text encoding."""

    def test_encode_text_single(self, clap_encoder, sample_prompt):
        """Test encoding a single text prompt."""
        embedding = clap_encoder.encode_text(sample_prompt)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert embedding.shape[0] == clap_encoder.config.embedding_dim

    def test_encode_text_batch(self, clap_encoder):
        """Test encoding multiple text prompts."""
        prompts = ["electronic music", "classical piano", "rock guitar"]
        embeddings = clap_encoder.encode_text(prompts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == len(prompts)
        assert embeddings.shape[1] == clap_encoder.config.embedding_dim

    def test_encode_text_caching(self, clap_encoder):
        """Test that text embeddings are cached."""
        prompt = "test caching prompt"

        # First call - should compute and cache
        embedding1 = clap_encoder.encode_text(prompt, use_cache=True)
        assert prompt in clap_encoder._text_cache

        # Second call - should return cached
        embedding2 = clap_encoder.encode_text(prompt, use_cache=True)

        # Should be the same object (cached)
        np.testing.assert_array_equal(embedding1, embedding2)

        # Clear cache and verify
        clap_encoder.clear_cache()
        assert prompt not in clap_encoder._text_cache


class TestCLAPEncoderAudioEncoding:
    """Tests for audio encoding."""

    def test_encode_audio_mono(self, clap_encoder, sample_audio_48k):
        """Test encoding mono audio."""
        audio, sample_rate = sample_audio_48k
        embedding = clap_encoder.encode_audio(audio, sample_rate)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert embedding.shape[0] == clap_encoder.config.embedding_dim

    def test_encode_audio_resampling(self, clap_encoder, sample_audio_44k):
        """Test that audio at different sample rates is resampled."""
        audio, sample_rate = sample_audio_44k

        # Should work even though sample rate is 44100, not 48000
        embedding = clap_encoder.encode_audio(audio, sample_rate)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert embedding.shape[0] == clap_encoder.config.embedding_dim


class TestCLAPEncoderSimilarity:
    """Tests for similarity computation."""

    def test_compute_similarity_range(self, clap_encoder, sample_audio_48k, sample_prompt):
        """Test that similarity is in expected range."""
        audio, sample_rate = sample_audio_48k

        text_embedding = clap_encoder.encode_text(sample_prompt)
        audio_embedding = clap_encoder.encode_audio(audio, sample_rate)

        similarity = clap_encoder.compute_similarity(text_embedding, audio_embedding)

        # Cosine similarity should be between -1 and 1
        assert -1.0 <= similarity <= 1.0

    def test_compute_similarity_self(self, clap_encoder, sample_prompt):
        """Test that similarity of identical embeddings is 1."""
        embedding = clap_encoder.encode_text(sample_prompt)

        similarity = clap_encoder.compute_similarity(embedding, embedding)

        # Self-similarity should be very close to 1
        assert similarity > 0.99


class TestCLAPEncoderFactoryMethod:
    """Tests for factory method."""

    def test_from_music_checkpoint(self):
        """Test creating encoder from factory method."""
        encoder = CLAPEncoder.from_music_checkpoint()

        assert isinstance(encoder, CLAPEncoder)
        assert encoder.config.model_name == "laion/larger_clap_music"
