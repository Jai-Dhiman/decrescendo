"""Flax models for the Output Classifier."""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import AudioEncoderConfig, OutputClassifierConfig, SpeakerConfig


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation."""

    features: int
    kernel_size: int = 7
    stride: int = 2
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding="SAME",
            dtype=self.dtype,
        )(x)
        x = nn.BatchNorm(use_running_average=not train, dtype=self.dtype)(x)
        x = nn.gelu(x)
        return x


class AudioEncoder(nn.Module):
    """CNN-based audio encoder.

    Extracts embeddings from raw audio waveforms using a stack of
    1D convolutional layers. This is a simplified encoder that can
    be replaced with pretrained models (MERT, WavLM) when available.

    Input: (batch, samples) raw audio
    Output: (batch, embedding_dim) audio embeddings
    """

    config: AudioEncoderConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        audio: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            audio: Raw audio of shape (batch, samples)
            train: Whether in training mode (affects BatchNorm, Dropout)

        Returns:
            Audio embeddings of shape (batch, embedding_dim)
        """
        # Add channel dimension: (batch, samples) -> (batch, samples, 1)
        x = audio[..., None]

        # Convolutional layers with increasing channels
        channels = self.config.base_channels
        for i in range(self.config.num_conv_layers):
            # Double channels every 2 layers
            if i > 0 and i % 2 == 0:
                channels *= self.config.channel_multiplier

            x = ConvBlock(
                features=channels,
                kernel_size=self.config.kernel_size,
                stride=self.config.stride,
                dtype=self.dtype,
            )(x, train=train)

        # Global average pooling: (batch, time, channels) -> (batch, channels)
        x = jnp.mean(x, axis=1)

        # Project to embedding dimension
        x = nn.Dense(
            self.config.embedding_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)

        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=not train)

        return x


class SpeakerEncoder(nn.Module):
    """Simplified speaker encoder for voice embeddings.

    Extracts speaker embeddings that can be compared against
    protected voice fingerprints. In production, this would be
    replaced with ECAPA-TDNN or similar.

    Input: (batch, samples) raw audio
    Output: (batch, speaker_embedding_dim) speaker embeddings
    """

    config: SpeakerConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        audio: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            audio: Raw audio of shape (batch, samples)
            train: Whether in training mode

        Returns:
            Speaker embeddings of shape (batch, embedding_dim)
        """
        # Add channel dimension
        x = audio[..., None]

        # Convolutional layers
        channels = self.config.base_channels
        for i in range(self.config.num_conv_layers):
            if i > 0:
                channels *= 2

            x = ConvBlock(
                features=channels,
                kernel_size=5,
                stride=2,
                dtype=self.dtype,
            )(x, train=train)

        # Statistics pooling: mean and std
        mean = jnp.mean(x, axis=1)
        std = jnp.std(x, axis=1)
        x = jnp.concatenate([mean, std], axis=-1)

        # Project to speaker embedding
        x = nn.Dense(
            self.config.embedding_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)

        # L2 normalize for cosine similarity comparison
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

        return x


class HarmClassifier(nn.Module):
    """Multi-label harm classification head.

    Takes audio embeddings and predicts probability scores for
    each of the 7 harm categories.

    Input: (batch, embedding_dim) audio embeddings
    Output: (batch, num_categories) harm logits
    """

    num_categories: int = 7
    hidden_dim: int = 256
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        embeddings: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            embeddings: Audio embeddings of shape (batch, embedding_dim)
            train: Whether in training mode

        Returns:
            Harm logits of shape (batch, num_categories)
        """
        x = nn.Dense(
            self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(embeddings)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        x = nn.Dense(
            self.hidden_dim // 2,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # Output logits for each harm category
        logits = nn.Dense(
            self.num_categories,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)

        return logits


class OutputClassifierModel(nn.Module):
    """Complete Output Classifier model.

    Combines audio encoding, speaker embedding, and harm classification
    into a single model for audio safety analysis.

    Input: (batch, samples) raw audio
    Output: Dictionary with harm logits and speaker embeddings
    """

    config: OutputClassifierConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        audio: jnp.ndarray,
        train: bool = True,
    ) -> dict[str, jnp.ndarray]:
        """Forward pass.

        Args:
            audio: Raw audio of shape (batch, samples)
            train: Whether in training mode

        Returns:
            Dictionary with:
            - harm_logits: (batch, num_harm_categories)
            - audio_embeddings: (batch, embedding_dim)
            - speaker_embeddings: (batch, speaker_embedding_dim)
        """
        # Audio encoder for content analysis
        audio_encoder = AudioEncoder(
            config=self.config.audio_encoder,
            dtype=self.dtype,
            name="audio_encoder",
        )
        audio_embeddings = audio_encoder(audio, train=train)

        # Speaker encoder for voice analysis
        speaker_encoder = SpeakerEncoder(
            config=self.config.speaker,
            dtype=self.dtype,
            name="speaker_encoder",
        )
        speaker_embeddings = speaker_encoder(audio, train=train)

        # Harm classifier
        harm_classifier = HarmClassifier(
            num_categories=self.config.num_harm_categories,
            hidden_dim=self.config.classifier_hidden_dim,
            dropout_rate=self.config.classifier_dropout,
            dtype=self.dtype,
            name="harm_classifier",
        )
        harm_logits = harm_classifier(audio_embeddings, train=train)

        return {
            "harm_logits": harm_logits,
            "audio_embeddings": audio_embeddings,
            "speaker_embeddings": speaker_embeddings,
        }


def compute_speaker_similarity(
    embedding1: jnp.ndarray,
    embedding2: jnp.ndarray,
) -> jnp.ndarray:
    """Compute cosine similarity between speaker embeddings.

    Args:
        embedding1: First embedding (batch, dim) or (dim,)
        embedding2: Second embedding (batch, dim) or (dim,)

    Returns:
        Cosine similarity scores
    """
    # Ensure 2D
    if embedding1.ndim == 1:
        embedding1 = embedding1[None, :]
    if embedding2.ndim == 1:
        embedding2 = embedding2[None, :]

    # Normalize (should already be normalized, but ensure)
    e1_norm = embedding1 / (jnp.linalg.norm(embedding1, axis=-1, keepdims=True) + 1e-8)
    e2_norm = embedding2 / (jnp.linalg.norm(embedding2, axis=-1, keepdims=True) + 1e-8)

    # Cosine similarity
    return jnp.sum(e1_norm * e2_norm, axis=-1)


def compare_against_protected_voices(
    query_embedding: jnp.ndarray,
    protected_embeddings: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compare speaker embedding against protected voice database.

    Args:
        query_embedding: Query embedding (dim,) or (batch, dim)
        protected_embeddings: Database of protected embeddings (num_voices, dim)

    Returns:
        Tuple of (max_similarity, best_match_index)
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding[None, :]

    # Compute similarity against all protected voices
    # query: (batch, dim), protected: (num_voices, dim)
    # result: (batch, num_voices)
    similarities = jnp.matmul(
        query_embedding,
        protected_embeddings.T,
    )

    # Find best match for each query
    max_similarity = jnp.max(similarities, axis=-1)
    best_match_index = jnp.argmax(similarities, axis=-1)

    return max_similarity, best_match_index
