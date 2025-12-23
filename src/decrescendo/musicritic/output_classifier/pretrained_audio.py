"""Pretrained audio encoder loading and adaptation for Output Classifier."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from .config import OutputClassifierConfig


class PretrainedAudioLoadError(Exception):
    """Raised when pretrained audio weights cannot be loaded."""

    pass


@dataclass(frozen=True)
class PretrainedAudioConfig:
    """Configuration for pretrained audio encoder.

    Attributes:
        model_name: HuggingFace model identifier (e.g., "m-a-p/MERT-v1-330M")
        output_dim: Output embedding dimension from pretrained model
        freeze_encoder: Whether to freeze pretrained weights during fine-tuning
        use_weighted_layer_sum: Whether to use weighted sum of all layers
        layer_weights_trainable: Whether layer weights are trainable
        target_sample_rate: Expected sample rate for pretrained model
    """

    model_name: str = "m-a-p/MERT-v1-330M"
    output_dim: int = 1024  # MERT hidden size
    freeze_encoder: bool = True
    use_weighted_layer_sum: bool = True
    layer_weights_trainable: bool = True
    target_sample_rate: int = 24000


def load_mert_encoder(
    model_name: str = "m-a-p/MERT-v1-330M",
    trust_remote_code: bool = True,
) -> tuple[Any, Any, int]:
    """Load MERT encoder from HuggingFace.

    MERT (Music undERstanding Transformer) is a music-specific audio encoder
    pretrained on 160K hours of music. It outputs 1024-dim embeddings.

    Args:
        model_name: HuggingFace model identifier
        trust_remote_code: Whether to trust remote code for custom models

    Returns:
        Tuple of (model, processor, hidden_size)

    Raises:
        PretrainedAudioLoadError: If model cannot be loaded
    """
    try:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
    except ImportError as e:
        raise PretrainedAudioLoadError(
            "transformers library required: pip install transformers"
        ) from e

    try:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        hidden_size = model.config.hidden_size

        return model, processor, hidden_size

    except Exception as e:
        raise PretrainedAudioLoadError(f"Failed to load MERT model '{model_name}': {e}") from e


def load_wavlm_encoder(
    model_name: str = "microsoft/wavlm-base-plus",
) -> tuple[Any, Any, int]:
    """Load WavLM encoder from HuggingFace.

    WavLM is optimized for speaker-related tasks and outputs
    768-dim (base) or 1024-dim (large) embeddings.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Tuple of (model, processor, hidden_size)

    Raises:
        PretrainedAudioLoadError: If model cannot be loaded
    """
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as e:
        raise PretrainedAudioLoadError(
            "transformers library required: pip install transformers"
        ) from e

    try:
        model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        hidden_size = model.config.hidden_size

        return model, processor, hidden_size

    except Exception as e:
        raise PretrainedAudioLoadError(f"Failed to load WavLM model '{model_name}': {e}") from e


class PretrainedAudioEncoderWrapper:
    """Wrapper for pretrained PyTorch audio encoders.

    This class wraps a PyTorch pretrained model (MERT or WavLM) and provides
    methods for extracting embeddings that can be used with JAX/Flax models.

    The wrapper handles:
    - Audio preprocessing using the model's processor
    - Forward pass through the pretrained encoder
    - Conversion of PyTorch outputs to NumPy/JAX arrays
    - Optional weighted layer sum

    Example:
        >>> wrapper = PretrainedAudioEncoderWrapper.from_mert()
        >>> embeddings = wrapper.extract_embeddings(audio, sample_rate=24000)
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        hidden_size: int,
        config: PretrainedAudioConfig,
    ) -> None:
        """Initialize wrapper.

        Args:
            model: PyTorch model
            processor: HuggingFace processor/feature extractor
            hidden_size: Model hidden dimension
            config: Pretrained encoder configuration
        """
        self.model = model
        self.processor = processor
        self.hidden_size = hidden_size
        self.config = config

        # Set model to eval mode
        self.model.eval()

        # Initialize layer weights if using weighted sum
        self._layer_weights: np.ndarray | None = None
        if config.use_weighted_layer_sum:
            num_layers = len(self.model.encoder.layers) + 1  # +1 for embedding layer
            self._layer_weights = np.ones(num_layers) / num_layers

    @classmethod
    def from_mert(
        cls,
        model_name: str = "m-a-p/MERT-v1-330M",
        config: PretrainedAudioConfig | None = None,
    ) -> "PretrainedAudioEncoderWrapper":
        """Create wrapper from MERT model.

        Args:
            model_name: HuggingFace model identifier
            config: Configuration (uses defaults if None)

        Returns:
            Initialized wrapper
        """
        if config is None:
            config = PretrainedAudioConfig(
                model_name=model_name,
                output_dim=1024,
                target_sample_rate=24000,
            )

        model, processor, hidden_size = load_mert_encoder(model_name)
        return cls(model, processor, hidden_size, config)

    @classmethod
    def from_wavlm(
        cls,
        model_name: str = "microsoft/wavlm-base-plus",
        config: PretrainedAudioConfig | None = None,
    ) -> "PretrainedAudioEncoderWrapper":
        """Create wrapper from WavLM model.

        Args:
            model_name: HuggingFace model identifier
            config: Configuration (uses defaults if None)

        Returns:
            Initialized wrapper
        """
        if config is None:
            config = PretrainedAudioConfig(
                model_name=model_name,
                output_dim=768,  # WavLM base
                target_sample_rate=16000,
            )

        model, processor, hidden_size = load_wavlm_encoder(model_name)
        return cls(model, processor, hidden_size, config)

    def extract_embeddings(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Extract embeddings from audio.

        Args:
            audio: Audio array of shape (samples,) or (batch, samples)
            sample_rate: Sample rate of input audio

        Returns:
            Embeddings of shape (batch, hidden_size) or (hidden_size,)
        """
        import torch

        # Ensure batch dimension
        single_sample = audio.ndim == 1
        if single_sample:
            audio = audio[None, :]

        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        if self.config.use_weighted_layer_sum and hasattr(outputs, "hidden_states"):
            # Weighted sum of all layers
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
            weights = torch.tensor(
                self._layer_weights,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            weights = torch.softmax(weights, dim=0)
            embeddings = (hidden_states * weights[:, None, None, None]).sum(dim=0)
        else:
            # Use last hidden state
            embeddings = outputs.last_hidden_state

        # Mean pool over time: (batch, time, hidden) -> (batch, hidden)
        embeddings = embeddings.mean(dim=1)

        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy()

        if single_sample:
            embeddings_np = embeddings_np[0]

        return embeddings_np

    def update_layer_weights(self, weights: np.ndarray) -> None:
        """Update layer weights for weighted sum.

        Args:
            weights: New layer weights (will be softmaxed)
        """
        self._layer_weights = weights


class PretrainedAudioProjection(nn.Module):
    """Flax module for projecting pretrained embeddings.

    This module takes precomputed embeddings from a pretrained encoder
    and projects them to the target dimension for the classifier.

    Use this when you want to:
    1. Extract embeddings offline using PretrainedAudioEncoderWrapper
    2. Train only the projection + classifier layers in JAX

    Attributes:
        target_dim: Target embedding dimension (e.g., 512)
        dropout_rate: Dropout rate
        dtype: Data type for computation
    """

    target_dim: int = 512
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        embeddings: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        """Project pretrained embeddings.

        Args:
            embeddings: Pretrained embeddings (batch, pretrained_dim)
            train: Training mode flag

        Returns:
            Projected embeddings (batch, target_dim)
        """
        # Project to target dimension
        x = nn.Dense(
            self.target_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(embeddings)

        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # Second projection layer for more capacity
        x = nn.Dense(
            self.target_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)

        return x


class HybridAudioClassifier(nn.Module):
    """Classifier that uses pretrained embeddings as input.

    This model is designed to work with precomputed embeddings from
    pretrained encoders (MERT, WavLM). It includes:
    - Embedding projection layer
    - Harm classification head
    - Speaker embedding head

    Use this when you've precomputed embeddings offline.

    Attributes:
        config: Output classifier configuration
        pretrained_dim: Dimension of pretrained embeddings
        dtype: Data type
    """

    config: OutputClassifierConfig
    pretrained_dim: int = 1024  # MERT default
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        pretrained_embeddings: jnp.ndarray,
        train: bool = True,
    ) -> dict[str, jnp.ndarray]:
        """Forward pass.

        Args:
            pretrained_embeddings: Precomputed embeddings (batch, pretrained_dim)
            train: Training mode

        Returns:
            Dictionary with harm_logits, audio_embeddings, speaker_embeddings
        """
        # Project pretrained embeddings
        projection = PretrainedAudioProjection(
            target_dim=self.config.audio_encoder.embedding_dim,
            dropout_rate=self.config.audio_encoder.dropout_rate,
            dtype=self.dtype,
        )
        audio_embeddings = projection(pretrained_embeddings, train=train)

        # Speaker projection (separate pathway)
        speaker_projection = nn.Dense(
            self.config.speaker.embedding_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(pretrained_embeddings)
        speaker_embeddings = speaker_projection / (
            jnp.linalg.norm(speaker_projection, axis=-1, keepdims=True) + 1e-8
        )

        # Harm classification
        x = nn.Dense(
            self.config.classifier_hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(audio_embeddings)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.config.classifier_dropout)(x, deterministic=not train)

        x = nn.Dense(
            self.config.classifier_hidden_dim // 2,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.config.classifier_dropout)(x, deterministic=not train)

        harm_logits = nn.Dense(
            self.config.num_harm_categories,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)

        return {
            "harm_logits": harm_logits,
            "audio_embeddings": audio_embeddings,
            "speaker_embeddings": speaker_embeddings,
        }


def precompute_embeddings(
    wrapper: PretrainedAudioEncoderWrapper,
    audio_paths: list[str],
    sample_rate: int = 24000,
    batch_size: int = 16,
) -> np.ndarray:
    """Precompute embeddings for a list of audio files.

    This is useful for caching embeddings before training the classifier.

    Args:
        wrapper: Pretrained encoder wrapper
        audio_paths: List of paths to audio files
        sample_rate: Sample rate to load audio at
        batch_size: Batch size for processing

    Returns:
        Array of embeddings (num_files, hidden_size)
    """
    import librosa

    all_embeddings = []

    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]
        batch_audio = []

        for path in batch_paths:
            audio, _ = librosa.load(path, sr=sample_rate)
            batch_audio.append(audio)

        # Pad to same length
        max_len = max(len(a) for a in batch_audio)
        batch_audio_padded = np.zeros((len(batch_audio), max_len))
        for j, audio in enumerate(batch_audio):
            batch_audio_padded[j, : len(audio)] = audio

        embeddings = wrapper.extract_embeddings(batch_audio_padded, sample_rate)
        all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)


def initialize_hybrid_classifier(
    config: OutputClassifierConfig,
    pretrained_dim: int,
    rng: jax.Array,
) -> tuple[HybridAudioClassifier, dict[str, Any]]:
    """Initialize HybridAudioClassifier with random weights.

    Args:
        config: Output classifier configuration
        pretrained_dim: Dimension of pretrained embeddings
        rng: JAX random key

    Returns:
        Tuple of (model, variables)
    """
    model = HybridAudioClassifier(
        config=config,
        pretrained_dim=pretrained_dim,
    )

    # Initialize with dummy input
    dummy_embeddings = jnp.zeros((1, pretrained_dim))
    variables = model.init(rng, dummy_embeddings, train=False)

    return model, variables


def create_pretrained_training_setup(
    pretrained_config: PretrainedAudioConfig,
    classifier_config: OutputClassifierConfig,
    rng: jax.Array,
) -> tuple[PretrainedAudioEncoderWrapper, HybridAudioClassifier, dict[str, Any]]:
    """Create complete training setup with pretrained encoder.

    This creates:
    1. Pretrained encoder wrapper (for extracting embeddings)
    2. HybridAudioClassifier (trainable, uses precomputed embeddings)
    3. Initialized model variables

    Args:
        pretrained_config: Configuration for pretrained encoder
        classifier_config: Configuration for classifier
        rng: JAX random key

    Returns:
        Tuple of (encoder_wrapper, classifier_model, classifier_variables)
    """
    # Create pretrained encoder wrapper
    model_name_lower = pretrained_config.model_name.lower()
    if "mert" in model_name_lower:
        wrapper = PretrainedAudioEncoderWrapper.from_mert(
            pretrained_config.model_name,
            pretrained_config,
        )
    elif "wavlm" in model_name_lower:
        wrapper = PretrainedAudioEncoderWrapper.from_wavlm(
            pretrained_config.model_name,
            pretrained_config,
        )
    else:
        raise PretrainedAudioLoadError(
            f"Unknown pretrained model type: {pretrained_config.model_name}. Supported: MERT, WavLM"
        )

    # Create classifier
    model, variables = initialize_hybrid_classifier(
        classifier_config,
        wrapper.hidden_size,
        rng,
    )

    return wrapper, model, variables
