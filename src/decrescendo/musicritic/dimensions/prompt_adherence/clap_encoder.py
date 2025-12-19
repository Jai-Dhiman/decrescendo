"""CLAP encoder wrapper for text-audio embedding extraction.

Uses HuggingFace transformers CLAP implementation for stability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .config import CLAPEncoderConfig
from .exceptions import CLAPDependencyError, CLAPLoadError

if TYPE_CHECKING:
    from transformers import ClapModel, ClapProcessor


class CLAPEncoder:
    """Wrapper for HuggingFace CLAP model.

    Provides methods for extracting text and audio embeddings using the
    CLAP (Contrastive Language-Audio Pretraining) model from HuggingFace.

    The encoder supports lazy loading - the model is only loaded when
    first needed, not at instantiation time.

    Example:
        >>> encoder = CLAPEncoder.from_music_checkpoint()
        >>> text_emb = encoder.encode_text("upbeat electronic music")
        >>> audio_emb = encoder.encode_audio(audio_array, sample_rate=48000)
        >>> similarity = encoder.compute_similarity(text_emb, audio_emb)

    Attributes:
        config: Encoder configuration.
    """

    def __init__(self, config: CLAPEncoderConfig | None = None) -> None:
        """Initialize encoder.

        Args:
            config: Encoder configuration. Uses defaults if None.
        """
        self.config = config or CLAPEncoderConfig()
        self._model: ClapModel | None = None
        self._processor: ClapProcessor | None = None
        self._text_cache: dict[str, np.ndarray] = {}

    @classmethod
    def from_music_checkpoint(
        cls,
        model_name: str | None = None,
        config: CLAPEncoderConfig | None = None,
    ) -> CLAPEncoder:
        """Create encoder from music-optimized checkpoint.

        Args:
            model_name: HuggingFace model name. If None, uses default.
            config: Encoder configuration.

        Returns:
            Initialized CLAPEncoder.
        """
        if config is None:
            config = CLAPEncoderConfig(
                model_name=model_name or "laion/larger_clap_music",
            )
        return cls(config)

    @property
    def model(self) -> ClapModel:
        """Lazily load and return the CLAP model."""
        if self._model is None:
            self._load_model()
        return self._model  # type: ignore

    @property
    def processor(self) -> ClapProcessor:
        """Lazily load and return the CLAP processor."""
        if self._processor is None:
            self._load_model()
        return self._processor  # type: ignore

    def _load_model(self) -> None:
        """Load CLAP model and processor from HuggingFace.

        Raises:
            CLAPDependencyError: If transformers is not properly configured.
            CLAPLoadError: If model fails to load.
        """
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as e:
            raise CLAPDependencyError(
                "transformers package with CLAP support required. "
                "Install with: uv sync --extra clap"
            ) from e

        try:
            self._processor = ClapProcessor.from_pretrained(self.config.model_name)
            self._model = ClapModel.from_pretrained(self.config.model_name)
            self._model.eval()  # Set to evaluation mode
        except Exception as e:
            raise CLAPLoadError(
                f"Failed to load CLAP model '{self.config.model_name}': {e}"
            ) from e

    def encode_text(
        self,
        text: str | list[str],
        use_cache: bool = True,
    ) -> np.ndarray:
        """Encode text prompt(s) to embeddings.

        Args:
            text: Single text prompt or list of prompts.
            use_cache: Whether to use cached embeddings.

        Returns:
            Text embeddings of shape (embedding_dim,) for single text
            or (batch_size, embedding_dim) for list.
        """
        import torch

        single_input = isinstance(text, str)
        texts = [text] if single_input else text

        embeddings: list[np.ndarray | None] = []
        texts_to_encode: list[str] = []
        indices_to_encode: list[int] = []

        for i, t in enumerate(texts):
            if use_cache and t in self._text_cache:
                embeddings.append(self._text_cache[t])
            else:
                texts_to_encode.append(t)
                indices_to_encode.append(i)
                embeddings.append(None)  # Placeholder

        if texts_to_encode:
            inputs = self.processor(
                text=texts_to_encode,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

            new_embeddings = text_features.cpu().numpy()

            for i, idx in enumerate(indices_to_encode):
                emb = new_embeddings[i]
                embeddings[idx] = emb
                if use_cache:
                    self._text_cache[texts[idx]] = emb

        # Stack embeddings, filtering out any remaining None values
        result = np.stack([e for e in embeddings if e is not None])
        return result[0] if single_input else result

    def encode_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Encode audio to embedding.

        Args:
            audio: Audio waveform of shape (samples,) or (batch, samples).
            sample_rate: Sample rate of the audio.

        Returns:
            Audio embedding of shape (embedding_dim,) for single audio
            or (batch_size, embedding_dim) for batch.
        """
        import librosa
        import torch

        single_input = audio.ndim == 1

        # Resample to 48kHz if needed (CLAP requires 48kHz)
        target_sr = self.config.sample_rate
        if sample_rate != target_sr:
            if single_input:
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=target_sr
                )
            else:
                audio = np.array(
                    [
                        librosa.resample(a, orig_sr=sample_rate, target_sr=target_sr)
                        for a in audio
                    ]
                )

        # Ensure correct shape for processor: list of arrays
        if single_input:
            audio_list = [audio]
        else:
            audio_list = list(audio)

        inputs = self.processor(
            audio=audio_list,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            audio_features = self.model.get_audio_features(**inputs)

        embedding = audio_features.cpu().numpy()

        return embedding[0] if single_input else embedding

    def compute_similarity(
        self,
        text_embedding: np.ndarray,
        audio_embedding: np.ndarray,
    ) -> float:
        """Compute cosine similarity between text and audio embeddings.

        Args:
            text_embedding: Text embedding of shape (embedding_dim,).
            audio_embedding: Audio embedding of shape (embedding_dim,).

        Returns:
            Cosine similarity score in range [-1, 1].
        """
        # Normalize embeddings
        text_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)
        audio_norm = audio_embedding / (np.linalg.norm(audio_embedding) + 1e-8)

        # Compute cosine similarity
        similarity = float(np.dot(text_norm, audio_norm))

        return similarity

    def clear_cache(self) -> None:
        """Clear the text embedding cache."""
        self._text_cache.clear()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
