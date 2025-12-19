"""Audio preprocessing utilities for the Output Classifier."""

from pathlib import Path
from typing import Iterator

import jax.numpy as jnp
import librosa
import numpy as np
import soundfile as sf

from .config import PreprocessingConfig


class AudioLoadError(Exception):
    """Raised when audio file cannot be loaded."""

    pass


class AudioPreprocessor:
    """Preprocessor for audio files.

    Handles loading, resampling, normalization, and chunking of audio
    for the output classifier pipeline.

    Example:
        >>> preprocessor = AudioPreprocessor()
        >>> chunks = preprocessor.process_file("audio.wav")
        >>> for chunk in chunks:
        ...     print(chunk.shape)  # (24000,) for 1-second chunks
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        """Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()

    def load_audio(self, path: Path | str) -> tuple[np.ndarray, int]:
        """Load audio file.

        Args:
            path: Path to audio file

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            AudioLoadError: If file cannot be loaded
        """
        path = Path(path)

        if not path.exists():
            raise AudioLoadError(f"Audio file not found: {path}")

        try:
            audio, sr = sf.read(path)
            return audio, sr
        except Exception as e:
            raise AudioLoadError(f"Failed to load audio file {path}: {e}") from e

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to target sample rate.

        Args:
            audio: Audio array
            orig_sr: Original sample rate

        Returns:
            Resampled audio array
        """
        if orig_sr == self.config.sample_rate:
            return audio

        return librosa.resample(
            audio,
            orig_sr=orig_sr,
            target_sr=self.config.sample_rate,
        )

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo audio to mono.

        Args:
            audio: Audio array (samples,) or (samples, channels)

        Returns:
            Mono audio array (samples,)
        """
        if audio.ndim == 1:
            return audio

        # Average channels
        return np.mean(audio, axis=1)

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target level.

        Uses RMS normalization to target dB level.

        Args:
            audio: Audio array

        Returns:
            Normalized audio array
        """
        if not self.config.normalize_audio:
            return audio

        # Compute current RMS
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-8:
            return audio

        # Target RMS from dB
        target_rms = 10 ** (self.config.target_db / 20)

        # Scale audio
        return audio * (target_rms / rms)

    def chunk_audio(self, audio: np.ndarray) -> Iterator[np.ndarray]:
        """Split audio into overlapping chunks.

        Args:
            audio: Audio array

        Yields:
            Audio chunks of size chunk_samples
        """
        chunk_size = self.config.chunk_samples
        hop_size = self.config.hop_samples

        num_samples = len(audio)

        for start in range(0, num_samples, hop_size):
            end = start + chunk_size

            if end <= num_samples:
                yield audio[start:end]
            else:
                # Zero-pad final chunk
                chunk = np.zeros(chunk_size, dtype=audio.dtype)
                chunk[: num_samples - start] = audio[start:]
                yield chunk

    def preprocess(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply full preprocessing pipeline.

        Args:
            audio: Raw audio array
            sample_rate: Original sample rate

        Returns:
            Preprocessed audio array
        """
        # Convert to mono if needed
        if self.config.mono:
            audio = self.to_mono(audio)

        # Resample
        audio = self.resample(audio, sample_rate)

        # Normalize
        audio = self.normalize(audio)

        return audio

    def process_file(self, path: Path | str) -> Iterator[jnp.ndarray]:
        """Load and process audio file into chunks.

        Args:
            path: Path to audio file

        Yields:
            JAX arrays of audio chunks

        Raises:
            AudioLoadError: If file cannot be loaded
        """
        # Load audio
        audio, sr = self.load_audio(path)

        # Preprocess
        audio = self.preprocess(audio, sr)

        # Chunk and convert to JAX arrays
        for chunk in self.chunk_audio(audio):
            yield jnp.array(chunk, dtype=jnp.float32)

    def process_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Iterator[jnp.ndarray]:
        """Process audio array into chunks.

        Args:
            audio: Audio array
            sample_rate: Sample rate of audio

        Yields:
            JAX arrays of audio chunks
        """
        # Preprocess
        audio = self.preprocess(audio, sample_rate)

        # Chunk and convert to JAX arrays
        for chunk in self.chunk_audio(audio):
            yield jnp.array(chunk, dtype=jnp.float32)

    def process_stream(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int,
    ) -> Iterator[jnp.ndarray]:
        """Process streaming audio into chunks.

        Buffers incoming audio and yields complete chunks.

        Args:
            audio_stream: Iterator yielding audio arrays
            sample_rate: Sample rate of audio

        Yields:
            JAX arrays of audio chunks
        """
        buffer = np.array([], dtype=np.float32)
        chunk_size = self.config.chunk_samples
        hop_size = self.config.hop_samples

        for incoming in audio_stream:
            # Preprocess incoming audio
            processed = self.preprocess(incoming, sample_rate)
            buffer = np.concatenate([buffer, processed])

            # Yield complete chunks
            while len(buffer) >= chunk_size:
                yield jnp.array(buffer[:chunk_size], dtype=jnp.float32)
                buffer = buffer[hop_size:]

        # Yield final chunk if any audio remains
        if len(buffer) > 0:
            final_chunk = np.zeros(chunk_size, dtype=np.float32)
            final_chunk[: len(buffer)] = buffer
            yield jnp.array(final_chunk, dtype=jnp.float32)


def compute_mel_spectrogram(
    audio: jnp.ndarray,
    sample_rate: int = 24000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> jnp.ndarray:
    """Compute mel spectrogram from audio.

    Args:
        audio: Audio array (samples,)
        sample_rate: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        Mel spectrogram (n_mels, time)
    """
    # Use librosa for mel spectrogram (numpy-based, then convert)
    audio_np = np.array(audio)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_np,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return jnp.array(mel_spec_db, dtype=jnp.float32)
