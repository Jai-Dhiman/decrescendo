"""Dataset classes for Output Classifier (audio) training."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import jax.numpy as jnp
import librosa
import numpy as np

from ..output_classifier.audio_preprocessing import AudioLoadError, AudioPreprocessor
from ..output_classifier.config import PreprocessingConfig


@dataclass
class AudioClassificationSample:
    """Single training sample for the Output Classifier.

    Attributes:
        audio_path: Path to the audio file
        harm_labels: Multi-hot vector for 7 harm categories
        speaker_id: Optional speaker identifier for verification training
        metadata: Optional additional metadata
    """

    audio_path: str
    harm_labels: list[int]
    speaker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate labels after initialization."""
        if len(self.harm_labels) != 7:
            raise ValueError(
                f"harm_labels must have 7 elements, got {len(self.harm_labels)}"
            )
        if not all(l in (0, 1) for l in self.harm_labels):
            raise ValueError("harm_labels must contain only 0 or 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "audio_path": self.audio_path,
            "harm_labels": self.harm_labels,
        }
        if self.speaker_id is not None:
            result["speaker_id"] = self.speaker_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for audio augmentations.

    All augmentations are applied probabilistically during training.
    """

    # Noise injection
    enable_noise: bool = True
    noise_prob: float = 0.5  # Probability of applying noise
    noise_snr_db_min: float = 10.0  # Minimum SNR in dB
    noise_snr_db_max: float = 30.0  # Maximum SNR in dB

    # Time masking (SpecAugment-style)
    enable_time_mask: bool = True
    time_mask_prob: float = 0.5
    time_mask_max_ratio: float = 0.1  # Max fraction of audio to mask

    # Pitch shifting
    enable_pitch_shift: bool = True
    pitch_shift_prob: float = 0.3
    pitch_shift_semitones_min: float = -2.0
    pitch_shift_semitones_max: float = 2.0

    # Speed perturbation
    enable_speed: bool = True
    speed_prob: float = 0.3
    speed_min: float = 0.9
    speed_max: float = 1.1

    # Room impulse response (reverb)
    enable_rir: bool = False  # Disabled by default (requires RIR files)
    rir_prob: float = 0.3
    rir_directory: str | None = None  # Path to directory with RIR files


class AudioDatasetError(Exception):
    """Raised when audio dataset operations fail."""

    pass


class AudioAugmenter:
    """Applies audio augmentations for training.

    Augmentations are applied probabilistically based on configuration.

    Example:
        >>> augmenter = AudioAugmenter(AugmentationConfig())
        >>> augmented = augmenter.apply(audio, sample_rate=24000)
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize augmenter.

        Args:
            config: Augmentation configuration
            seed: Random seed for reproducibility
        """
        self.config = config or AugmentationConfig()
        self.rng = np.random.default_rng(seed)
        self._rir_files: list[Path] | None = None

    def apply(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply configured augmentations to audio.

        Args:
            audio: Audio array (samples,)
            sample_rate: Sample rate

        Returns:
            Augmented audio array
        """
        # Apply augmentations in order
        if self.config.enable_speed and self.rng.random() < self.config.speed_prob:
            audio = self._apply_speed(audio, sample_rate)

        if self.config.enable_pitch_shift and self.rng.random() < self.config.pitch_shift_prob:
            audio = self._apply_pitch_shift(audio, sample_rate)

        if self.config.enable_noise and self.rng.random() < self.config.noise_prob:
            audio = self._apply_noise(audio)

        if self.config.enable_time_mask and self.rng.random() < self.config.time_mask_prob:
            audio = self._apply_time_mask(audio)

        if self.config.enable_rir and self.rng.random() < self.config.rir_prob:
            audio = self._apply_rir(audio, sample_rate)

        return audio

    def _apply_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add Gaussian noise at random SNR."""
        snr_db = self.rng.uniform(
            self.config.noise_snr_db_min,
            self.config.noise_snr_db_max,
        )

        # Compute signal power
        signal_power = np.mean(audio**2)
        if signal_power < 1e-10:
            return audio

        # Compute noise power for target SNR
        # SNR = 10 * log10(signal_power / noise_power)
        noise_power = signal_power / (10 ** (snr_db / 10))

        # Generate noise
        noise = self.rng.normal(0, np.sqrt(noise_power), size=audio.shape)

        return (audio + noise).astype(audio.dtype)

    def _apply_time_mask(self, audio: np.ndarray) -> np.ndarray:
        """Apply time masking (set random segment to zero)."""
        max_mask_len = int(len(audio) * self.config.time_mask_max_ratio)
        if max_mask_len < 1:
            return audio

        mask_len = self.rng.integers(1, max_mask_len + 1)
        mask_start = self.rng.integers(0, len(audio) - mask_len + 1)

        audio_masked = audio.copy()
        audio_masked[mask_start : mask_start + mask_len] = 0.0

        return audio_masked

    def _apply_pitch_shift(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply pitch shifting."""
        n_steps = self.rng.uniform(
            self.config.pitch_shift_semitones_min,
            self.config.pitch_shift_semitones_max,
        )

        return librosa.effects.pitch_shift(
            audio,
            sr=sample_rate,
            n_steps=n_steps,
        )

    def _apply_speed(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply speed perturbation (time stretch without pitch change)."""
        rate = self.rng.uniform(self.config.speed_min, self.config.speed_max)

        # Time stretch changes duration; we need to resample to maintain length
        stretched = librosa.effects.time_stretch(audio, rate=rate)

        # Adjust length to match original
        if len(stretched) > len(audio):
            stretched = stretched[: len(audio)]
        elif len(stretched) < len(audio):
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))

        return stretched

    def _apply_rir(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply room impulse response convolution."""
        if self.config.rir_directory is None:
            return audio

        # Lazy load RIR files
        if self._rir_files is None:
            rir_dir = Path(self.config.rir_directory)
            if not rir_dir.exists():
                return audio
            self._rir_files = list(rir_dir.glob("*.wav")) + list(rir_dir.glob("*.flac"))

        if not self._rir_files:
            return audio

        # Select random RIR
        rir_path = self.rng.choice(self._rir_files)

        try:
            rir, rir_sr = librosa.load(rir_path, sr=sample_rate)
            # Normalize RIR
            rir = rir / (np.max(np.abs(rir)) + 1e-8)

            # Convolve
            convolved = np.convolve(audio, rir, mode="full")[: len(audio)]

            # Normalize to match original level
            orig_rms = np.sqrt(np.mean(audio**2))
            conv_rms = np.sqrt(np.mean(convolved**2))
            if conv_rms > 1e-8:
                convolved = convolved * (orig_rms / conv_rms)

            return convolved.astype(audio.dtype)
        except Exception:
            # If RIR loading fails, return original audio
            return audio


class AudioDataset:
    """Dataset for audio classification.

    Supports loading from:
    - JSONL manifest files
    - Directory with labels file
    - In-memory lists of samples
    """

    def __init__(self, samples: list[AudioClassificationSample]) -> None:
        """Initialize dataset with samples.

        Args:
            samples: List of AudioClassificationSample objects
        """
        self.samples = samples
        self._speaker_to_id: dict[str, int] | None = None

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> AudioClassificationSample:
        """Get sample by index."""
        return self.samples[idx]

    @property
    def speaker_to_id(self) -> dict[str, int]:
        """Mapping from speaker_id string to integer index."""
        if self._speaker_to_id is None:
            unique_speakers = set()
            for sample in self.samples:
                if sample.speaker_id is not None:
                    unique_speakers.add(sample.speaker_id)
            self._speaker_to_id = {
                spk: idx for idx, spk in enumerate(sorted(unique_speakers))
            }
        return self._speaker_to_id

    @classmethod
    def from_manifest(cls, path: Path | str) -> "AudioDataset":
        """Load dataset from JSONL manifest file.

        Expected format (one JSON object per line):
        {
            "audio_path": "/path/to/audio.wav",
            "harm_labels": [0, 1, 0, 0, 0, 0, 0],
            "speaker_id": "speaker_001",  // optional
            "metadata": {}  // optional
        }

        Args:
            path: Path to JSONL manifest file

        Returns:
            AudioDataset instance

        Raises:
            AudioDatasetError: If file doesn't exist or contains invalid data
        """
        path = Path(path)

        if not path.exists():
            raise AudioDatasetError(f"Manifest file not found: {path}")

        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise AudioDatasetError(
                        f"Invalid JSON at line {line_num} in {path}: {e}"
                    ) from e

                required_keys = ["audio_path", "harm_labels"]
                missing_keys = [k for k in required_keys if k not in data]
                if missing_keys:
                    raise AudioDatasetError(
                        f"Missing keys {missing_keys} at line {line_num} in {path}"
                    )

                try:
                    sample = AudioClassificationSample(
                        audio_path=data["audio_path"],
                        harm_labels=data["harm_labels"],
                        speaker_id=data.get("speaker_id"),
                        metadata=data.get("metadata", {}),
                    )
                    samples.append(sample)
                except (ValueError, TypeError) as e:
                    raise AudioDatasetError(
                        f"Invalid data at line {line_num} in {path}: {e}"
                    ) from e

        return cls(samples)

    @classmethod
    def from_directory(
        cls,
        audio_dir: Path | str,
        labels_file: Path | str,
    ) -> "AudioDataset":
        """Load dataset from directory with separate labels file.

        The labels file should be a JSON file mapping audio filenames to labels:
        {
            "audio1.wav": {
                "harm_labels": [0, 1, 0, 0, 0, 0, 0],
                "speaker_id": "speaker_001"  // optional
            },
            ...
        }

        Args:
            audio_dir: Directory containing audio files
            labels_file: Path to JSON labels file

        Returns:
            AudioDataset instance

        Raises:
            AudioDatasetError: If directory or labels file issues
        """
        audio_dir = Path(audio_dir)
        labels_file = Path(labels_file)

        if not audio_dir.exists():
            raise AudioDatasetError(f"Audio directory not found: {audio_dir}")

        if not labels_file.exists():
            raise AudioDatasetError(f"Labels file not found: {labels_file}")

        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                labels_data = json.load(f)
        except json.JSONDecodeError as e:
            raise AudioDatasetError(f"Invalid JSON in labels file: {e}") from e

        samples = []
        for filename, label_info in labels_data.items():
            audio_path = audio_dir / filename

            if not audio_path.exists():
                raise AudioDatasetError(
                    f"Audio file not found: {audio_path} (referenced in labels)"
                )

            if "harm_labels" not in label_info:
                raise AudioDatasetError(
                    f"Missing 'harm_labels' for {filename} in labels file"
                )

            try:
                sample = AudioClassificationSample(
                    audio_path=str(audio_path),
                    harm_labels=label_info["harm_labels"],
                    speaker_id=label_info.get("speaker_id"),
                    metadata=label_info.get("metadata", {}),
                )
                samples.append(sample)
            except (ValueError, TypeError) as e:
                raise AudioDatasetError(
                    f"Invalid data for {filename}: {e}"
                ) from e

        return cls(samples)

    def save_manifest(self, path: Path | str) -> None:
        """Save dataset to JSONL manifest file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict()) + "\n")


@dataclass
class AudioChunk:
    """Internal representation of a single audio chunk with labels."""

    audio: np.ndarray  # (chunk_samples,) audio data
    harm_labels: np.ndarray  # (7,) float32
    speaker_id: int | None  # Integer speaker ID (or None)
    sample_idx: int  # Index of original sample
    chunk_idx: int  # Index of chunk within sample


class AudioDataLoader:
    """JAX-compatible audio data loader with chunking and augmentation.

    Handles on-the-fly audio loading, preprocessing, and batching.
    Each batch contains individual audio chunks (not full files).

    Example:
        >>> dataset = AudioDataset.from_manifest("train.jsonl")
        >>> loader = AudioDataLoader(dataset, batch_size=32)
        >>> for batch in loader:
        ...     print(batch["audio"].shape)  # (32, 24000)
    """

    def __init__(
        self,
        dataset: AudioDataset,
        batch_size: int,
        preprocessing_config: PreprocessingConfig | None = None,
        augmentation_config: AugmentationConfig | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        speaker_aware_sampling: bool = False,
        min_samples_per_speaker: int = 2,
        seed: int | None = None,
    ) -> None:
        """Initialize data loader.

        Args:
            dataset: AudioDataset to load from
            batch_size: Number of chunks per batch
            preprocessing_config: Audio preprocessing configuration
            augmentation_config: Data augmentation configuration (None to disable)
            shuffle: Whether to shuffle data each epoch
            drop_last: Whether to drop incomplete final batch
            speaker_aware_sampling: If True, ensure each batch has multiple
                samples per speaker for contrastive learning
            min_samples_per_speaker: Minimum samples per speaker when
                speaker_aware_sampling is enabled
            seed: Random seed for shuffling and augmentation
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.speaker_aware_sampling = speaker_aware_sampling
        self.min_samples_per_speaker = min_samples_per_speaker

        self.preprocessor = AudioPreprocessor(preprocessing_config)
        self.augmenter = (
            AudioAugmenter(augmentation_config, seed=seed)
            if augmentation_config is not None
            else None
        )
        self.rng = np.random.default_rng(seed)

        # Pre-compute chunk indices for all samples
        self._chunk_index: list[tuple[int, int]] | None = None

    def _build_chunk_index(self) -> list[tuple[int, int]]:
        """Build index of all chunks across all samples.

        Returns:
            List of (sample_idx, chunk_idx) tuples
        """
        if self._chunk_index is not None:
            return self._chunk_index

        chunk_index = []
        config = self.preprocessor.config

        for sample_idx, sample in enumerate(self.dataset.samples):
            try:
                # Load audio to determine number of chunks
                audio, sr = self.preprocessor.load_audio(sample.audio_path)
                audio = self.preprocessor.preprocess(audio, sr)

                num_samples = len(audio)
                chunk_size = config.chunk_samples
                hop_size = config.hop_samples

                # Calculate number of chunks
                num_chunks = 0
                for start in range(0, num_samples, hop_size):
                    if start + chunk_size <= num_samples or start < num_samples:
                        num_chunks += 1

                for chunk_idx in range(num_chunks):
                    chunk_index.append((sample_idx, chunk_idx))

            except AudioLoadError:
                # Skip samples that can't be loaded
                continue

        self._chunk_index = chunk_index
        return chunk_index

    def __len__(self) -> int:
        """Return number of batches."""
        chunk_index = self._build_chunk_index()
        n = len(chunk_index)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[dict[str, jnp.ndarray]]:
        """Iterate over batches."""
        chunk_index = self._build_chunk_index()

        if self.speaker_aware_sampling:
            yield from self._iter_speaker_aware(chunk_index)
        else:
            yield from self._iter_standard(chunk_index)

    def _iter_standard(
        self,
        chunk_index: list[tuple[int, int]],
    ) -> Iterator[dict[str, jnp.ndarray]]:
        """Standard iteration with optional shuffling."""
        indices = np.arange(len(chunk_index))

        if self.shuffle:
            self.rng.shuffle(indices)

        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]

            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            chunks = self._load_chunks([chunk_index[i] for i in batch_indices])
            yield self._collate(chunks)

    def _iter_speaker_aware(
        self,
        chunk_index: list[tuple[int, int]],
    ) -> Iterator[dict[str, jnp.ndarray]]:
        """Speaker-aware iteration ensuring multiple samples per speaker."""
        # Group chunks by speaker
        speaker_to_chunks: dict[int, list[int]] = {}
        no_speaker_chunks: list[int] = []

        for idx, (sample_idx, _) in enumerate(chunk_index):
            sample = self.dataset.samples[sample_idx]
            if sample.speaker_id is not None:
                speaker_int = self.dataset.speaker_to_id.get(sample.speaker_id)
                if speaker_int is not None:
                    if speaker_int not in speaker_to_chunks:
                        speaker_to_chunks[speaker_int] = []
                    speaker_to_chunks[speaker_int].append(idx)
            else:
                no_speaker_chunks.append(idx)

        # Filter speakers with enough samples
        valid_speakers = [
            spk for spk, chunks in speaker_to_chunks.items()
            if len(chunks) >= self.min_samples_per_speaker
        ]

        if not valid_speakers:
            # Fall back to standard iteration if not enough speakers
            yield from self._iter_standard(chunk_index)
            return

        # Build batches with speaker awareness
        all_batches: list[list[int]] = []
        remaining_chunks = {spk: list(chunks) for spk, chunks in speaker_to_chunks.items()}

        while True:
            # Shuffle speakers for this round
            available_speakers = [
                spk for spk in valid_speakers
                if len(remaining_chunks.get(spk, [])) >= self.min_samples_per_speaker
            ]

            if not available_speakers:
                break

            self.rng.shuffle(available_speakers)

            batch: list[int] = []
            for spk in available_speakers:
                if len(batch) >= self.batch_size:
                    break

                # Add min_samples_per_speaker chunks from this speaker
                spk_chunks = remaining_chunks[spk]
                self.rng.shuffle(spk_chunks)

                n_to_add = min(
                    self.min_samples_per_speaker,
                    len(spk_chunks),
                    self.batch_size - len(batch),
                )
                batch.extend(spk_chunks[:n_to_add])
                remaining_chunks[spk] = spk_chunks[n_to_add:]

            if len(batch) >= self.min_samples_per_speaker * 2:
                all_batches.append(batch)

        # Yield batches
        if self.shuffle:
            self.rng.shuffle(all_batches)

        for batch_indices in all_batches:
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            chunks = self._load_chunks([chunk_index[i] for i in batch_indices])
            yield self._collate(chunks)

    def _load_chunks(
        self,
        chunk_refs: list[tuple[int, int]],
    ) -> list[AudioChunk]:
        """Load audio chunks for given references."""
        chunks = []
        config = self.preprocessor.config

        # Cache loaded audio to avoid reloading same file
        audio_cache: dict[int, np.ndarray] = {}

        for sample_idx, chunk_idx in chunk_refs:
            sample = self.dataset.samples[sample_idx]

            # Load and preprocess audio (with caching)
            if sample_idx not in audio_cache:
                try:
                    audio, sr = self.preprocessor.load_audio(sample.audio_path)
                    audio = self.preprocessor.preprocess(audio, sr)
                    audio_cache[sample_idx] = audio
                except AudioLoadError:
                    continue

            audio = audio_cache[sample_idx]

            # Extract chunk
            start = chunk_idx * config.hop_samples
            end = start + config.chunk_samples

            if end <= len(audio):
                chunk_audio = audio[start:end]
            else:
                # Zero-pad final chunk
                chunk_audio = np.zeros(config.chunk_samples, dtype=np.float32)
                chunk_audio[: len(audio) - start] = audio[start:]

            # Apply augmentation if configured
            if self.augmenter is not None:
                chunk_audio = self.augmenter.apply(
                    chunk_audio,
                    sample_rate=config.sample_rate,
                )

            # Get speaker ID as integer
            speaker_int = None
            if sample.speaker_id is not None:
                speaker_int = self.dataset.speaker_to_id.get(sample.speaker_id)

            chunks.append(
                AudioChunk(
                    audio=chunk_audio.astype(np.float32),
                    harm_labels=np.array(sample.harm_labels, dtype=np.float32),
                    speaker_id=speaker_int,
                    sample_idx=sample_idx,
                    chunk_idx=chunk_idx,
                )
            )

        return chunks

    def _collate(self, chunks: list[AudioChunk]) -> dict[str, jnp.ndarray]:
        """Collate chunks into a batch.

        Args:
            chunks: List of AudioChunk objects

        Returns:
            Dictionary with batched tensors:
                - audio: (batch, samples) audio data
                - harm_labels: (batch, 7) harm labels
                - speaker_ids: (batch,) speaker IDs (-1 for unknown)
        """
        audio = np.stack([c.audio for c in chunks])
        harm_labels = np.stack([c.harm_labels for c in chunks])

        # Use -1 for samples without speaker_id
        speaker_ids = np.array(
            [c.speaker_id if c.speaker_id is not None else -1 for c in chunks],
            dtype=np.int32,
        )

        return {
            "audio": jnp.array(audio),
            "harm_labels": jnp.array(harm_labels),
            "speaker_ids": jnp.array(speaker_ids),
        }


def create_dummy_audio_dataset(
    num_samples: int = 50,
    num_speakers: int = 5,
    seed: int = 42,
) -> AudioDataset:
    """Create a dummy audio dataset for testing.

    Note: This creates samples with non-existent audio paths.
    For actual testing, use real audio files.

    Args:
        num_samples: Number of samples to generate
        num_speakers: Number of unique speakers
        seed: Random seed

    Returns:
        AudioDataset with random samples
    """
    rng = np.random.default_rng(seed)

    samples = []
    for i in range(num_samples):
        sample = AudioClassificationSample(
            audio_path=f"/dummy/audio_{i:04d}.wav",
            harm_labels=[int(rng.integers(0, 2)) for _ in range(7)],
            speaker_id=f"speaker_{i % num_speakers:03d}",
        )
        samples.append(sample)

    return AudioDataset(samples)
