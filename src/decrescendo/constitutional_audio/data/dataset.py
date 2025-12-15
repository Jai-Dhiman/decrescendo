"""Dataset classes for Input Classifier training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import jax.numpy as jnp
import numpy as np


@dataclass
class InputClassifierSample:
    """Single training sample for the Input Classifier.

    Attributes:
        text: The text prompt to classify
        intent_label: 0=benign, 1=suspicious, 2=malicious
        artist_label: 0=none, 1=named_artist, 2=style_reference
        voice_label: 0=none, 1=celebrity, 2=politician
        policy_labels: Multi-hot vector for 7 harm categories
    """

    text: str
    intent_label: int
    artist_label: int
    voice_label: int
    policy_labels: list[int]

    def __post_init__(self) -> None:
        """Validate labels after initialization."""
        if not 0 <= self.intent_label <= 2:
            raise ValueError(f"intent_label must be 0-2, got {self.intent_label}")
        if not 0 <= self.artist_label <= 2:
            raise ValueError(f"artist_label must be 0-2, got {self.artist_label}")
        if not 0 <= self.voice_label <= 2:
            raise ValueError(f"voice_label must be 0-2, got {self.voice_label}")
        if len(self.policy_labels) != 7:
            raise ValueError(f"policy_labels must have 7 elements, got {len(self.policy_labels)}")
        if not all(l in (0, 1) for l in self.policy_labels):
            raise ValueError("policy_labels must contain only 0 or 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "intent_label": self.intent_label,
            "artist_label": self.artist_label,
            "voice_label": self.voice_label,
            "policy_labels": self.policy_labels,
        }


class DatasetLoadError(Exception):
    """Raised when dataset loading fails."""

    pass


class InputClassifierDataset:
    """Dataset for Constitutional Audio Input Classifier.

    Supports loading from:
    - JSONL files (one JSON object per line)
    - In-memory lists of samples

    Each JSON object should have keys:
    - text: str
    - intent_label: int (0-2)
    - artist_label: int (0-2)
    - voice_label: int (0-2)
    - policy_labels: list[int] (7 binary values)
    """

    def __init__(self, samples: list[InputClassifierSample]) -> None:
        """Initialize dataset with samples.

        Args:
            samples: List of InputClassifierSample objects
        """
        self.samples = samples

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> InputClassifierSample:
        """Get sample by index."""
        return self.samples[idx]

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "InputClassifierDataset":
        """Load dataset from JSONL file.

        Args:
            path: Path to JSONL file

        Returns:
            InputClassifierDataset instance

        Raises:
            DatasetLoadError: If file doesn't exist or contains invalid data
        """
        path = Path(path)

        if not path.exists():
            raise DatasetLoadError(f"Dataset file not found: {path}")

        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise DatasetLoadError(
                        f"Invalid JSON at line {line_num} in {path}: {e}"
                    ) from e

                required_keys = ["text", "intent_label", "artist_label", "voice_label", "policy_labels"]
                missing_keys = [k for k in required_keys if k not in data]
                if missing_keys:
                    raise DatasetLoadError(
                        f"Missing keys {missing_keys} at line {line_num} in {path}"
                    )

                try:
                    sample = InputClassifierSample(
                        text=data["text"],
                        intent_label=data["intent_label"],
                        artist_label=data["artist_label"],
                        voice_label=data["voice_label"],
                        policy_labels=data["policy_labels"],
                    )
                    samples.append(sample)
                except (ValueError, TypeError) as e:
                    raise DatasetLoadError(
                        f"Invalid data at line {line_num} in {path}: {e}"
                    ) from e

        return cls(samples)

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        intent_column: str = "intent_label",
        artist_column: str = "artist_label",
        voice_column: str = "voice_label",
        policy_column: str = "policy_labels",
    ) -> "InputClassifierDataset":
        """Load dataset from HuggingFace Hub.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load
            text_column: Column name for text
            intent_column: Column name for intent labels
            artist_column: Column name for artist labels
            voice_column: Column name for voice labels
            policy_column: Column name for policy labels

        Returns:
            InputClassifierDataset instance

        Raises:
            DatasetLoadError: If dataset cannot be loaded
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise DatasetLoadError("datasets library required: pip install datasets") from e

        try:
            hf_dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            raise DatasetLoadError(f"Failed to load HuggingFace dataset '{dataset_name}': {e}") from e

        samples = []
        for idx, item in enumerate(hf_dataset):
            try:
                sample = InputClassifierSample(
                    text=item[text_column],
                    intent_label=item[intent_column],
                    artist_label=item[artist_column],
                    voice_label=item[voice_column],
                    policy_labels=list(item[policy_column]),
                )
                samples.append(sample)
            except (KeyError, ValueError, TypeError) as e:
                raise DatasetLoadError(f"Invalid data at index {idx}: {e}") from e

        return cls(samples)

    def save_jsonl(self, path: Path | str) -> None:
        """Save dataset to JSONL file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict()) + "\n")


class DataLoader:
    """JAX-compatible data loader with batching and shuffling.

    Uses numpy for data loading (CPU-bound) and converts to JAX arrays
    only when creating batches for efficient memory usage.
    """

    def __init__(
        self,
        dataset: InputClassifierDataset,
        tokenizer: Any,
        batch_size: int,
        max_length: int = 512,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialize data loader.

        Args:
            dataset: InputClassifierDataset to load from
            tokenizer: HuggingFace tokenizer for text encoding
            batch_size: Number of samples per batch
            max_length: Maximum sequence length for tokenization
            shuffle: Whether to shuffle data each epoch
            drop_last: Whether to drop incomplete final batch
            seed: Random seed for shuffling (None for random)
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return number of batches."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[dict[str, jnp.ndarray]]:
        """Iterate over batches."""
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            self.rng.shuffle(indices)

        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]

            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            batch_samples = [self.dataset[i] for i in batch_indices]
            yield self._collate(batch_samples)

    def _collate(self, samples: list[InputClassifierSample]) -> dict[str, jnp.ndarray]:
        """Collate samples into a batch.

        Args:
            samples: List of samples to collate

        Returns:
            Dictionary with batched tensors
        """
        texts = [s.text for s in samples]

        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        # Prepare labels
        intent_labels = np.array([s.intent_label for s in samples], dtype=np.int32)
        artist_labels = np.array([s.artist_label for s in samples], dtype=np.int32)
        voice_labels = np.array([s.voice_label for s in samples], dtype=np.int32)
        policy_labels = np.array([s.policy_labels for s in samples], dtype=np.float32)

        return {
            "input_ids": jnp.array(encodings["input_ids"]),
            "attention_mask": jnp.array(encodings["attention_mask"]),
            "intent_labels": jnp.array(intent_labels),
            "artist_labels": jnp.array(artist_labels),
            "voice_labels": jnp.array(voice_labels),
            "policy_labels": jnp.array(policy_labels),
        }


def create_dummy_dataset(num_samples: int = 100, seed: int = 42) -> InputClassifierDataset:
    """Create a dummy dataset for testing.

    Args:
        num_samples: Number of samples to generate
        seed: Random seed

    Returns:
        InputClassifierDataset with random samples
    """
    rng = np.random.default_rng(seed)

    example_texts = [
        "Generate a peaceful piano melody",
        "Create music in the style of a famous artist",
        "Make a song with celebrity voice",
        "Generate harmful content",
        "Create a relaxing ambient track",
        "Produce a rock song",
        "Make electronic dance music",
        "Generate a classical symphony",
    ]

    samples = []
    for _ in range(num_samples):
        text = rng.choice(example_texts)
        sample = InputClassifierSample(
            text=text,
            intent_label=int(rng.integers(0, 3)),
            artist_label=int(rng.integers(0, 3)),
            voice_label=int(rng.integers(0, 3)),
            policy_labels=[int(rng.integers(0, 2)) for _ in range(7)],
        )
        samples.append(sample)

    return InputClassifierDataset(samples)
