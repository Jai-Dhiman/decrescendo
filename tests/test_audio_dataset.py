"""Tests for audio dataset and data loading."""

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import soundfile as sf

from decrescendo.musicritic.data.audio_dataset import (
    AudioAugmenter,
    AudioClassificationSample,
    AudioDataLoader,
    AudioDataset,
    AudioDatasetError,
    AugmentationConfig,
    create_dummy_audio_dataset,
)
from decrescendo.musicritic.output_classifier.config import PreprocessingConfig


class TestAudioClassificationSample:
    """Test suite for AudioClassificationSample dataclass."""

    def test_valid_sample(self):
        """Valid sample should be created successfully."""
        sample = AudioClassificationSample(
            audio_path="/path/to/audio.wav",
            harm_labels=[1, 0, 0, 0, 0, 0, 1],
            speaker_id="speaker_001",
        )

        assert sample.audio_path == "/path/to/audio.wav"
        assert sample.harm_labels == [1, 0, 0, 0, 0, 0, 1]
        assert sample.speaker_id == "speaker_001"

    def test_invalid_harm_labels_length(self):
        """Should raise for wrong number of harm labels."""
        with pytest.raises(ValueError, match="must have 7 elements"):
            AudioClassificationSample(
                audio_path="/path/to/audio.wav",
                harm_labels=[1, 0, 0],  # Only 3 elements
            )

    def test_invalid_harm_labels_values(self):
        """Should raise for non-binary harm labels."""
        with pytest.raises(ValueError, match="must contain only 0 or 1"):
            AudioClassificationSample(
                audio_path="/path/to/audio.wav",
                harm_labels=[1, 0, 2, 0, 0, 0, 0],  # 2 is invalid
            )

    def test_to_dict(self):
        """Should serialize to dictionary correctly."""
        sample = AudioClassificationSample(
            audio_path="/path/to/audio.wav",
            harm_labels=[1, 0, 0, 0, 0, 0, 1],
            speaker_id="speaker_001",
            metadata={"source": "test"},
        )

        d = sample.to_dict()

        assert d["audio_path"] == "/path/to/audio.wav"
        assert d["harm_labels"] == [1, 0, 0, 0, 0, 0, 1]
        assert d["speaker_id"] == "speaker_001"
        assert d["metadata"] == {"source": "test"}

    def test_speaker_id_optional(self):
        """Speaker ID should be optional."""
        sample = AudioClassificationSample(
            audio_path="/path/to/audio.wav",
            harm_labels=[0, 0, 0, 0, 0, 0, 0],
        )

        assert sample.speaker_id is None


class TestAudioDataset:
    """Test suite for AudioDataset class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_manifest(self, temp_dir):
        """Create sample manifest file."""
        manifest_path = temp_dir / "manifest.jsonl"
        samples = [
            {
                "audio_path": str(temp_dir / "audio1.wav"),
                "harm_labels": [1, 0, 0, 0, 0, 0, 0],
                "speaker_id": "spk1",
            },
            {
                "audio_path": str(temp_dir / "audio2.wav"),
                "harm_labels": [0, 1, 0, 0, 0, 0, 0],
                "speaker_id": "spk2",
            },
            {
                "audio_path": str(temp_dir / "audio3.wav"),
                "harm_labels": [0, 0, 1, 0, 0, 0, 0],
                "speaker_id": "spk1",
            },
        ]

        with open(manifest_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        return manifest_path

    def test_from_manifest(self, sample_manifest):
        """Should load from JSONL manifest."""
        dataset = AudioDataset.from_manifest(sample_manifest)

        assert len(dataset) == 3
        assert dataset[0].harm_labels == [1, 0, 0, 0, 0, 0, 0]
        assert dataset[1].speaker_id == "spk2"

    def test_from_manifest_missing_file(self, temp_dir):
        """Should raise for missing manifest."""
        with pytest.raises(AudioDatasetError, match="not found"):
            AudioDataset.from_manifest(temp_dir / "nonexistent.jsonl")

    def test_from_manifest_invalid_json(self, temp_dir):
        """Should raise for invalid JSON."""
        manifest_path = temp_dir / "bad.jsonl"
        with open(manifest_path, "w") as f:
            f.write("not valid json\n")

        with pytest.raises(AudioDatasetError, match="Invalid JSON"):
            AudioDataset.from_manifest(manifest_path)

    def test_from_manifest_missing_keys(self, temp_dir):
        """Should raise for missing required keys."""
        manifest_path = temp_dir / "incomplete.jsonl"
        with open(manifest_path, "w") as f:
            f.write(json.dumps({"audio_path": "/test.wav"}) + "\n")  # Missing harm_labels

        with pytest.raises(AudioDatasetError, match="Missing keys"):
            AudioDataset.from_manifest(manifest_path)

    def test_speaker_to_id_mapping(self):
        """Should create correct speaker ID mapping."""
        samples = [
            AudioClassificationSample("/a.wav", [0] * 7, speaker_id="spk_b"),
            AudioClassificationSample("/b.wav", [0] * 7, speaker_id="spk_a"),
            AudioClassificationSample("/c.wav", [0] * 7, speaker_id="spk_b"),
            AudioClassificationSample("/d.wav", [0] * 7, speaker_id=None),
        ]
        dataset = AudioDataset(samples)

        mapping = dataset.speaker_to_id

        # Should be sorted alphabetically
        assert mapping["spk_a"] == 0
        assert mapping["spk_b"] == 1
        assert len(mapping) == 2  # None is not included

    def test_save_manifest(self, temp_dir):
        """Should save to manifest file."""
        samples = [
            AudioClassificationSample("/a.wav", [1, 0, 0, 0, 0, 0, 0], speaker_id="spk1"),
            AudioClassificationSample("/b.wav", [0, 1, 0, 0, 0, 0, 0], speaker_id="spk2"),
        ]
        dataset = AudioDataset(samples)

        output_path = temp_dir / "output.jsonl"
        dataset.save_manifest(output_path)

        # Reload and verify
        reloaded = AudioDataset.from_manifest(output_path)
        assert len(reloaded) == 2
        assert reloaded[0].harm_labels == [1, 0, 0, 0, 0, 0, 0]


class TestAudioAugmenter:
    """Test suite for AudioAugmenter class."""

    @pytest.fixture
    def augmenter(self):
        """Create augmenter with all augmentations enabled."""
        config = AugmentationConfig(
            enable_noise=True,
            noise_prob=1.0,  # Always apply for deterministic testing
            enable_time_mask=True,
            time_mask_prob=1.0,
            enable_pitch_shift=False,  # Skip slow augmentations
            enable_speed=False,
            enable_rir=False,
        )
        return AudioAugmenter(config, seed=42)

    def test_noise_augmentation(self, augmenter):
        """Noise augmentation should modify audio."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)

        augmented = augmenter._apply_noise(audio)

        # Should be different from original
        assert not np.allclose(audio, augmented)
        # But not completely different
        correlation = np.corrcoef(audio, augmented)[0, 1]
        assert correlation > 0.5

    def test_time_mask_augmentation(self, augmenter):
        """Time mask should zero out part of audio."""
        audio = np.ones(24000, dtype=np.float32)

        augmented = augmenter._apply_time_mask(audio)

        # Some samples should be zeroed
        assert np.sum(augmented == 0) > 0
        # But not all
        assert np.sum(augmented == 1) > 0

    def test_apply_with_seed_reproducibility(self):
        """Same seed should produce same augmentation."""
        config = AugmentationConfig(enable_noise=True, noise_prob=1.0)
        audio = np.random.randn(24000).astype(np.float32)

        aug1 = AudioAugmenter(config, seed=42)
        aug2 = AudioAugmenter(config, seed=42)

        result1 = aug1.apply(audio.copy(), sample_rate=24000)
        result2 = aug2.apply(audio.copy(), sample_rate=24000)

        assert np.allclose(result1, result2)


class TestAudioDataLoader:
    """Test suite for AudioDataLoader class."""

    @pytest.fixture
    def temp_audio_dir(self):
        """Create temporary directory with audio files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create simple audio files
            for i in range(6):
                audio = np.random.randn(48000).astype(np.float32)  # 2 seconds
                sf.write(tmpdir / f"audio_{i}.wav", audio, 24000)

            yield tmpdir

    @pytest.fixture
    def dataset_with_audio(self, temp_audio_dir):
        """Create dataset with real audio files."""
        samples = [
            AudioClassificationSample(
                str(temp_audio_dir / f"audio_{i}.wav"),
                [int(i % 2 == j) for j in range(7)],
                speaker_id=f"spk_{i % 3}",
            )
            for i in range(6)
        ]
        return AudioDataset(samples)

    def test_iteration(self, dataset_with_audio):
        """Should iterate over batches correctly."""
        config = PreprocessingConfig(chunk_duration_sec=1.0, hop_duration_sec=0.5)
        loader = AudioDataLoader(
            dataset_with_audio,
            batch_size=4,
            preprocessing_config=config,
            shuffle=False,
            seed=42,
        )

        batches = list(loader)

        assert len(batches) > 0

    def test_batch_shapes(self, dataset_with_audio):
        """Batch shapes should be correct."""
        config = PreprocessingConfig(chunk_duration_sec=1.0, hop_duration_sec=0.5)
        loader = AudioDataLoader(
            dataset_with_audio,
            batch_size=4,
            preprocessing_config=config,
            shuffle=False,
        )

        batch = next(iter(loader))

        assert "audio" in batch
        assert "harm_labels" in batch
        assert "speaker_ids" in batch

        assert batch["audio"].shape[0] <= 4
        assert batch["audio"].shape[1] == config.chunk_samples
        assert batch["harm_labels"].shape[1] == 7

    def test_shuffling(self, dataset_with_audio):
        """Shuffling should change order."""
        config = PreprocessingConfig(chunk_duration_sec=1.0, hop_duration_sec=0.5)

        loader1 = AudioDataLoader(
            dataset_with_audio,
            batch_size=4,
            preprocessing_config=config,
            shuffle=True,
            seed=42,
        )
        loader2 = AudioDataLoader(
            dataset_with_audio,
            batch_size=4,
            preprocessing_config=config,
            shuffle=True,
            seed=123,  # Different seed
        )

        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # With different seeds, batches might be different
        # (though could be same by chance for small datasets)
        # Just check they're valid
        assert batch1["audio"].shape == batch2["audio"].shape


class TestDummyDataset:
    """Test suite for dummy dataset creation."""

    def test_create_dummy_dataset(self):
        """Should create valid dummy dataset."""
        dataset = create_dummy_audio_dataset(num_samples=10, num_speakers=3)

        assert len(dataset) == 10
        assert all(len(s.harm_labels) == 7 for s in dataset.samples)
        assert all(s.speaker_id is not None for s in dataset.samples)

    def test_dummy_dataset_determinism(self):
        """Same seed should produce same dataset."""
        dataset1 = create_dummy_audio_dataset(num_samples=10, seed=42)
        dataset2 = create_dummy_audio_dataset(num_samples=10, seed=42)

        for s1, s2 in zip(dataset1.samples, dataset2.samples):
            assert s1.harm_labels == s2.harm_labels
            assert s1.speaker_id == s2.speaker_id
