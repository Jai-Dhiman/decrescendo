"""Tests for data loading."""

import tempfile
from pathlib import Path

import pytest

from decrescendo.constitutional_audio.data import (
    DatasetLoadError,
    InputClassifierDataset,
    InputClassifierSample,
    create_dummy_dataset,
)


class TestInputClassifierSample:
    """Test suite for InputClassifierSample."""

    def test_valid_sample(self):
        """Test creating a valid sample."""
        sample = InputClassifierSample(
            text="Test prompt",
            intent_label=0,
            artist_label=1,
            voice_label=2,
            policy_labels=[0, 1, 0, 0, 1, 0, 0],
        )
        assert sample.text == "Test prompt"
        assert sample.intent_label == 0

    def test_invalid_intent_label(self):
        """Test that invalid intent label raises error."""
        with pytest.raises(ValueError, match="intent_label must be 0-2"):
            InputClassifierSample(
                text="Test",
                intent_label=5,
                artist_label=0,
                voice_label=0,
                policy_labels=[0] * 7,
            )

    def test_invalid_policy_labels_length(self):
        """Test that wrong policy labels length raises error."""
        with pytest.raises(ValueError, match="policy_labels must have 7 elements"):
            InputClassifierSample(
                text="Test",
                intent_label=0,
                artist_label=0,
                voice_label=0,
                policy_labels=[0] * 5,
            )

    def test_invalid_policy_labels_values(self):
        """Test that non-binary policy labels raise error."""
        with pytest.raises(ValueError, match="policy_labels must contain only 0 or 1"):
            InputClassifierSample(
                text="Test",
                intent_label=0,
                artist_label=0,
                voice_label=0,
                policy_labels=[0, 1, 2, 0, 0, 0, 0],
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        sample = InputClassifierSample(
            text="Test",
            intent_label=1,
            artist_label=2,
            voice_label=0,
            policy_labels=[1, 0, 1, 0, 0, 0, 0],
        )
        d = sample.to_dict()
        assert d["text"] == "Test"
        assert d["intent_label"] == 1
        assert d["policy_labels"] == [1, 0, 1, 0, 0, 0, 0]


class TestInputClassifierDataset:
    """Test suite for InputClassifierDataset."""

    def test_from_jsonl(self):
        """Test loading from JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "Test 1", "intent_label": 0, "artist_label": 0, "voice_label": 0, "policy_labels": [0,0,0,0,0,0,0]}\n')
            f.write('{"text": "Test 2", "intent_label": 1, "artist_label": 1, "voice_label": 1, "policy_labels": [1,0,0,0,0,0,0]}\n')
            f.flush()

            dataset = InputClassifierDataset.from_jsonl(Path(f.name))

        assert len(dataset) == 2
        assert dataset[0].text == "Test 1"
        assert dataset[1].intent_label == 1

    def test_from_jsonl_missing_file(self):
        """Test that missing file raises error."""
        with pytest.raises(DatasetLoadError, match="not found"):
            InputClassifierDataset.from_jsonl(Path("/nonexistent/file.jsonl"))

    def test_from_jsonl_invalid_json(self):
        """Test that invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not valid json\n")
            f.flush()

            with pytest.raises(DatasetLoadError, match="Invalid JSON"):
                InputClassifierDataset.from_jsonl(Path(f.name))

    def test_from_jsonl_missing_keys(self):
        """Test that missing keys raise error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "Test"}\n')
            f.flush()

            with pytest.raises(DatasetLoadError, match="Missing keys"):
                InputClassifierDataset.from_jsonl(Path(f.name))


class TestDummyDataset:
    """Test suite for dummy dataset creation."""

    def test_create_dummy_dataset(self):
        """Test creating dummy dataset."""
        dataset = create_dummy_dataset(num_samples=50, seed=42)
        assert len(dataset) == 50

    def test_dummy_dataset_deterministic(self):
        """Test that same seed produces same dataset."""
        dataset1 = create_dummy_dataset(num_samples=10, seed=123)
        dataset2 = create_dummy_dataset(num_samples=10, seed=123)

        for i in range(10):
            assert dataset1[i].text == dataset2[i].text
            assert dataset1[i].intent_label == dataset2[i].intent_label
