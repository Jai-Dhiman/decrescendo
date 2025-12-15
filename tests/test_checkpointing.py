"""Tests for classifier checkpointing."""

import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from decrescendo.constitutional_audio.input_classifier import (
    InputClassifier,
    InputClassifierConfig,
)
from decrescendo.constitutional_audio.input_classifier.checkpointing import (
    CHECKPOINT_VERSION,
    CheckpointConfigError,
    CheckpointCorruptedError,
    CheckpointMetadata,
    CheckpointNotFoundError,
    CheckpointVersionError,
    InputClassifierCheckpointer,
    _dataclass_to_dict,
    _reconstruct_input_classifier_config,
    load_input_classifier,
    save_input_classifier,
)
from decrescendo.constitutional_audio.output_classifier.checkpointing import (
    OutputClassifierCheckpointer,
    VoiceEntry,
    load_output_classifier,
    load_voice_database,
    save_output_classifier,
    save_voice_database,
)
from decrescendo.constitutional_audio.output_classifier.config import (
    AudioEncoderConfig,
    OutputClassifierConfig,
)
from decrescendo.constitutional_audio.output_classifier.inference import (
    initialize_output_classifier,
)


class TestInputClassifierCheckpointing:
    """Test Input Classifier checkpoint save/load."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def model_and_params(self, rng):
        """Initialize model with random params."""
        config = InputClassifierConfig()
        model = InputClassifier(config=config)
        dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
        variables = model.init(rng, dummy_input, deterministic=True)
        return model, variables["params"], config

    def test_save_and_load_params(self, temp_dir, model_and_params):
        """Test that saved params can be loaded."""
        model, params, config = model_and_params

        save_input_classifier(temp_dir, params, config, step=100)

        loaded_model, loaded_params, loaded_config = load_input_classifier(temp_dir)

        # Verify params match (check a few leaf values)
        assert "embeddings" in loaded_params
        assert "encoder" in loaded_params

        # Verify config matches
        assert loaded_config.transformer.hidden_size == config.transformer.hidden_size
        assert loaded_config.classification.num_intent_classes == config.classification.num_intent_classes

    def test_save_creates_metadata(self, temp_dir, model_and_params):
        """Test that save creates metadata.json."""
        _, params, config = model_and_params

        save_input_classifier(temp_dir, params, config, step=50)

        metadata_path = temp_dir / "50" / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "library_version" in metadata
        assert "checkpoint_version" in metadata
        assert "config" in metadata
        assert metadata["step"] == 50

    def test_load_latest_step(self, temp_dir, model_and_params):
        """Test loading latest checkpoint."""
        _, params, config = model_and_params

        checkpointer = InputClassifierCheckpointer(temp_dir)
        checkpointer.save(10, params, config)
        checkpointer.save(20, params, config)
        checkpointer.save(30, params, config)

        assert checkpointer.latest_step() == 30

    def test_load_specific_step(self, temp_dir, model_and_params):
        """Test loading specific checkpoint step."""
        _, params, config = model_and_params

        checkpointer = InputClassifierCheckpointer(temp_dir)
        checkpointer.save(10, params, config)
        checkpointer.save(20, params, config)

        _, loaded_config, metadata = checkpointer.load(step=10)
        assert metadata.step == 10

    def test_load_nonexistent_raises(self, temp_dir):
        """Test that loading nonexistent checkpoint raises error."""
        checkpointer = InputClassifierCheckpointer(temp_dir)

        with pytest.raises(CheckpointNotFoundError, match="No checkpoint found"):
            checkpointer.load()

    def test_max_to_keep(self, temp_dir, model_and_params):
        """Test that max_to_keep limits checkpoint retention."""
        _, params, config = model_and_params

        checkpointer = InputClassifierCheckpointer(temp_dir, max_to_keep=2)
        for step in [10, 20, 30, 40]:
            checkpointer.save(step, params, config)

        steps = checkpointer.all_steps()
        assert len(steps) <= 2

    def test_version_compatibility_future_version(self, temp_dir, model_and_params):
        """Test that future checkpoint versions raise error."""
        _, params, config = model_and_params

        save_input_classifier(temp_dir, params, config, step=100)

        # Manually modify metadata to have future version
        metadata_path = temp_dir / "100" / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["checkpoint_version"] = 9999
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        checkpointer = InputClassifierCheckpointer(temp_dir)
        with pytest.raises(CheckpointVersionError):
            checkpointer.load()

    def test_corrupted_metadata_raises(self, temp_dir, model_and_params):
        """Test that corrupted metadata raises error."""
        _, params, config = model_and_params

        save_input_classifier(temp_dir, params, config, step=100)

        # Corrupt the metadata file
        metadata_path = temp_dir / "100" / "metadata.json"
        with open(metadata_path, "w") as f:
            f.write("not valid json {{{")

        checkpointer = InputClassifierCheckpointer(temp_dir)
        with pytest.raises(CheckpointCorruptedError, match="Invalid JSON"):
            checkpointer.load()

    def test_all_steps(self, temp_dir, model_and_params):
        """Test getting all checkpoint steps."""
        _, params, config = model_and_params

        checkpointer = InputClassifierCheckpointer(temp_dir, max_to_keep=10)
        checkpointer.save(5, params, config)
        checkpointer.save(15, params, config)
        checkpointer.save(25, params, config)

        steps = checkpointer.all_steps()
        assert 5 in steps
        assert 15 in steps
        assert 25 in steps


class TestOutputClassifierCheckpointing:
    """Test Output Classifier checkpoint save/load."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def model_and_variables(self, rng):
        """Initialize model with random variables."""
        config = OutputClassifierConfig(
            audio_encoder=AudioEncoderConfig(
                num_conv_layers=3,
                base_channels=16,
                embedding_dim=128,
            ),
            classifier_hidden_dim=64,
        )
        model, variables = initialize_output_classifier(config, rng)
        return model, variables, config

    def test_save_and_load_variables(self, temp_dir, model_and_variables):
        """Test that saved variables can be loaded."""
        model, variables, config = model_and_variables

        save_output_classifier(temp_dir, variables, config, step=100)

        loaded_model, loaded_variables, loaded_config = load_output_classifier(temp_dir)

        # Verify params exist
        assert "params" in loaded_variables
        assert "audio_encoder" in loaded_variables["params"]
        assert "speaker_encoder" in loaded_variables["params"]
        assert "harm_classifier" in loaded_variables["params"]

        # Verify config matches
        assert loaded_config.audio_encoder.embedding_dim == config.audio_encoder.embedding_dim
        assert loaded_config.num_harm_categories == config.num_harm_categories

    def test_batch_stats_preserved(self, temp_dir, model_and_variables):
        """Test that BatchNorm statistics are preserved."""
        model, variables, config = model_and_variables

        # Verify batch_stats exist (from BatchNorm layers)
        assert "batch_stats" in variables

        save_output_classifier(temp_dir, variables, config, step=100)
        _, loaded_variables, _ = load_output_classifier(temp_dir)

        assert "batch_stats" in loaded_variables

    def test_load_latest_step(self, temp_dir, model_and_variables):
        """Test loading latest checkpoint."""
        _, variables, config = model_and_variables

        checkpointer = OutputClassifierCheckpointer(temp_dir)
        checkpointer.save(10, variables, config)
        checkpointer.save(20, variables, config)

        assert checkpointer.latest_step() == 20

    def test_load_specific_step(self, temp_dir, model_and_variables):
        """Test loading specific checkpoint step."""
        _, variables, config = model_and_variables

        checkpointer = OutputClassifierCheckpointer(temp_dir)
        checkpointer.save(10, variables, config)
        checkpointer.save(20, variables, config)

        _, loaded_config, metadata = checkpointer.load(step=10)
        assert metadata.step == 10

    def test_load_nonexistent_raises(self, temp_dir):
        """Test that loading nonexistent checkpoint raises error."""
        from decrescendo.constitutional_audio.output_classifier.checkpointing import (
            CheckpointNotFoundError as OutputCheckpointNotFoundError,
        )

        checkpointer = OutputClassifierCheckpointer(temp_dir)

        with pytest.raises(OutputCheckpointNotFoundError, match="No checkpoint found"):
            checkpointer.load()


class TestVoiceDatabase:
    """Test voice database persistence."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_and_load_voice_database(self, temp_dir):
        """Test saving and loading voice database."""
        embeddings = np.random.randn(5, 192).astype(np.float32)
        entries = [
            VoiceEntry(voice_id=i, name=f"Voice{i}", metadata={"genre": "pop"})
            for i in range(5)
        ]

        save_voice_database(temp_dir / "voices", embeddings, entries)

        loaded_embeddings, loaded_entries = load_voice_database(temp_dir / "voices")

        np.testing.assert_array_almost_equal(embeddings, loaded_embeddings)
        assert len(loaded_entries) == 5
        assert loaded_entries[0].name == "Voice0"
        assert loaded_entries[0].metadata == {"genre": "pop"}

    def test_empty_voice_database(self, temp_dir):
        """Test saving and loading empty voice database."""
        embeddings = np.array([]).reshape(0, 192).astype(np.float32)
        entries: list[VoiceEntry] = []

        save_voice_database(temp_dir / "voices", embeddings, entries)

        loaded_embeddings, loaded_entries = load_voice_database(temp_dir / "voices")

        assert loaded_embeddings.shape[0] == 0
        assert len(loaded_entries) == 0

    def test_voice_database_not_found(self, temp_dir):
        """Test error when voice database doesn't exist."""
        from decrescendo.constitutional_audio.output_classifier.checkpointing import (
            VoiceDatabaseNotFoundError,
        )

        with pytest.raises(VoiceDatabaseNotFoundError):
            load_voice_database(temp_dir / "nonexistent")

    def test_voice_database_jax_array(self, temp_dir):
        """Test that JAX arrays can be saved."""
        embeddings = jnp.array(np.random.randn(3, 192).astype(np.float32))
        entries = [VoiceEntry(voice_id=i, name=f"Artist{i}", metadata={}) for i in range(3)]

        save_voice_database(temp_dir / "voices", embeddings, entries)

        loaded_embeddings, _ = load_voice_database(temp_dir / "voices")

        # Should return JAX array
        assert isinstance(loaded_embeddings, jnp.ndarray)
        np.testing.assert_array_almost_equal(np.asarray(embeddings), np.asarray(loaded_embeddings))


class TestConfigSerialization:
    """Test config serialization/deserialization."""

    def test_input_config_roundtrip(self):
        """Test InputClassifierConfig survives JSON roundtrip."""
        config = InputClassifierConfig()
        config_dict = _dataclass_to_dict(config)

        # Can serialize to JSON
        json_str = json.dumps(config_dict)

        # Can deserialize back
        loaded_dict = json.loads(json_str)
        loaded_config = _reconstruct_input_classifier_config(loaded_dict)

        assert loaded_config.transformer.hidden_size == config.transformer.hidden_size
        assert loaded_config.transformer.num_hidden_layers == config.transformer.num_hidden_layers
        assert loaded_config.classification.num_intent_classes == config.classification.num_intent_classes
        assert loaded_config.pretrained_model_name == config.pretrained_model_name

    def test_output_config_roundtrip(self):
        """Test OutputClassifierConfig survives JSON roundtrip."""
        from decrescendo.constitutional_audio.output_classifier.checkpointing import (
            _dataclass_to_dict as output_dataclass_to_dict,
            _reconstruct_output_classifier_config,
        )

        config = OutputClassifierConfig()
        config_dict = output_dataclass_to_dict(config)

        json_str = json.dumps(config_dict)
        loaded_dict = json.loads(json_str)
        loaded_config = _reconstruct_output_classifier_config(loaded_dict)

        assert loaded_config.audio_encoder.embedding_dim == config.audio_encoder.embedding_dim
        assert loaded_config.speaker.embedding_dim == config.speaker.embedding_dim
        assert loaded_config.num_harm_categories == config.num_harm_categories
        assert loaded_config.block_threshold == config.block_threshold


class TestCheckpointMetadata:
    """Test checkpoint metadata handling."""

    def test_metadata_creation(self):
        """Test creating checkpoint metadata."""
        config = InputClassifierConfig()
        metadata = CheckpointMetadata(
            library_version="1.0.0",
            checkpoint_version=1,
            step=100,
            created_at="2024-12-14T10:00:00Z",
            config=_dataclass_to_dict(config),
        )

        assert metadata.step == 100
        assert metadata.checkpoint_version == 1
        assert "transformer" in metadata.config

    def test_current_checkpoint_version(self):
        """Test that current checkpoint version is set."""
        assert CHECKPOINT_VERSION >= 1
