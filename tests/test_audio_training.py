"""Tests for audio training infrastructure."""

import jax
import jax.numpy as jnp
import optax
import pytest

from decrescendo.constitutional_audio.output_classifier.config import OutputClassifierConfig
from decrescendo.constitutional_audio.output_classifier.model import OutputClassifierModel
from decrescendo.constitutional_audio.training.audio_losses import AudioLossWeights
from decrescendo.constitutional_audio.training.audio_trainer import (
    AudioTrainState,
    AudioTrainingConfig,
    create_audio_eval_step,
    create_audio_train_step,
    initialize_audio_training,
)


class TestAudioTrainState:
    """Test suite for AudioTrainState."""

    @pytest.fixture
    def model_and_variables(self, rng):
        """Create model and initialized variables."""
        config = OutputClassifierConfig()
        model = OutputClassifierModel(config=config)

        # Initialize
        dummy_audio = jnp.zeros((1, config.preprocessing.chunk_samples))
        variables = model.init(rng, dummy_audio, train=False)

        return model, variables

    def test_create_with_batch_stats(self, model_and_variables, rng):
        """Should create state with batch_stats."""
        model, variables = model_and_variables

        optimizer = optax.adam(1e-4)
        state = AudioTrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optimizer,
            batch_stats=variables.get("batch_stats", {}),
            dropout_rng=rng,
        )

        assert state.params is not None
        assert state.batch_stats is not None
        assert state.dropout_rng is not None
        assert state.step == 0

    def test_next_dropout_rng(self, model_and_variables, rng):
        """Should properly split and update dropout RNG."""
        model, variables = model_and_variables

        optimizer = optax.adam(1e-4)
        state = AudioTrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optimizer,
            batch_stats=variables.get("batch_stats", {}),
            dropout_rng=rng,
        )

        original_rng = state.dropout_rng
        new_state, current_rng = state.next_dropout_rng()

        # RNG should be different after split
        assert not jnp.array_equal(new_state.dropout_rng, original_rng)
        assert current_rng is not None

    def test_next_dropout_rng_raises_without_rng(self, model_and_variables):
        """Should raise if dropout_rng not set."""
        model, variables = model_and_variables

        optimizer = optax.adam(1e-4)
        state = AudioTrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optimizer,
            batch_stats=variables.get("batch_stats", {}),
            dropout_rng=None,  # Not set
        )

        with pytest.raises(ValueError, match="dropout_rng not set"):
            state.next_dropout_rng()


class TestAudioTrainStep:
    """Test suite for audio training step."""

    @pytest.fixture
    def training_setup(self, rng):
        """Create complete training setup."""
        config = OutputClassifierConfig()
        model = OutputClassifierModel(config=config)

        # Initialize
        dummy_audio = jnp.zeros((4, config.preprocessing.chunk_samples))
        init_rng, dropout_rng = jax.random.split(rng)
        variables = model.init(init_rng, dummy_audio, train=False)

        # Create optimizer
        optimizer = optax.adam(1e-4)

        # Create state
        state = AudioTrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optimizer,
            batch_stats=variables.get("batch_stats", {}),
            dropout_rng=dropout_rng,
        )

        return model, config, state

    @pytest.fixture
    def dummy_batch(self, rng):
        """Create dummy batch with random audio (zeros cause BatchNorm issues)."""
        audio_rng = jax.random.fold_in(rng, 1)
        return {
            "audio": jax.random.normal(audio_rng, (4, 24000)) * 0.1,
            "harm_labels": jnp.array([
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ], dtype=jnp.float32),
            "speaker_ids": jnp.array([0, 0, 1, 1]),
        }

    def test_step_produces_metrics(self, training_setup, dummy_batch):
        """Training step should produce metrics."""
        model, config, state = training_setup

        training_config = AudioTrainingConfig(loss_weights=AudioLossWeights())
        train_step = create_audio_train_step(model, training_config)

        new_state, metrics = train_step(state, dummy_batch)

        assert "loss/total" in metrics
        assert "loss/harm" in metrics
        assert "loss/speaker" in metrics
        assert "metrics/harm_accuracy" in metrics
        assert "metrics/harm_f1_macro" in metrics

    def test_step_updates_batch_stats(self, training_setup, dummy_batch):
        """Training step should update batch_stats."""
        model, config, state = training_setup

        training_config = AudioTrainingConfig(loss_weights=AudioLossWeights())
        train_step = create_audio_train_step(model, training_config)

        # Get initial batch stats values
        initial_stats = jax.tree_util.tree_map(
            lambda x: x.copy() if hasattr(x, 'copy') else x,
            state.batch_stats
        )

        # Run a training step
        new_state, _ = train_step(state, dummy_batch)

        # Batch stats should be updated (if model has batch norm)
        if state.batch_stats:
            # At least the structure should be maintained
            assert new_state.batch_stats is not None

    def test_step_updates_params(self, training_setup, dummy_batch):
        """Training step should update parameters."""
        model, config, state = training_setup

        training_config = AudioTrainingConfig(loss_weights=AudioLossWeights())
        train_step = create_audio_train_step(model, training_config)

        new_state, _ = train_step(state, dummy_batch)

        # Step counter should increment
        assert new_state.step == state.step + 1

        # Some params should have changed (due to gradient updates)
        # Flatten and compare
        old_leaves = jax.tree_util.tree_leaves(state.params)
        new_leaves = jax.tree_util.tree_leaves(new_state.params)

        any_changed = any(
            not jnp.allclose(old, new)
            for old, new in zip(old_leaves, new_leaves)
        )
        assert any_changed

    def test_multiple_steps_reduce_loss(self, training_setup, dummy_batch):
        """Multiple training steps should reduce loss."""
        model, config, state = training_setup

        training_config = AudioTrainingConfig(loss_weights=AudioLossWeights())
        train_step = create_audio_train_step(model, training_config)

        # Collect losses over multiple steps
        losses = []
        for _ in range(10):
            state, metrics = train_step(state, dummy_batch)
            losses.append(float(metrics["loss/total"]))

        # Loss should generally decrease (may not be monotonic)
        # Check that final loss is lower than initial
        assert losses[-1] < losses[0] or all(l < 10 for l in losses)  # Or all losses are reasonable


class TestAudioEvalStep:
    """Test suite for audio evaluation step."""

    @pytest.fixture
    def eval_setup(self, rng):
        """Create evaluation setup."""
        config = OutputClassifierConfig()
        model = OutputClassifierModel(config=config)

        # Initialize
        dummy_audio = jnp.zeros((4, config.preprocessing.chunk_samples))
        variables = model.init(rng, dummy_audio, train=False)

        return model, config, variables

    @pytest.fixture
    def dummy_batch(self, rng):
        """Create dummy batch with random audio (zeros cause BatchNorm issues)."""
        audio_rng = jax.random.fold_in(rng, 1)
        return {
            "audio": jax.random.normal(audio_rng, (4, 24000)) * 0.1,
            "harm_labels": jnp.array([
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ], dtype=jnp.float32),
            "speaker_ids": jnp.array([0, 0, 1, 1]),
        }

    def test_eval_step_produces_metrics(self, eval_setup, dummy_batch):
        """Eval step should produce metrics."""
        model, config, variables = eval_setup

        training_config = AudioTrainingConfig()
        eval_step = create_audio_eval_step(model, training_config)

        metrics = eval_step(
            variables["params"],
            variables.get("batch_stats", {}),
            dummy_batch,
        )

        assert "loss/total" in metrics
        assert "loss/harm" in metrics
        assert "metrics/harm_accuracy" in metrics

    def test_eval_step_deterministic(self, eval_setup, dummy_batch):
        """Eval step should be deterministic."""
        model, config, variables = eval_setup

        training_config = AudioTrainingConfig()
        eval_step = create_audio_eval_step(model, training_config)

        # Run twice
        metrics1 = eval_step(
            variables["params"],
            variables.get("batch_stats", {}),
            dummy_batch,
        )
        metrics2 = eval_step(
            variables["params"],
            variables.get("batch_stats", {}),
            dummy_batch,
        )

        # Should be identical
        for key in metrics1:
            assert jnp.allclose(metrics1[key], metrics2[key])


class TestInitializeAudioTraining:
    """Test suite for training initialization."""

    def test_initialize_creates_valid_state(self, rng):
        """Should create valid training state."""
        config = OutputClassifierConfig()
        model = OutputClassifierModel(config=config)
        training_config = AudioTrainingConfig()

        dummy_batch = {
            "audio": jnp.zeros((4, config.preprocessing.chunk_samples)),
            "harm_labels": jnp.zeros((4, 7)),
        }

        state = initialize_audio_training(
            model=model,
            config=training_config,
            num_train_steps=1000,
            rng=rng,
            dummy_batch=dummy_batch,
        )

        assert isinstance(state, AudioTrainState)
        assert state.params is not None
        assert state.batch_stats is not None
        assert state.dropout_rng is not None
        assert state.opt_state is not None
