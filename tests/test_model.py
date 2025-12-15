"""Tests for Input Classifier model."""

import jax
import jax.numpy as jnp
import pytest

from decrescendo.constitutional_audio.input_classifier import (
    InputClassifier,
    InputClassifierConfig,
)


class TestInputClassifier:
    """Test suite for InputClassifier model."""

    @pytest.fixture
    def config(self):
        """Create a small config for fast testing."""
        return InputClassifierConfig()

    @pytest.fixture
    def model(self, config):
        """Create model instance."""
        return InputClassifier(config=config)

    def test_model_init(self, model, rng):
        """Test that model initializes correctly."""
        batch_size = 2
        seq_length = 128

        dummy_input = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
        variables = model.init(rng, dummy_input, deterministic=True)

        assert "params" in variables
        assert "embeddings" in variables["params"]
        assert "encoder" in variables["params"]
        assert "pooler" in variables["params"]

    def test_model_forward(self, model, rng):
        """Test forward pass produces correct output shapes."""
        batch_size = 2
        seq_length = 128

        dummy_input = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
        attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)

        variables = model.init(rng, dummy_input, deterministic=True)
        outputs = model.apply(
            variables,
            dummy_input,
            attention_mask=attention_mask,
            deterministic=True,
        )

        # Check output keys
        assert "intent_logits" in outputs
        assert "artist_logits" in outputs
        assert "voice_logits" in outputs
        assert "policy_logits" in outputs
        assert "pooled_output" in outputs
        assert "last_hidden_state" in outputs

        # Check shapes
        config = model.config
        assert outputs["intent_logits"].shape == (batch_size, config.classification.num_intent_classes)
        assert outputs["artist_logits"].shape == (batch_size, config.classification.num_artist_classes)
        assert outputs["voice_logits"].shape == (batch_size, config.classification.num_voice_classes)
        assert outputs["policy_logits"].shape == (batch_size, config.classification.num_policy_labels)
        assert outputs["pooled_output"].shape == (batch_size, config.transformer.hidden_size)
        assert outputs["last_hidden_state"].shape == (batch_size, seq_length, config.transformer.hidden_size)

    def test_model_with_dropout(self, model, rng):
        """Test that dropout produces different outputs in training mode."""
        batch_size = 2
        seq_length = 128

        dummy_input = jnp.ones((batch_size, seq_length), dtype=jnp.int32)

        init_rng, dropout_rng1, dropout_rng2 = jax.random.split(rng, 3)
        variables = model.init(init_rng, dummy_input, deterministic=True)

        # Two forward passes with different dropout keys
        outputs1 = model.apply(
            variables,
            dummy_input,
            deterministic=False,
            rngs={"dropout": dropout_rng1},
        )
        outputs2 = model.apply(
            variables,
            dummy_input,
            deterministic=False,
            rngs={"dropout": dropout_rng2},
        )

        # Outputs should differ due to dropout
        assert not jnp.allclose(outputs1["intent_logits"], outputs2["intent_logits"])

    def test_model_deterministic_mode(self, model, rng):
        """Test that deterministic mode produces same outputs."""
        batch_size = 2
        seq_length = 128

        dummy_input = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
        variables = model.init(rng, dummy_input, deterministic=True)

        # Two forward passes in deterministic mode
        outputs1 = model.apply(variables, dummy_input, deterministic=True)
        outputs2 = model.apply(variables, dummy_input, deterministic=True)

        # Outputs should be identical
        assert jnp.allclose(outputs1["intent_logits"], outputs2["intent_logits"])
