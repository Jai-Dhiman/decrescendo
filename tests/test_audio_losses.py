"""Tests for audio loss functions."""

import jax
import jax.numpy as jnp
import pytest

from decrescendo.constitutional_audio.training.audio_losses import (
    AudioLossOutput,
    AudioLossWeights,
    combined_audio_loss,
    harm_classification_loss,
    speaker_contrastive_loss,
    speaker_triplet_loss,
)


class TestHarmClassificationLoss:
    """Test suite for harm classification loss."""

    def test_perfect_predictions(self):
        """Loss should be near zero for perfect predictions."""
        # Perfect predictions: logits strongly favor correct labels
        labels = jnp.array([[1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0]], dtype=jnp.float32)
        # Large positive logits where label=1, large negative where label=0
        logits = jnp.array([[10, -10, 10, -10, -10, -10, 10], [-10, 10, -10, -10, 10, -10, -10]], dtype=jnp.float32)

        loss = harm_classification_loss(logits, labels)

        # Loss should be very small
        assert loss < 0.01

    def test_random_predictions(self):
        """Loss should be positive for random predictions."""
        rng = jax.random.PRNGKey(42)
        labels = jax.random.bernoulli(rng, 0.3, shape=(4, 7)).astype(jnp.float32)
        logits = jax.random.normal(rng, shape=(4, 7))

        loss = harm_classification_loss(logits, labels)

        assert loss > 0

    def test_label_smoothing(self):
        """Label smoothing should increase loss for perfect predictions."""
        labels = jnp.array([[1, 0, 1, 0, 0, 0, 1]], dtype=jnp.float32)
        logits = jnp.array([[10, -10, 10, -10, -10, -10, 10]], dtype=jnp.float32)

        loss_no_smoothing = harm_classification_loss(logits, labels, label_smoothing=0.0)
        loss_with_smoothing = harm_classification_loss(logits, labels, label_smoothing=0.1)

        # With smoothing, loss should be higher for perfect predictions
        assert loss_with_smoothing > loss_no_smoothing

    def test_shape_mismatch_raises(self):
        """Should raise error for mismatched shapes."""
        labels = jnp.zeros((4, 7))
        logits = jnp.zeros((4, 5))

        with pytest.raises(ValueError, match="must match"):
            harm_classification_loss(logits, labels)

    def test_invalid_label_smoothing_raises(self):
        """Should raise error for invalid label smoothing."""
        labels = jnp.zeros((4, 7))
        logits = jnp.zeros((4, 7))

        with pytest.raises(ValueError, match="label_smoothing"):
            harm_classification_loss(logits, labels, label_smoothing=1.5)

    def test_gradient_flow(self, rng):
        """Gradients should flow through loss computation."""
        labels = jnp.array([[1, 0, 1, 0, 0, 0, 1]], dtype=jnp.float32)
        logits = jax.random.normal(rng, shape=(1, 7))

        def loss_fn(logits):
            return harm_classification_loss(logits, labels)

        grads = jax.grad(loss_fn)(logits)

        # Gradients should be non-zero
        assert jnp.any(grads != 0)


class TestSpeakerContrastiveLoss:
    """Test suite for speaker contrastive loss."""

    def test_same_speaker_batch(self):
        """Loss should be defined when all samples are same speaker."""
        rng = jax.random.PRNGKey(42)
        embeddings = jax.random.normal(rng, shape=(4, 192))
        embeddings = embeddings / jnp.linalg.norm(embeddings, axis=-1, keepdims=True)

        # All same speaker
        speaker_ids = jnp.array([0, 0, 0, 0])

        loss = speaker_contrastive_loss(embeddings, speaker_ids)

        # With all same speaker, each sample has positives (all others)
        assert loss >= 0

    def test_all_different_speakers(self):
        """Loss should be 0 when no sample has positives."""
        rng = jax.random.PRNGKey(42)
        embeddings = jax.random.normal(rng, shape=(4, 192))
        embeddings = embeddings / jnp.linalg.norm(embeddings, axis=-1, keepdims=True)

        # All different speakers (no positives exist)
        speaker_ids = jnp.array([0, 1, 2, 3])

        loss = speaker_contrastive_loss(embeddings, speaker_ids)

        # Should be 0 since no sample has positives
        assert loss == 0.0

    def test_mixed_speakers(self):
        """Loss computation with mixed speaker batch."""
        rng = jax.random.PRNGKey(42)
        embeddings = jax.random.normal(rng, shape=(6, 192))
        embeddings = embeddings / jnp.linalg.norm(embeddings, axis=-1, keepdims=True)

        # Mixed: 2 speakers with 3 samples each
        speaker_ids = jnp.array([0, 0, 0, 1, 1, 1])

        loss = speaker_contrastive_loss(embeddings, speaker_ids)

        assert loss > 0

    def test_normalized_embeddings(self):
        """Loss should work correctly with L2-normalized embeddings."""
        # Create embeddings that are explicitly normalized
        embeddings = jnp.array([
            [1.0, 0.0, 0.0],
            [0.707, 0.707, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        speaker_ids = jnp.array([0, 0, 1, 1])

        loss = speaker_contrastive_loss(embeddings, speaker_ids)

        # Should compute without errors
        assert jnp.isfinite(loss)

    def test_single_sample_returns_zero(self):
        """Should return 0 for batch size of 1."""
        embeddings = jnp.array([[1.0, 0.0, 0.0]])
        speaker_ids = jnp.array([0])

        loss = speaker_contrastive_loss(embeddings, speaker_ids)

        assert loss == 0.0

    def test_temperature_effect(self):
        """Lower temperature should make loss sharper."""
        rng = jax.random.PRNGKey(42)
        embeddings = jax.random.normal(rng, shape=(4, 192))
        embeddings = embeddings / jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
        speaker_ids = jnp.array([0, 0, 1, 1])

        loss_high_temp = speaker_contrastive_loss(embeddings, speaker_ids, temperature=1.0)
        loss_low_temp = speaker_contrastive_loss(embeddings, speaker_ids, temperature=0.01)

        # Both should be finite
        assert jnp.isfinite(loss_high_temp)
        assert jnp.isfinite(loss_low_temp)


class TestSpeakerTripletLoss:
    """Test suite for speaker triplet loss."""

    def test_correct_ordering(self):
        """Loss should be 0 when anchor is closer to positive."""
        # Anchor and positive are similar, negative is far
        anchor = jnp.array([[1.0, 0.0, 0.0]])
        positive = jnp.array([[0.9, 0.1, 0.0]])
        negative = jnp.array([[-1.0, 0.0, 0.0]])

        # Normalize
        anchor = anchor / jnp.linalg.norm(anchor, axis=-1, keepdims=True)
        positive = positive / jnp.linalg.norm(positive, axis=-1, keepdims=True)
        negative = negative / jnp.linalg.norm(negative, axis=-1, keepdims=True)

        loss = speaker_triplet_loss(anchor, positive, negative, margin=0.2)

        # Should be 0 or very small since ordering is correct
        assert loss < 0.1

    def test_incorrect_ordering(self):
        """Loss should be positive when anchor is closer to negative."""
        # Anchor and negative are similar, positive is far
        anchor = jnp.array([[1.0, 0.0, 0.0]])
        positive = jnp.array([[-1.0, 0.0, 0.0]])
        negative = jnp.array([[0.9, 0.1, 0.0]])

        # Normalize
        anchor = anchor / jnp.linalg.norm(anchor, axis=-1, keepdims=True)
        positive = positive / jnp.linalg.norm(positive, axis=-1, keepdims=True)
        negative = negative / jnp.linalg.norm(negative, axis=-1, keepdims=True)

        loss = speaker_triplet_loss(anchor, positive, negative, margin=0.2)

        # Should be positive
        assert loss > 0

    def test_margin_effect(self):
        """Larger margin should increase loss."""
        anchor = jnp.array([[1.0, 0.0, 0.0]])
        positive = jnp.array([[0.8, 0.2, 0.0]])
        negative = jnp.array([[0.5, 0.5, 0.0]])

        # Normalize
        anchor = anchor / jnp.linalg.norm(anchor, axis=-1, keepdims=True)
        positive = positive / jnp.linalg.norm(positive, axis=-1, keepdims=True)
        negative = negative / jnp.linalg.norm(negative, axis=-1, keepdims=True)

        loss_small_margin = speaker_triplet_loss(anchor, positive, negative, margin=0.1)
        loss_large_margin = speaker_triplet_loss(anchor, positive, negative, margin=0.5)

        assert loss_large_margin >= loss_small_margin


class TestCombinedAudioLoss:
    """Test suite for combined audio loss."""

    def test_weights_applied(self):
        """Loss weights should affect total correctly."""
        outputs = {
            "harm_logits": jnp.zeros((4, 7)),
            "speaker_embeddings": jnp.ones((4, 192)) / jnp.sqrt(192),
        }
        batch = {
            "harm_labels": jnp.ones((4, 7)),
            "speaker_ids": jnp.array([0, 0, 1, 1]),
        }

        # Different weights
        weights_equal = AudioLossWeights(harm=1.0, speaker=1.0)
        weights_harm_only = AudioLossWeights(harm=1.0, speaker=0.0)

        loss_equal = combined_audio_loss(outputs, batch, weights=weights_equal)
        loss_harm_only = combined_audio_loss(outputs, batch, weights=weights_harm_only)

        # Harm-only should have smaller total (no speaker component)
        assert loss_harm_only.total < loss_equal.total

    def test_speaker_loss_optional(self):
        """Should work without speaker_ids in batch."""
        outputs = {
            "harm_logits": jnp.zeros((4, 7)),
            "speaker_embeddings": jnp.ones((4, 192)) / jnp.sqrt(192),
        }
        batch = {
            "harm_labels": jnp.ones((4, 7)),
            # No speaker_ids
        }

        loss = combined_audio_loss(outputs, batch)

        # Speaker loss should be 0
        assert loss.speaker == 0.0
        assert loss.total == loss.harm

    def test_output_structure(self):
        """Should return correct output structure."""
        outputs = {
            "harm_logits": jnp.zeros((4, 7)),
            "speaker_embeddings": jnp.ones((4, 192)) / jnp.sqrt(192),
        }
        batch = {
            "harm_labels": jnp.ones((4, 7)),
            "speaker_ids": jnp.array([0, 0, 1, 1]),
        }

        loss = combined_audio_loss(outputs, batch)

        assert isinstance(loss, AudioLossOutput)
        assert hasattr(loss, "total")
        assert hasattr(loss, "harm")
        assert hasattr(loss, "speaker")

    def test_is_jax_pytree(self):
        """Output should be JAX PyTree compatible."""
        outputs = {
            "harm_logits": jnp.zeros((4, 7)),
            "speaker_embeddings": jnp.ones((4, 192)) / jnp.sqrt(192),
        }
        batch = {
            "harm_labels": jnp.ones((4, 7)),
        }

        loss = combined_audio_loss(outputs, batch)

        # Should be able to use with JAX transformations
        leaves = jax.tree_util.tree_leaves(loss)
        assert len(leaves) == 3  # total, harm, speaker
