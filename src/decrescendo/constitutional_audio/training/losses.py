"""Loss functions for Input Classifier training."""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import optax


@dataclass
class LossWeights:
    """Weights for combining multiple loss components.

    Higher weights increase the importance of that loss component
    during training.
    """

    intent: float = 1.0
    artist: float = 1.0
    voice: float = 1.0
    policy: float = 2.0  # Weight policy violations higher by default


class LossOutput(NamedTuple):
    """Output from loss computation (JAX PyTree-compatible).

    Contains individual losses for each head and the weighted total.
    """

    total: jnp.ndarray
    intent: jnp.ndarray
    artist: jnp.ndarray
    voice: jnp.ndarray
    policy: jnp.ndarray


def compute_classification_loss(
    logits: dict[str, jnp.ndarray],
    labels: dict[str, jnp.ndarray],
    weights: LossWeights | None = None,
) -> LossOutput:
    """Compute combined loss for all classification heads.

    Uses:
    - Softmax cross-entropy for multi-class heads (intent, artist, voice)
    - Sigmoid binary cross-entropy for multi-label head (policy)

    Args:
        logits: Dictionary with logits from each head:
            - intent_logits: (batch, 3)
            - artist_logits: (batch, 3)
            - voice_logits: (batch, 3)
            - policy_logits: (batch, 7)
        labels: Dictionary with labels for each head:
            - intent_labels: (batch,) integer labels
            - artist_labels: (batch,) integer labels
            - voice_labels: (batch,) integer labels
            - policy_labels: (batch, 7) binary labels

    Returns:
        LossOutput with individual and total losses
    """
    if weights is None:
        weights = LossWeights()

    # Multi-class losses (cross-entropy with integer labels)
    intent_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits["intent_logits"], labels["intent_labels"]
    ).mean()

    artist_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits["artist_logits"], labels["artist_labels"]
    ).mean()

    voice_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits["voice_logits"], labels["voice_labels"]
    ).mean()

    # Multi-label loss (binary cross-entropy per label)
    policy_loss = optax.sigmoid_binary_cross_entropy(
        logits["policy_logits"], labels["policy_labels"]
    ).mean()

    # Weighted total
    total_loss = (
        weights.intent * intent_loss
        + weights.artist * artist_loss
        + weights.voice * voice_loss
        + weights.policy * policy_loss
    )

    return LossOutput(
        total=total_loss,
        intent=intent_loss,
        artist=artist_loss,
        voice=voice_loss,
        policy=policy_loss,
    )
