"""Loss functions for Output Classifier (audio) training."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


@dataclass
class AudioLossWeights:
    """Weights for combining audio loss components.

    Higher weights increase the importance of that loss component
    during training.

    Attributes:
        harm: Weight for harm classification loss
        speaker: Weight for speaker verification loss
    """

    harm: float = 1.0
    speaker: float = 0.5  # Lower by default as speaker is auxiliary


class AudioLossOutput(NamedTuple):
    """Output from audio loss computation (JAX PyTree-compatible).

    Contains individual losses for each component and the weighted total.
    """

    total: jnp.ndarray
    harm: jnp.ndarray
    speaker: jnp.ndarray


def harm_classification_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    label_smoothing: float = 0.0,
) -> jnp.ndarray:
    """Binary cross-entropy loss for multi-label harm classification.

    Computes sigmoid binary cross-entropy for each of the 7 harm categories
    independently, then averages across all labels and samples.

    Args:
        logits: Unnormalized logits of shape (batch, 7)
        labels: Binary labels of shape (batch, 7)
        label_smoothing: Optional label smoothing factor in [0, 1).
            When > 0, labels are smoothed towards 0.5.

    Returns:
        Scalar loss value (averaged over batch and categories)

    Raises:
        ValueError: If logits and labels have incompatible shapes
    """
    if logits.shape != labels.shape:
        raise ValueError(
            f"logits shape {logits.shape} must match labels shape {labels.shape}"
        )

    # Apply label smoothing if specified
    if label_smoothing > 0:
        if not 0 <= label_smoothing < 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

    # Compute binary cross-entropy per element
    loss = optax.sigmoid_binary_cross_entropy(logits, labels)

    # Average over all elements
    return loss.mean()


def speaker_contrastive_loss(
    embeddings: jnp.ndarray,
    speaker_ids: jnp.ndarray,
    temperature: float = 0.07,
) -> jnp.ndarray:
    """In-batch contrastive loss (InfoNCE) for speaker verification.

    Uses samples from the same speaker as positives and different speakers
    as negatives within the same batch. This avoids explicit negative mining.

    This function is JIT-compatible - all conditionals use jnp.where.

    Args:
        embeddings: L2-normalized speaker embeddings of shape (batch, dim)
        speaker_ids: Integer speaker IDs of shape (batch,)
        temperature: Temperature scaling for softmax. Lower values make
            the distribution sharper (more confident).

    Returns:
        Scalar loss value

    Note:
        - Embeddings should be L2-normalized before passing to this function
        - Requires at least 2 samples per speaker for meaningful gradients
        - Returns 0.0 if no positive pairs exist in the batch
    """
    batch_size = embeddings.shape[0]

    # Compute cosine similarity matrix: (batch, batch)
    # Since embeddings are L2-normalized, dot product = cosine similarity
    similarity = jnp.matmul(embeddings, embeddings.T) / temperature

    # Create positive mask: same speaker, excluding diagonal (self)
    labels_equal = speaker_ids[:, None] == speaker_ids[None, :]
    mask_self = jnp.eye(batch_size, dtype=jnp.bool_)
    positive_mask = labels_equal & ~mask_self

    # Check if any positives exist per sample
    num_positives_per_sample = jnp.sum(positive_mask, axis=1)
    has_positive = num_positives_per_sample > 0

    # Total samples with positives (for averaging)
    total_has_positive = jnp.sum(has_positive.astype(jnp.float32))

    # For numerical stability, subtract max before exp
    similarity_max = jnp.max(similarity, axis=1, keepdims=True)
    exp_sim = jnp.exp(similarity - similarity_max)

    # Mask out self-similarity for denominator
    exp_sim_masked = exp_sim * (~mask_self).astype(jnp.float32)

    # Denominator: sum of exp similarities excluding self
    denominator = jnp.sum(exp_sim_masked, axis=1)

    # Numerator: sum of exp similarities for positives
    # Use safe masking: set numerator to denominator for samples without positives
    # This ensures log(numerator/denominator) = 0 for those samples
    numerator_raw = jnp.sum(exp_sim * positive_mask.astype(jnp.float32), axis=1)
    numerator = jnp.where(has_positive, numerator_raw, denominator)

    # Compute log probability using safe division
    # Clamp denominator to avoid division by zero
    safe_denominator = jnp.maximum(denominator, 1e-8)
    log_prob = jnp.log(numerator + 1e-8) - jnp.log(safe_denominator)

    # Loss is negative log probability (only for samples with positives)
    # Use multiplication mask instead of jnp.where to avoid gradient issues
    loss_per_sample = -log_prob * has_positive.astype(jnp.float32)

    # Average over samples that have positives
    # Use safe division with a minimum denominator
    safe_total = jnp.maximum(total_has_positive, 1.0)
    avg_loss = jnp.sum(loss_per_sample) / safe_total

    # Return 0.0 if no positives exist (multiply by indicator)
    # Also clamp to >= 0 to handle numerical precision issues
    has_any_positive = total_has_positive > 0
    return jnp.maximum(0.0, avg_loss * has_any_positive.astype(jnp.float32))


def speaker_triplet_loss(
    anchor: jnp.ndarray,
    positive: jnp.ndarray,
    negative: jnp.ndarray,
    margin: float = 0.2,
) -> jnp.ndarray:
    """Triplet loss for speaker verification.

    Encourages anchor to be closer to positive (same speaker) than to
    negative (different speaker) by at least the specified margin.

    Args:
        anchor: Anchor embeddings of shape (batch, dim)
        positive: Positive embeddings (same speaker) of shape (batch, dim)
        negative: Negative embeddings (different speaker) of shape (batch, dim)
        margin: Margin between positive and negative distances

    Returns:
        Scalar loss value

    Note:
        Embeddings should be L2-normalized. With normalized embeddings,
        cosine similarity = dot product, so we use similarity instead of
        Euclidean distance.
    """
    # Cosine similarity (for L2-normalized embeddings)
    pos_sim = jnp.sum(anchor * positive, axis=-1)
    neg_sim = jnp.sum(anchor * negative, axis=-1)

    # Triplet loss: max(0, margin - pos_sim + neg_sim)
    # We want: pos_sim > neg_sim + margin
    # So loss is positive when pos_sim < neg_sim + margin
    loss = jnp.maximum(0.0, margin - pos_sim + neg_sim)

    return loss.mean()


def combined_audio_loss(
    outputs: dict[str, jnp.ndarray],
    batch: dict[str, jnp.ndarray],
    weights: AudioLossWeights | None = None,
    harm_label_smoothing: float = 0.0,
    speaker_temperature: float = 0.07,
) -> AudioLossOutput:
    """Combined loss for audio output classifier training.

    Combines harm classification loss with optional speaker verification loss.
    Speaker loss is only computed if speaker_ids are provided in the batch.

    Args:
        outputs: Model outputs dictionary containing:
            - harm_logits: (batch, 7) unnormalized logits for harm categories
            - speaker_embeddings: (batch, speaker_dim) L2-normalized embeddings
        batch: Batch dictionary containing:
            - harm_labels: (batch, 7) binary labels
            - speaker_ids: (batch,) integer speaker IDs (optional)
        weights: Loss component weights. Uses defaults if None.
        harm_label_smoothing: Label smoothing for harm classification
        speaker_temperature: Temperature for contrastive loss

    Returns:
        AudioLossOutput with total, harm, and speaker losses
    """
    if weights is None:
        weights = AudioLossWeights()

    # Harm classification loss (always computed)
    harm_loss = harm_classification_loss(
        outputs["harm_logits"],
        batch["harm_labels"],
        label_smoothing=harm_label_smoothing,
    )

    # Speaker verification loss (only if speaker_ids provided)
    speaker_loss = jnp.array(0.0)
    if "speaker_ids" in batch and "speaker_embeddings" in outputs:
        speaker_loss = speaker_contrastive_loss(
            outputs["speaker_embeddings"],
            batch["speaker_ids"],
            temperature=speaker_temperature,
        )

    # Weighted total
    total_loss = weights.harm * harm_loss + weights.speaker * speaker_loss

    return AudioLossOutput(
        total=total_loss,
        harm=harm_loss,
        speaker=speaker_loss,
    )
