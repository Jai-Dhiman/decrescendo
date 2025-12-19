"""Evaluation metrics for Output Classifier (audio)."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class AudioMetricsOutput(NamedTuple):
    """Output from audio metrics computation (JAX PyTree-compatible).

    Contains metrics for harm classification and optionally speaker verification.
    """

    harm_accuracy: jnp.ndarray  # Per-label accuracy averaged
    harm_f1_macro: jnp.ndarray  # Macro F1 across harm categories
    harm_f1_per_category: jnp.ndarray  # (7,) F1 per category


def compute_harm_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    threshold: float = 0.5,
) -> AudioMetricsOutput:
    """Compute metrics for multi-label harm classification.

    Args:
        logits: Unnormalized logits of shape (batch, 7)
        labels: Binary labels of shape (batch, 7)
        threshold: Threshold for converting probabilities to predictions

    Returns:
        AudioMetricsOutput with accuracy and F1 metrics
    """
    probs = jax.nn.sigmoid(logits)
    predictions = (probs > threshold).astype(jnp.float32)

    # Per-label accuracy (averaged across all labels and samples)
    correct = predictions == labels
    accuracy = jnp.mean(correct)

    # Per-category metrics
    eps = 1e-7

    # True positives, false positives, false negatives per category
    tp = jnp.sum(predictions * labels, axis=0)
    fp = jnp.sum(predictions * (1 - labels), axis=0)
    fn = jnp.sum((1 - predictions) * labels, axis=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_per_category = 2 * precision * recall / (precision + recall + eps)

    # Handle categories with no positive samples in this batch
    has_positives = jnp.sum(labels, axis=0) > 0
    f1_per_category = jnp.where(has_positives, f1_per_category, 0.0)
    num_categories_with_positives = jnp.sum(has_positives)

    # Macro F1 (average F1 across categories that have positive samples)
    macro_f1 = jnp.where(
        num_categories_with_positives > 0,
        jnp.sum(f1_per_category) / num_categories_with_positives,
        0.0,
    )

    return AudioMetricsOutput(
        harm_accuracy=accuracy,
        harm_f1_macro=macro_f1,
        harm_f1_per_category=f1_per_category,
    )


def compute_audio_metrics(
    outputs: dict[str, jnp.ndarray],
    batch: dict[str, jnp.ndarray],
    threshold: float = 0.5,
) -> AudioMetricsOutput:
    """Compute all audio metrics for a batch.

    Convenience function that extracts the relevant tensors from model
    outputs and batch dictionaries.

    Args:
        outputs: Model outputs with 'harm_logits' key
        batch: Batch with 'harm_labels' key
        threshold: Classification threshold

    Returns:
        AudioMetricsOutput with all metrics
    """
    return compute_harm_metrics(
        outputs["harm_logits"],
        batch["harm_labels"],
        threshold=threshold,
    )


def aggregate_audio_metrics(
    metrics_list: list[AudioMetricsOutput],
) -> dict[str, float]:
    """Aggregate audio metrics across batches.

    Args:
        metrics_list: List of AudioMetricsOutput from each batch

    Returns:
        Dictionary with averaged metrics
    """
    if not metrics_list:
        return {}

    n = len(metrics_list)

    # Aggregate scalar metrics
    aggregated = {
        "metrics/harm_accuracy": sum(float(m.harm_accuracy) for m in metrics_list) / n,
        "metrics/harm_f1_macro": sum(float(m.harm_f1_macro) for m in metrics_list) / n,
    }

    # Aggregate per-category F1
    f1_per_cat = np.mean(
        [np.array(m.harm_f1_per_category) for m in metrics_list], axis=0
    )
    for i, f1 in enumerate(f1_per_cat):
        aggregated[f"metrics/harm_f1_category_{i}"] = float(f1)

    return aggregated


def compute_speaker_eer(
    embeddings: np.ndarray,
    speaker_ids: np.ndarray,
    num_thresholds: int = 1000,
) -> float:
    """Compute Equal Error Rate (EER) for speaker verification.

    EER is the point where False Accept Rate (FAR) equals False Reject Rate (FRR).
    This metric is standard for evaluating speaker verification systems.

    Note: This function is computed offline (not JIT-compiled) as it requires
    threshold sweeping and is typically run on validation/test sets.

    Args:
        embeddings: Speaker embeddings of shape (N, dim), should be L2-normalized
        speaker_ids: Speaker IDs of shape (N,)
        num_thresholds: Number of thresholds to sweep for EER computation

    Returns:
        EER value in range [0, 1], where lower is better

    Raises:
        ValueError: If embeddings and speaker_ids have incompatible shapes
    """
    if embeddings.shape[0] != speaker_ids.shape[0]:
        raise ValueError(
            f"embeddings ({embeddings.shape[0]}) and speaker_ids ({speaker_ids.shape[0]}) "
            "must have same number of samples"
        )

    n = len(speaker_ids)
    if n < 2:
        raise ValueError("Need at least 2 samples to compute EER")

    # Compute all pairwise cosine similarities
    # Assuming embeddings are L2-normalized, dot product = cosine similarity
    similarities = embeddings @ embeddings.T

    # Create ground truth: same speaker = 1 (positive), different = 0 (negative)
    same_speaker = speaker_ids[:, None] == speaker_ids[None, :]

    # Get upper triangle indices (excluding diagonal to avoid self-comparison)
    triu_indices = np.triu_indices(n, k=1)
    sim_pairs = similarities[triu_indices]
    gt_pairs = same_speaker[triu_indices].astype(np.float32)

    # Count positives and negatives
    num_positives = np.sum(gt_pairs)
    num_negatives = len(gt_pairs) - num_positives

    if num_positives == 0 or num_negatives == 0:
        raise ValueError(
            "Need both positive pairs (same speaker) and negative pairs "
            "(different speakers) to compute EER"
        )

    # Sweep thresholds to find EER
    thresholds = np.linspace(0, 1, num_thresholds)

    far_values = []  # False Accept Rate
    frr_values = []  # False Reject Rate

    for thresh in thresholds:
        predictions = (sim_pairs >= thresh).astype(np.float32)

        # FAR: Fraction of negative pairs incorrectly accepted (predicted as same)
        false_accepts = np.sum((predictions == 1) & (gt_pairs == 0))
        far = false_accepts / num_negatives

        # FRR: Fraction of positive pairs incorrectly rejected (predicted as different)
        false_rejects = np.sum((predictions == 0) & (gt_pairs == 1))
        frr = false_rejects / num_positives

        far_values.append(far)
        frr_values.append(frr)

    far_values = np.array(far_values)
    frr_values = np.array(frr_values)

    # Find threshold where FAR == FRR (or closest point)
    # EER is the value at this crossing point
    diff = np.abs(far_values - frr_values)
    eer_idx = np.argmin(diff)

    # EER is the average of FAR and FRR at the crossing point
    eer = (far_values[eer_idx] + frr_values[eer_idx]) / 2

    return float(eer)


def compute_speaker_metrics_offline(
    embeddings: np.ndarray,
    speaker_ids: np.ndarray,
) -> dict[str, float]:
    """Compute offline speaker verification metrics.

    This is meant to be called after collecting embeddings from multiple
    batches during evaluation.

    Args:
        embeddings: Collected speaker embeddings of shape (N, dim)
        speaker_ids: Corresponding speaker IDs of shape (N,)

    Returns:
        Dictionary with speaker metrics including EER
    """
    # Normalize embeddings if not already normalized
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-8)

    try:
        eer = compute_speaker_eer(embeddings_normalized, speaker_ids)
        return {
            "metrics/speaker_eer": eer,
            "metrics/speaker_num_samples": len(speaker_ids),
            "metrics/speaker_num_unique": len(np.unique(speaker_ids)),
        }
    except ValueError as e:
        # Return empty metrics if EER cannot be computed
        return {
            "metrics/speaker_eer_error": str(e),
            "metrics/speaker_num_samples": len(speaker_ids),
        }
