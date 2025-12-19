"""Evaluation metrics for Input Classifier."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class MetricsOutput(NamedTuple):
    """Output from metrics computation (JAX PyTree-compatible).

    Contains accuracy metrics for each classification head.
    """

    intent_accuracy: jnp.ndarray
    artist_accuracy: jnp.ndarray
    voice_accuracy: jnp.ndarray
    policy_f1: jnp.ndarray  # Macro F1 for multi-label
    policy_accuracy: jnp.ndarray  # Per-label accuracy (averaged)


def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Compute accuracy for multi-class classification.

    Args:
        logits: (batch, num_classes) logits
        labels: (batch,) integer labels

    Returns:
        Scalar accuracy value
    """
    predictions = jnp.argmax(logits, axis=-1)
    correct = predictions == labels
    return jnp.mean(correct)


def compute_multilabel_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    threshold: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute metrics for multi-label classification.

    Args:
        logits: (batch, num_labels) logits
        labels: (batch, num_labels) binary labels
        threshold: Threshold for converting probabilities to predictions

    Returns:
        Tuple of (macro_f1, per_label_accuracy)
    """
    probs = jax.nn.sigmoid(logits)
    predictions = (probs > threshold).astype(jnp.float32)

    # Per-label accuracy (averaged across labels)
    correct = predictions == labels
    per_label_accuracy = jnp.mean(correct)

    # Macro F1 score
    # For each label, compute precision, recall, F1, then average
    eps = 1e-7

    # True positives, false positives, false negatives per label
    tp = jnp.sum(predictions * labels, axis=0)
    fp = jnp.sum(predictions * (1 - labels), axis=0)
    fn = jnp.sum((1 - predictions) * labels, axis=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # Handle labels with no positive samples
    has_positives = jnp.sum(labels, axis=0) > 0
    f1 = jnp.where(has_positives, f1, 0.0)
    num_labels_with_positives = jnp.sum(has_positives)

    # Macro F1 (average F1 across labels that have positive samples)
    macro_f1 = jnp.where(
        num_labels_with_positives > 0,
        jnp.sum(f1) / num_labels_with_positives,
        0.0,
    )

    return macro_f1, per_label_accuracy


def compute_metrics(
    logits: dict[str, jnp.ndarray],
    batch: dict[str, jnp.ndarray],
) -> MetricsOutput:
    """Compute all metrics for a batch.

    Args:
        logits: Dictionary with logits from each head
        batch: Dictionary with labels for each head

    Returns:
        MetricsOutput with all metrics
    """
    intent_accuracy = compute_accuracy(logits["intent_logits"], batch["intent_labels"])
    artist_accuracy = compute_accuracy(logits["artist_logits"], batch["artist_labels"])
    voice_accuracy = compute_accuracy(logits["voice_logits"], batch["voice_labels"])

    policy_f1, policy_accuracy = compute_multilabel_metrics(
        logits["policy_logits"], batch["policy_labels"]
    )

    return MetricsOutput(
        intent_accuracy=intent_accuracy,
        artist_accuracy=artist_accuracy,
        voice_accuracy=voice_accuracy,
        policy_f1=policy_f1,
        policy_accuracy=policy_accuracy,
    )


def aggregate_metrics(metrics_list: list[MetricsOutput]) -> dict[str, float]:
    """Aggregate metrics across batches.

    Args:
        metrics_list: List of MetricsOutput from each batch

    Returns:
        Dictionary with averaged metrics
    """
    if not metrics_list:
        return {}

    # Aggregate each field of MetricsOutput
    aggregated = {
        "metrics/intent_accuracy": sum(float(m.intent_accuracy) for m in metrics_list) / len(metrics_list),
        "metrics/artist_accuracy": sum(float(m.artist_accuracy) for m in metrics_list) / len(metrics_list),
        "metrics/voice_accuracy": sum(float(m.voice_accuracy) for m in metrics_list) / len(metrics_list),
        "metrics/policy_f1": sum(float(m.policy_f1) for m in metrics_list) / len(metrics_list),
        "metrics/policy_accuracy": sum(float(m.policy_accuracy) for m in metrics_list) / len(metrics_list),
    }

    return aggregated
