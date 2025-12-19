"""Training infrastructure for Constitutional Audio models."""

from .losses import LossOutput, LossWeights, compute_classification_loss
from .metrics import MetricsOutput, aggregate_metrics, compute_metrics
from .train_state import TrainState
from .trainer import Trainer, TrainingConfig, create_optimizer

# Audio-specific training
from .audio_losses import (
    AudioLossOutput,
    AudioLossWeights,
    combined_audio_loss,
    harm_classification_loss,
    speaker_contrastive_loss,
    speaker_triplet_loss,
)
from .audio_metrics import (
    AudioMetricsOutput,
    aggregate_audio_metrics,
    compute_audio_metrics,
    compute_harm_metrics,
    compute_speaker_eer,
    compute_speaker_metrics_offline,
)
from .audio_trainer import (
    AudioTrainer,
    AudioTrainingConfig,
    AudioTrainState,
    create_audio_eval_step,
    create_audio_optimizer,
    create_audio_train_step,
    initialize_audio_training,
)

__all__ = [
    # Train state (text)
    "TrainState",
    # Losses (text)
    "LossWeights",
    "LossOutput",
    "compute_classification_loss",
    # Metrics (text)
    "MetricsOutput",
    "compute_metrics",
    "aggregate_metrics",
    # Trainer (text)
    "Trainer",
    "TrainingConfig",
    "create_optimizer",
    # Audio losses
    "AudioLossWeights",
    "AudioLossOutput",
    "harm_classification_loss",
    "speaker_contrastive_loss",
    "speaker_triplet_loss",
    "combined_audio_loss",
    # Audio metrics
    "AudioMetricsOutput",
    "compute_harm_metrics",
    "compute_audio_metrics",
    "aggregate_audio_metrics",
    "compute_speaker_eer",
    "compute_speaker_metrics_offline",
    # Audio trainer
    "AudioTrainState",
    "AudioTrainingConfig",
    "AudioTrainer",
    "create_audio_optimizer",
    "create_audio_train_step",
    "create_audio_eval_step",
    "initialize_audio_training",
]
