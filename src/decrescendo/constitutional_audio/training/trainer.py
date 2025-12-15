"""Training loop for Input Classifier."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from .losses import LossOutput, LossWeights, compute_classification_loss
from .metrics import MetricsOutput, aggregate_metrics, compute_metrics
from .train_state import TrainState


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimizer settings
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Training settings
    num_epochs: int = 3
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

    # Loss weights
    loss_weights: LossWeights = field(default_factory=LossWeights)

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))


def create_optimizer(
    config: TrainingConfig,
    num_train_steps: int,
) -> optax.GradientTransformation:
    """Create optimizer with learning rate schedule.

    Uses warmup + cosine decay schedule with AdamW optimizer.

    Args:
        config: Training configuration
        num_train_steps: Total number of training steps

    Returns:
        Optax gradient transformation
    """
    # Ensure warmup doesn't exceed total steps
    warmup_steps = min(config.warmup_steps, num_train_steps // 2)

    # Warmup + cosine decay schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_train_steps,
        end_value=config.learning_rate * 0.1,
    )

    # AdamW with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )

    return optimizer


def create_train_step(
    model: Any,
    loss_weights: LossWeights,
) -> Callable[[TrainState, dict[str, jnp.ndarray]], tuple[TrainState, dict[str, jnp.ndarray]]]:
    """Create JIT-compiled training step function.

    Args:
        model: Flax model
        loss_weights: Weights for loss components

    Returns:
        JIT-compiled train_step function
    """

    @jax.jit
    def train_step(
        state: TrainState,
        batch: dict[str, jnp.ndarray],
    ) -> tuple[TrainState, dict[str, jnp.ndarray]]:
        """Single training step.

        Args:
            state: Current training state
            batch: Batch of training data

        Returns:
            Tuple of (updated_state, metrics_dict)
        """
        # Split dropout RNG
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def loss_fn(params: dict[str, Any]) -> tuple[jnp.ndarray, dict[str, Any]]:
            logits = model.apply(
                {"params": params},
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                deterministic=False,
                rngs={"dropout": dropout_rng},
            )

            labels = {
                "intent_labels": batch["intent_labels"],
                "artist_labels": batch["artist_labels"],
                "voice_labels": batch["voice_labels"],
                "policy_labels": batch["policy_labels"],
            }

            losses = compute_classification_loss(logits, labels, loss_weights)
            return losses.total, (logits, losses)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, (logits, losses)), grads = grad_fn(state.params)

        # Apply gradients and update dropout RNG
        state = state.apply_gradients(grads=grads)
        state = state.replace(dropout_rng=new_dropout_rng)

        # Compute metrics
        metrics_output = compute_metrics(logits, batch)

        # Combine all metrics (avoid method calls in jitted code)
        all_metrics = {
            "loss/total": losses.total,
            "loss/intent": losses.intent,
            "loss/artist": losses.artist,
            "loss/voice": losses.voice,
            "loss/policy": losses.policy,
            "metrics/intent_accuracy": metrics_output.intent_accuracy,
            "metrics/artist_accuracy": metrics_output.artist_accuracy,
            "metrics/voice_accuracy": metrics_output.voice_accuracy,
            "metrics/policy_f1": metrics_output.policy_f1,
            "metrics/policy_accuracy": metrics_output.policy_accuracy,
        }

        return state, all_metrics

    return train_step


def create_eval_step(
    model: Any,
    loss_weights: LossWeights,
) -> Callable[[dict[str, Any], dict[str, jnp.ndarray]], dict[str, jnp.ndarray]]:
    """Create JIT-compiled evaluation step function.

    Args:
        model: Flax model
        loss_weights: Weights for loss components

    Returns:
        JIT-compiled eval_step function
    """

    @jax.jit
    def eval_step(
        params: dict[str, Any],
        batch: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        """Single evaluation step.

        Args:
            params: Model parameters
            batch: Batch of evaluation data

        Returns:
            Dictionary with loss and metrics
        """
        logits = model.apply(
            {"params": params},
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            deterministic=True,
        )

        labels = {
            "intent_labels": batch["intent_labels"],
            "artist_labels": batch["artist_labels"],
            "voice_labels": batch["voice_labels"],
            "policy_labels": batch["policy_labels"],
        }

        losses = compute_classification_loss(logits, labels, loss_weights)
        metrics_output = compute_metrics(logits, batch)

        # Combine all metrics (avoid method calls in jitted code)
        all_metrics = {
            "loss/total": losses.total,
            "loss/intent": losses.intent,
            "loss/artist": losses.artist,
            "loss/voice": losses.voice,
            "loss/policy": losses.policy,
            "metrics/intent_accuracy": metrics_output.intent_accuracy,
            "metrics/artist_accuracy": metrics_output.artist_accuracy,
            "metrics/voice_accuracy": metrics_output.voice_accuracy,
            "metrics/policy_f1": metrics_output.policy_f1,
            "metrics/policy_accuracy": metrics_output.policy_accuracy,
        }

        return all_metrics

    return eval_step


class Trainer:
    """Training orchestrator for Input Classifier."""

    def __init__(
        self,
        model: Any,
        config: TrainingConfig,
        train_dataloader: Any,
        eval_dataloader: Any | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Flax model to train
            config: Training configuration
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation data
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Create compiled step functions
        self.train_step = create_train_step(model, config.loss_weights)
        self.eval_step = create_eval_step(model, config.loss_weights)

        # Setup checkpointing
        self._checkpoint_manager: ocp.CheckpointManager | None = None

    def _setup_checkpointing(self) -> ocp.CheckpointManager:
        """Setup Orbax checkpoint manager."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=3,
            save_interval_steps=self.config.save_steps,
        )

        return ocp.CheckpointManager(
            self.config.checkpoint_dir,
            options=options,
        )

    @property
    def checkpoint_manager(self) -> ocp.CheckpointManager:
        """Lazy initialization of checkpoint manager."""
        if self._checkpoint_manager is None:
            self._checkpoint_manager = self._setup_checkpointing()
        return self._checkpoint_manager

    def train(
        self,
        state: TrainState,
        num_epochs: int | None = None,
    ) -> TrainState:
        """Run training loop.

        Args:
            state: Initial training state
            num_epochs: Override config num_epochs (optional)

        Returns:
            Final training state
        """
        num_epochs = num_epochs or self.config.num_epochs
        global_step = int(state.step)

        print(f"Starting training for {num_epochs} epochs...")
        print(f"  - Train batches per epoch: {len(self.train_dataloader)}")
        if self.eval_dataloader:
            print(f"  - Eval batches: {len(self.eval_dataloader)}")

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_metrics: list[dict[str, jnp.ndarray]] = []

            for batch in self.train_dataloader:
                state, metrics = self.train_step(state, batch)
                global_step += 1
                epoch_metrics.append(metrics)

                # Logging
                if global_step % self.config.logging_steps == 0:
                    recent_metrics = epoch_metrics[-self.config.logging_steps :]
                    avg_metrics = self._average_metrics(recent_metrics)
                    self._log_metrics(avg_metrics, global_step, prefix="train")

                # Evaluation
                if self.eval_dataloader and global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(state.params)
                    self._log_metrics(eval_metrics, global_step, prefix="eval")

                # Checkpointing
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(state, global_step)

            epoch_time = time.time() - epoch_start
            epoch_avg = self._average_metrics(epoch_metrics)
            print(f"\nEpoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s")
            print(f"  Train loss: {epoch_avg.get('loss/total', 0):.4f}")
            print(f"  Intent acc: {epoch_avg.get('metrics/intent_accuracy', 0):.4f}")

        return state

    def evaluate(self, params: dict[str, Any]) -> dict[str, float]:
        """Run evaluation on eval dataset.

        Args:
            params: Model parameters to evaluate

        Returns:
            Dictionary with averaged evaluation metrics

        Raises:
            ValueError: If no evaluation dataloader provided
        """
        if self.eval_dataloader is None:
            raise ValueError("No evaluation dataloader provided")

        all_metrics: list[dict[str, jnp.ndarray]] = []
        for batch in self.eval_dataloader:
            metrics = self.eval_step(params, batch)
            all_metrics.append(metrics)

        return self._average_metrics(all_metrics)

    def _average_metrics(
        self, metrics_list: list[dict[str, jnp.ndarray]]
    ) -> dict[str, float]:
        """Average metrics across batches."""
        if not metrics_list:
            return {}

        avg: dict[str, float] = {}
        for key in metrics_list[0].keys():
            if key == "step":
                continue
            values = [float(m[key]) for m in metrics_list]
            avg[key] = sum(values) / len(values)

        return avg

    def _log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics to console.

        Override this method to add W&B logging, etc.
        """
        prefix_str = f"[{prefix}] " if prefix else ""
        metrics_str = ", ".join(f"{k.split('/')[-1]}: {v:.4f}" for k, v in metrics.items())
        print(f"Step {step} {prefix_str}{metrics_str}")

    def _save_checkpoint(self, state: TrainState, step: int) -> None:
        """Save checkpoint using Orbax."""
        self.checkpoint_manager.save(
            step,
            args=ocp.args.StandardSave(state),
        )
        print(f"Saved checkpoint at step {step}")

    def load_checkpoint(self, step: int | None = None) -> TrainState:
        """Load checkpoint.

        Args:
            step: Specific step to load, or None for latest

        Returns:
            Loaded TrainState

        Raises:
            ValueError: If no checkpoint found
        """
        if step is None:
            step = self.checkpoint_manager.latest_step()

        if step is None:
            raise ValueError("No checkpoint found")

        return self.checkpoint_manager.restore(step)
