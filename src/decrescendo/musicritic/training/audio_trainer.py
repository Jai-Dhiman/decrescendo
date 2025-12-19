"""Training loop for Output Classifier (audio) with BatchNorm support."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import train_state

from .audio_losses import AudioLossOutput, AudioLossWeights, combined_audio_loss
from .audio_metrics import AudioMetricsOutput, compute_harm_metrics


class AudioTrainState(train_state.TrainState):
    """Extended TrainState for audio models with BatchNorm.

    Unlike the text-based TrainState, this includes batch_stats for
    BatchNorm layers which need to track running statistics.

    Attributes:
        batch_stats: Running mean/variance for BatchNorm layers
        dropout_rng: JAX random key for dropout during training
    """

    batch_stats: dict[str, Any]
    dropout_rng: jax.Array | None = None

    @classmethod
    def create(
        cls,
        *,
        apply_fn: Callable[..., Any],
        params: dict[str, Any],
        tx: optax.GradientTransformation,
        batch_stats: dict[str, Any],
        dropout_rng: jax.Array | None = None,
        **kwargs: Any,
    ) -> "AudioTrainState":
        """Create a new AudioTrainState.

        Args:
            apply_fn: Model apply function
            params: Model parameters
            tx: Optax optimizer/gradient transformation
            batch_stats: Initial batch statistics from model.init()
            dropout_rng: Optional random key for dropout
            **kwargs: Additional arguments passed to parent

        Returns:
            Initialized AudioTrainState
        """
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            batch_stats=batch_stats,
            dropout_rng=dropout_rng,
            **kwargs,
        )

    def next_dropout_rng(self) -> tuple["AudioTrainState", jax.Array]:
        """Split the dropout RNG and return new state with updated RNG.

        Returns:
            Tuple of (new_state, rng_for_this_step)

        Raises:
            ValueError: If dropout_rng was not set during creation
        """
        if self.dropout_rng is None:
            raise ValueError("dropout_rng not set in AudioTrainState")

        current_rng, new_rng = jax.random.split(self.dropout_rng)
        new_state = self.replace(dropout_rng=new_rng)
        return new_state, current_rng


@dataclass
class AudioTrainingConfig:
    """Training hyperparameters for audio classifier."""

    # Optimizer settings
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Training settings
    num_epochs: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

    # Loss settings
    loss_weights: AudioLossWeights = field(default_factory=AudioLossWeights)
    harm_label_smoothing: float = 0.0
    speaker_temperature: float = 0.07

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))


def create_audio_optimizer(
    config: AudioTrainingConfig,
    num_train_steps: int,
) -> optax.GradientTransformation:
    """Create optimizer with learning rate schedule for audio training.

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


def create_audio_train_step(
    model: Any,
    config: AudioTrainingConfig,
) -> Callable[
    [AudioTrainState, dict[str, jnp.ndarray]],
    tuple[AudioTrainState, dict[str, jnp.ndarray]],
]:
    """Create JIT-compiled training step for audio model.

    This training step handles BatchNorm statistics properly by:
    1. Passing batch_stats to model.apply()
    2. Using mutable=['batch_stats'] to get updated statistics
    3. Updating state.batch_stats after each step

    Args:
        model: Flax audio model (OutputClassifierModel)
        config: Training configuration

    Returns:
        JIT-compiled train_step function
    """

    @jax.jit
    def train_step(
        state: AudioTrainState,
        batch: dict[str, jnp.ndarray],
    ) -> tuple[AudioTrainState, dict[str, jnp.ndarray]]:
        """Single training step.

        Args:
            state: Current training state with params and batch_stats
            batch: Batch containing 'audio', 'harm_labels', optionally 'speaker_ids'

        Returns:
            Tuple of (updated_state, metrics_dict)
        """
        # Split dropout RNG
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def loss_fn(
            params: dict[str, Any],
        ) -> tuple[jnp.ndarray, tuple[dict[str, Any], AudioLossOutput, dict[str, Any]]]:
            # Run forward pass with mutable batch_stats
            outputs, updates = model.apply(
                {"params": params, "batch_stats": state.batch_stats},
                batch["audio"],
                train=True,
                rngs={"dropout": dropout_rng},
                mutable=["batch_stats"],
            )

            # Compute losses
            losses = combined_audio_loss(
                outputs,
                batch,
                weights=config.loss_weights,
                harm_label_smoothing=config.harm_label_smoothing,
                speaker_temperature=config.speaker_temperature,
            )

            return losses.total, (outputs, losses, updates)

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, (outputs, losses, updates)), grads = grad_fn(state.params)

        # Apply gradients
        state = state.apply_gradients(grads=grads)

        # Update batch_stats and dropout_rng
        state = state.replace(
            batch_stats=updates["batch_stats"],
            dropout_rng=new_dropout_rng,
        )

        # Compute metrics
        metrics_output = compute_harm_metrics(
            outputs["harm_logits"],
            batch["harm_labels"],
        )

        # Build metrics dictionary
        all_metrics = {
            "loss/total": losses.total,
            "loss/harm": losses.harm,
            "loss/speaker": losses.speaker,
            "metrics/harm_accuracy": metrics_output.harm_accuracy,
            "metrics/harm_f1_macro": metrics_output.harm_f1_macro,
        }

        return state, all_metrics

    return train_step


def create_audio_eval_step(
    model: Any,
    config: AudioTrainingConfig,
) -> Callable[
    [dict[str, Any], dict[str, Any], dict[str, jnp.ndarray]],
    dict[str, jnp.ndarray],
]:
    """Create JIT-compiled evaluation step for audio model.

    During evaluation, BatchNorm uses running statistics (train=False)
    instead of batch statistics.

    Args:
        model: Flax audio model
        config: Training configuration

    Returns:
        JIT-compiled eval_step function
    """

    @jax.jit
    def eval_step(
        params: dict[str, Any],
        batch_stats: dict[str, Any],
        batch: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        """Single evaluation step.

        Args:
            params: Model parameters
            batch_stats: BatchNorm running statistics
            batch: Batch containing 'audio', 'harm_labels', optionally 'speaker_ids'

        Returns:
            Dictionary with loss and metrics
        """
        # Run forward pass with train=False (use running averages)
        outputs = model.apply(
            {"params": params, "batch_stats": batch_stats},
            batch["audio"],
            train=False,
        )

        # Compute losses
        losses = combined_audio_loss(
            outputs,
            batch,
            weights=config.loss_weights,
            harm_label_smoothing=0.0,  # No smoothing during eval
            speaker_temperature=config.speaker_temperature,
        )

        # Compute metrics
        metrics_output = compute_harm_metrics(
            outputs["harm_logits"],
            batch["harm_labels"],
        )

        all_metrics = {
            "loss/total": losses.total,
            "loss/harm": losses.harm,
            "loss/speaker": losses.speaker,
            "metrics/harm_accuracy": metrics_output.harm_accuracy,
            "metrics/harm_f1_macro": metrics_output.harm_f1_macro,
        }

        return all_metrics

    return eval_step


class AudioTrainer:
    """Training orchestrator for Output Classifier (audio).

    Handles the training loop with proper BatchNorm statistics management,
    checkpointing, logging, and evaluation.

    Example:
        >>> model = OutputClassifierModel(config)
        >>> trainer = AudioTrainer(model, training_config, train_loader, eval_loader)
        >>> final_state = trainer.train(initial_state)
    """

    def __init__(
        self,
        model: Any,
        config: AudioTrainingConfig,
        train_dataloader: Any,
        eval_dataloader: Any | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Flax OutputClassifierModel to train
            config: Training configuration
            train_dataloader: AudioDataLoader for training data
            eval_dataloader: Optional AudioDataLoader for evaluation data
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Create compiled step functions
        self.train_step = create_audio_train_step(model, config)
        self.eval_step = create_audio_eval_step(model, config)

        # Setup checkpointing (lazy)
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
        state: AudioTrainState,
        num_epochs: int | None = None,
    ) -> AudioTrainState:
        """Run training loop.

        Args:
            state: Initial training state with params and batch_stats
            num_epochs: Override config num_epochs (optional)

        Returns:
            Final training state
        """
        num_epochs = num_epochs or self.config.num_epochs
        global_step = int(state.step)

        print(f"Starting audio training for {num_epochs} epochs...")
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
                    eval_metrics = self.evaluate(state)
                    self._log_metrics(eval_metrics, global_step, prefix="eval")

                # Checkpointing
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(state, global_step)

            epoch_time = time.time() - epoch_start
            epoch_avg = self._average_metrics(epoch_metrics)
            print(f"\nEpoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s")
            print(f"  Train loss: {epoch_avg.get('loss/total', 0):.4f}")
            print(f"  Harm F1: {epoch_avg.get('metrics/harm_f1_macro', 0):.4f}")

        return state

    def evaluate(self, state: AudioTrainState) -> dict[str, float]:
        """Run evaluation on eval dataset.

        Args:
            state: Training state with params and batch_stats

        Returns:
            Dictionary with averaged evaluation metrics

        Raises:
            ValueError: If no evaluation dataloader provided
        """
        if self.eval_dataloader is None:
            raise ValueError("No evaluation dataloader provided")

        all_metrics: list[dict[str, jnp.ndarray]] = []

        for batch in self.eval_dataloader:
            metrics = self.eval_step(state.params, state.batch_stats, batch)
            all_metrics.append(metrics)

        return self._average_metrics(all_metrics)

    def _average_metrics(
        self,
        metrics_list: list[dict[str, jnp.ndarray]],
    ) -> dict[str, float]:
        """Average metrics across batches."""
        if not metrics_list:
            return {}

        avg: dict[str, float] = {}
        for key in metrics_list[0].keys():
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

        Override this method to add W&B logging, TensorBoard, etc.
        """
        prefix_str = f"[{prefix}] " if prefix else ""
        metrics_str = ", ".join(
            f"{k.split('/')[-1]}: {v:.4f}" for k, v in metrics.items()
        )
        print(f"Step {step} {prefix_str}{metrics_str}")

    def _save_checkpoint(self, state: AudioTrainState, step: int) -> None:
        """Save checkpoint using Orbax.

        Saves params, batch_stats, opt_state, and step.
        """
        self.checkpoint_manager.save(
            step,
            args=ocp.args.StandardSave(state),
        )
        print(f"Saved checkpoint at step {step}")

    def load_checkpoint(
        self,
        state: AudioTrainState,
        step: int | None = None,
    ) -> AudioTrainState:
        """Load checkpoint.

        Args:
            state: Template state for restoration
            step: Specific step to load, or None for latest

        Returns:
            Loaded AudioTrainState

        Raises:
            ValueError: If no checkpoint found
        """
        if step is None:
            step = self.checkpoint_manager.latest_step()

        if step is None:
            raise ValueError("No checkpoint found")

        return self.checkpoint_manager.restore(
            step,
            args=ocp.args.StandardRestore(state),
        )


def initialize_audio_training(
    model: Any,
    config: AudioTrainingConfig,
    num_train_steps: int,
    rng: jax.Array,
    dummy_batch: dict[str, jnp.ndarray],
) -> AudioTrainState:
    """Initialize training state for audio model.

    Creates model variables including batch_stats, sets up optimizer,
    and returns ready-to-train AudioTrainState.

    Args:
        model: OutputClassifierModel instance
        config: Training configuration
        num_train_steps: Total training steps (for LR schedule)
        rng: JAX random key
        dummy_batch: Example batch for shape inference

    Returns:
        Initialized AudioTrainState
    """
    # Split RNG
    init_rng, dropout_rng = jax.random.split(rng)

    # Initialize model with dummy input
    variables = model.init(init_rng, dummy_batch["audio"], train=False)

    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    # Create optimizer
    optimizer = create_audio_optimizer(config, num_train_steps)

    # Create training state
    state = AudioTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
        dropout_rng=dropout_rng,
    )

    return state
