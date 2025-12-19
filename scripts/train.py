#!/usr/bin/env python
"""Training script for Input Classifier."""

from pathlib import Path

import hydra
import jax
from omegaconf import DictConfig

from decrescendo.musicritic.data.dataset import (
    DataLoader,
    InputClassifierDataset,
    create_dummy_dataset,
)
from decrescendo.musicritic.input_classifier.config import (
    ClassificationConfig,
    InputClassifierConfig,
    TransformerConfig,
)
from decrescendo.musicritic.input_classifier.pretrained import (
    initialize_from_pretrained,
)
from decrescendo.musicritic.training.losses import LossWeights
from decrescendo.musicritic.training.train_state import TrainState
from decrescendo.musicritic.training.trainer import (
    Trainer,
    TrainingConfig,
    create_optimizer,
)


def build_config_from_hydra(cfg: DictConfig) -> InputClassifierConfig:
    """Build InputClassifierConfig from Hydra config."""
    transformer_config = TransformerConfig(
        vocab_size=cfg.model.transformer.vocab_size,
        hidden_size=cfg.model.transformer.hidden_size,
        num_hidden_layers=cfg.model.transformer.num_hidden_layers,
        num_attention_heads=cfg.model.transformer.num_attention_heads,
        intermediate_size=cfg.model.transformer.intermediate_size,
        hidden_dropout_prob=cfg.model.transformer.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.model.transformer.attention_probs_dropout_prob,
        max_position_embeddings=cfg.model.transformer.max_position_embeddings,
        layer_norm_eps=cfg.model.transformer.layer_norm_eps,
    )

    classification_config = ClassificationConfig(
        num_intent_classes=cfg.model.classification.num_intent_classes,
        num_artist_classes=cfg.model.classification.num_artist_classes,
        num_voice_classes=cfg.model.classification.num_voice_classes,
        num_policy_labels=cfg.model.classification.num_policy_labels,
        classifier_dropout=cfg.model.classification.classifier_dropout,
    )

    return InputClassifierConfig(
        transformer=transformer_config,
        classification=classification_config,
        pretrained_model_name=cfg.model.pretrained_model_name,
        use_pretrained=cfg.model.use_pretrained,
    )


def build_training_config_from_hydra(cfg: DictConfig) -> TrainingConfig:
    """Build TrainingConfig from Hydra config."""
    loss_weights = LossWeights(
        intent=cfg.training.loss_weights.intent,
        artist=cfg.training.loss_weights.artist,
        voice=cfg.training.loss_weights.voice,
        policy=cfg.training.loss_weights.policy,
    )

    return TrainingConfig(
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        num_epochs=cfg.training.num_epochs,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        logging_steps=cfg.training.logging_steps,
        loss_weights=loss_weights,
        output_dir=Path(cfg.experiment.output_dir),
        checkpoint_dir=Path(cfg.experiment.output_dir) / "checkpoints",
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Output dir: {cfg.experiment.output_dir}")

    # Set random seed
    rng = jax.random.PRNGKey(cfg.experiment.seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    # Build configurations
    model_config = build_config_from_hydra(cfg)
    training_config = build_training_config_from_hydra(cfg)

    # Initialize model with pretrained weights
    print(f"Loading pretrained model: {model_config.pretrained_model_name}")
    model, params, tokenizer = initialize_from_pretrained(
        model_config, init_rng, max_length=cfg.data.max_length
    )
    print(f"Model initialized with {sum(p.size for p in jax.tree_util.tree_leaves(params)):,} parameters")

    # Load datasets
    print("Loading datasets...")
    train_path = Path(cfg.data.train_path)
    eval_path = Path(cfg.data.eval_path) if cfg.data.eval_path else None

    if train_path.exists():
        train_dataset = InputClassifierDataset.from_jsonl(train_path)
    else:
        print(f"Train file not found at {train_path}, using dummy dataset")
        train_dataset = create_dummy_dataset(num_samples=1000, seed=cfg.experiment.seed)

    if eval_path and eval_path.exists():
        eval_dataset = InputClassifierDataset.from_jsonl(eval_path)
    else:
        print("Eval file not found, using subset of training data")
        eval_dataset = create_dummy_dataset(num_samples=100, seed=cfg.experiment.seed + 1)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        tokenizer=tokenizer,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length,
        shuffle=True,
        seed=cfg.experiment.seed,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        tokenizer=tokenizer,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length,
        shuffle=False,
    )

    # Calculate training steps
    num_train_steps = len(train_dataloader) * training_config.num_epochs
    print(f"Total training steps: {num_train_steps}")

    # Create optimizer
    optimizer = create_optimizer(training_config, num_train_steps)

    # Create training state
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dropout_rng=dropout_rng,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    # Train
    final_state = trainer.train(state)

    # Save final model
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer._save_checkpoint(final_state, int(final_state.step))
    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
