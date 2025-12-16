#!/usr/bin/env python
"""Training script for Output Classifier (audio).

Usage:
    uv run python scripts/train_audio.py \
        --train-data data/audio_train.jsonl \
        --eval-data data/audio_eval.jsonl \
        --output-dir outputs/output_classifier_v1

Data format (JSONL):
    {"audio_path": "path/to/audio.wav", "harm_labels": [0,0,0,0,0,0,0], "speaker_id": "spk_001"}

Where harm_labels are 7 binary values for:
    [copyright_ip, voice_cloning, cultural, misinformation, emotional_manipulation, content_safety, physical_safety]
"""

import argparse
from pathlib import Path

import jax


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Output Classifier for Constitutional Audio"
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Path to training data (JSONL)",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        help="Path to evaluation data (JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/output_classifier"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )

    args = parser.parse_args()

    print(f"Output directory: {args.output_dir}")
    print(f"Training data: {args.train_data}")

    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    # Import here for faster --help
    from decrescendo.constitutional_audio.output_classifier.config import (
        OutputClassifierConfig,
    )
    from decrescendo.constitutional_audio.output_classifier.inference import (
        initialize_output_classifier,
    )
    from decrescendo.constitutional_audio.data.audio_dataset import (
        AudioClassifierDataset,
        AudioDataLoader,
    )
    from decrescendo.constitutional_audio.training.audio_trainer import (
        AudioTrainer,
        AudioTrainingConfig,
    )
    from decrescendo.constitutional_audio.training.audio_losses import (
        AudioLossWeights,
    )

    # Model config
    config = OutputClassifierConfig()

    # Initialize model
    print("Initializing model...")
    model, variables = initialize_output_classifier(config, init_rng)
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))
    print(f"Model initialized with {num_params:,} parameters")

    # Load datasets
    print("Loading datasets...")
    if not args.train_data.exists():
        print(f"ERROR: Training data not found at {args.train_data}")
        print("\nExpected format (JSONL):")
        print('  {"audio_path": "audio/sample.wav", "harm_labels": [0,0,0,0,0,0,0], "speaker_id": "spk_001"}')
        return

    train_dataset = AudioClassifierDataset.from_jsonl(args.train_data)
    print(f"Training samples: {len(train_dataset)}")

    eval_dataset = None
    if args.eval_data and args.eval_data.exists():
        eval_dataset = AudioClassifierDataset.from_jsonl(args.eval_data)
        print(f"Evaluation samples: {len(eval_dataset)}")

    # Create data loaders
    train_dataloader = AudioDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        preprocessing_config=config.preprocessing,
        shuffle=True,
        seed=args.seed,
    )

    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = AudioDataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            preprocessing_config=config.preprocessing,
            shuffle=False,
        )

    # Training config
    num_train_steps = len(train_dataloader) * args.num_epochs
    print(f"Total training steps: {num_train_steps}")

    training_config = AudioTrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        checkpoint_dir=args.output_dir / "checkpoints",
        loss_weights=AudioLossWeights(
            harm=1.0,
            speaker_contrastive=0.5,
            speaker_triplet=0.5,
        ),
    )

    # Create trainer
    trainer = AudioTrainer(
        model=model,
        variables=variables,
        config=training_config,
        model_config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        rng=dropout_rng,
    )

    # Train
    print("\nStarting training...")
    final_state = trainer.train()

    # Save final checkpoint
    trainer.save_checkpoint(final_state, int(final_state.step))
    print(f"\nTraining complete! Checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
