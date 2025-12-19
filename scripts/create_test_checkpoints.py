#!/usr/bin/env python
"""Create test checkpoints with random weights for CLI testing.

This creates checkpoints that can be used to test the CLI interface.
Note: Predictions will be random/meaningless without proper training.

Usage:
    uv run python scripts/create_test_checkpoints.py
"""

from pathlib import Path

import jax


def create_input_classifier_checkpoint(output_dir: Path) -> None:
    """Create input classifier checkpoint with random weights."""
    from decrescendo.musicritic.input_classifier.config import (
        InputClassifierConfig,
    )
    from decrescendo.musicritic.input_classifier.pretrained import (
        initialize_from_pretrained,
    )
    from decrescendo.musicritic.input_classifier.checkpointing import (
        save_input_classifier,
    )

    print("Creating input classifier checkpoint...")

    rng = jax.random.PRNGKey(42)
    config = InputClassifierConfig(use_pretrained=True)

    model, params, tokenizer = initialize_from_pretrained(config, rng)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_input_classifier(output_dir, params, config, step=0)

    print(f"  Saved to: {output_dir}")


def create_output_classifier_checkpoint(output_dir: Path) -> None:
    """Create output classifier checkpoint with random weights."""
    from decrescendo.musicritic.output_classifier.config import (
        OutputClassifierConfig,
    )
    from decrescendo.musicritic.output_classifier.inference import (
        initialize_output_classifier,
    )
    from decrescendo.musicritic.output_classifier.checkpointing import (
        save_output_classifier,
    )

    print("Creating output classifier checkpoint...")

    rng = jax.random.PRNGKey(42)
    config = OutputClassifierConfig()

    model, variables = initialize_output_classifier(config, rng)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_output_classifier(output_dir, variables, config, step=0)

    print(f"  Saved to: {output_dir}")


def create_voice_database(output_dir: Path) -> None:
    """Create an empty voice database."""
    from decrescendo.musicritic.output_classifier.voice_database import (
        VoiceDatabase,
    )

    print("Creating empty voice database...")

    db = VoiceDatabase(embedding_dim=192)
    output_dir.mkdir(parents=True, exist_ok=True)
    db.save(output_dir)

    print(f"  Saved to: {output_dir}")


def main() -> None:
    """Create all test checkpoints."""
    base_dir = Path("checkpoints").resolve()  # Use absolute path

    print("Creating test checkpoints (random weights)...")
    print("=" * 50)

    create_input_classifier_checkpoint(base_dir / "input")
    create_output_classifier_checkpoint(base_dir / "output")
    create_voice_database(base_dir / "voices")

    print("=" * 50)
    print("\nDone! You can now test the CLI:")
    print()
    print("  # Classify a prompt (random predictions)")
    print('  constitutional-audio classify-prompt "Generate a piano melody" \\')
    print("    --input-checkpoint ./checkpoints/input")
    print()
    print("  # List voices (empty)")
    print("  constitutional-audio list-voices --voice-db ./checkpoints/voices")
    print()
    print("NOTE: Predictions are random without proper training!")


if __name__ == "__main__":
    main()
