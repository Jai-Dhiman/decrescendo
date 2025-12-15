"""Checkpointing utilities for the Input Classifier.

Provides save/load functionality for trained Input Classifier models using Orbax,
with JSON metadata for human-readable configuration storage.
"""

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import jax
import orbax.checkpoint as ocp

from .config import (
    ClassificationConfig,
    InputClassifierConfig,
    TransformerConfig,
)
from .inference import InferenceConfig, InputClassifierInference
from .model import InputClassifier
from .pretrained import get_tokenizer

# Version constants
CHECKPOINT_VERSION = 1
LIBRARY_VERSION = "1.0.0"

T = TypeVar("T")


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""

    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when checkpoint does not exist."""

    pass


class CheckpointVersionError(CheckpointError):
    """Raised when checkpoint version is incompatible."""

    pass


class CheckpointConfigError(CheckpointError):
    """Raised when checkpoint config is incompatible."""

    pass


class CheckpointCorruptedError(CheckpointError):
    """Raised when checkpoint data is corrupted."""

    pass


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata for checkpoint version tracking."""

    library_version: str
    checkpoint_version: int
    step: int
    created_at: str
    config: dict[str, Any]


# -----------------------------------------------------------------------------
# Serialization Helpers
# -----------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert frozen dataclass to dict for JSON serialization.

    Args:
        obj: Object to convert (dataclass, dict, list, or primitive)

    Returns:
        JSON-serializable representation
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    return obj


def _dict_to_dataclass(cls: type[T], data: dict[str, Any]) -> T:
    """Recursively reconstruct frozen dataclass from dict.

    Args:
        cls: Target dataclass type
        data: Dictionary with field values

    Returns:
        Reconstructed dataclass instance
    """
    if not is_dataclass(cls) or isinstance(cls, type) is False:
        return data  # type: ignore

    field_info = {f.name: f.type for f in fields(cls)}
    kwargs: dict[str, Any] = {}

    for name, value in data.items():
        if name not in field_info:
            continue

        field_type = field_info[name]

        # Handle nested dataclasses
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[name] = value

    return cls(**kwargs)


def _reconstruct_input_classifier_config(data: dict[str, Any]) -> InputClassifierConfig:
    """Reconstruct InputClassifierConfig from dict.

    Handles nested TransformerConfig and ClassificationConfig.

    Args:
        data: Config dictionary from metadata

    Returns:
        InputClassifierConfig instance
    """
    transformer = TransformerConfig(**data.get("transformer", {}))
    classification = ClassificationConfig(**data.get("classification", {}))

    return InputClassifierConfig(
        transformer=transformer,
        classification=classification,
        pretrained_model_name=data.get("pretrained_model_name", "roberta-base"),
        use_pretrained=data.get("use_pretrained", True),
    )


# -----------------------------------------------------------------------------
# Version Checking
# -----------------------------------------------------------------------------


def check_version_compatibility(
    metadata: CheckpointMetadata,
    current_version: int = CHECKPOINT_VERSION,
) -> None:
    """Verify checkpoint version is compatible.

    Args:
        metadata: Loaded checkpoint metadata
        current_version: Current checkpoint format version

    Raises:
        CheckpointVersionError: If versions are incompatible
    """
    if metadata.checkpoint_version > current_version:
        raise CheckpointVersionError(
            f"Checkpoint version {metadata.checkpoint_version} is newer than "
            f"supported version {current_version}. Please upgrade the library."
        )


def _get_nested(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Get nested value from dict using tuple of keys."""
    result = d
    for key in keys:
        if not isinstance(result, dict):
            return None
        result = result.get(key)
    return result


def check_config_compatibility(
    checkpoint_config: dict[str, Any],
    model_config: InputClassifierConfig,
) -> list[str]:
    """Check if checkpoint config is compatible with model config.

    Args:
        checkpoint_config: Config from checkpoint metadata
        model_config: Current model configuration

    Returns:
        List of warnings for non-critical mismatches

    Raises:
        CheckpointConfigError: For critical mismatches that prevent loading
    """
    warnings: list[str] = []
    current_config = _dataclass_to_dict(model_config)

    # Critical fields that must match for weight compatibility
    critical_fields: list[tuple[str, ...]] = [
        ("transformer", "hidden_size"),
        ("transformer", "num_hidden_layers"),
        ("transformer", "num_attention_heads"),
        ("transformer", "intermediate_size"),
        ("transformer", "vocab_size"),
        ("classification", "num_intent_classes"),
        ("classification", "num_artist_classes"),
        ("classification", "num_voice_classes"),
        ("classification", "num_policy_labels"),
    ]

    for field_path in critical_fields:
        ckpt_val = _get_nested(checkpoint_config, field_path)
        curr_val = _get_nested(current_config, field_path)
        if ckpt_val is not None and curr_val is not None and ckpt_val != curr_val:
            raise CheckpointConfigError(
                f"Critical config mismatch at {'.'.join(field_path)}: "
                f"checkpoint has {ckpt_val}, model expects {curr_val}"
            )

    return warnings


# -----------------------------------------------------------------------------
# Checkpointer Class
# -----------------------------------------------------------------------------


class InputClassifierCheckpointer:
    """Manages saving and loading Input Classifier checkpoints.

    Uses Orbax for efficient array storage and JSON for human-readable metadata.

    Example:
        >>> checkpointer = InputClassifierCheckpointer("checkpoints/input")
        >>> checkpointer.save(step=1000, params=params, config=config)
        >>> params, config, metadata = checkpointer.load()
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        max_to_keep: int = 3,
    ) -> None:
        """Initialize checkpointer.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_to_keep: Maximum number of checkpoints to retain
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep)
        self._manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            options=options,
        )

    def save(
        self,
        step: int,
        params: dict[str, Any],
        config: InputClassifierConfig,
    ) -> Path:
        """Save checkpoint.

        Args:
            step: Training step number
            params: Model parameters
            config: Model configuration

        Returns:
            Path to saved checkpoint directory
        """
        # Save params with Orbax
        self._manager.save(
            step,
            args=ocp.args.StandardSave({"params": params}),
        )

        # Wait for save to complete
        self._manager.wait_until_finished()

        # Save metadata as JSON
        metadata = CheckpointMetadata(
            library_version=LIBRARY_VERSION,
            checkpoint_version=CHECKPOINT_VERSION,
            step=step,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            config=_dataclass_to_dict(config),
        )
        metadata_path = self.checkpoint_dir / f"{step}" / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        return self.checkpoint_dir / f"{step}"

    def load(
        self,
        step: int | None = None,
        config: InputClassifierConfig | None = None,
    ) -> tuple[dict[str, Any], InputClassifierConfig, CheckpointMetadata]:
        """Load checkpoint.

        Args:
            step: Step to load (None for latest)
            config: Optional config to verify compatibility against

        Returns:
            Tuple of (params, config, metadata)

        Raises:
            CheckpointNotFoundError: If no checkpoint found
            CheckpointVersionError: If checkpoint version incompatible
            CheckpointConfigError: If config incompatible
            CheckpointCorruptedError: If checkpoint data is corrupted
        """
        if step is None:
            step = self._manager.latest_step()

        if step is None:
            raise CheckpointNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")

        # Load metadata
        metadata_path = self.checkpoint_dir / f"{step}" / "metadata.json"
        if not metadata_path.exists():
            raise CheckpointCorruptedError(f"Metadata file missing at {metadata_path}")

        try:
            with open(metadata_path) as f:
                metadata_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise CheckpointCorruptedError(f"Invalid JSON in metadata: {e}") from e

        metadata = CheckpointMetadata(**metadata_dict)

        # Check version
        check_version_compatibility(metadata)

        # Reconstruct config
        loaded_config = _reconstruct_input_classifier_config(metadata.config)

        # Check config compatibility if provided
        if config is not None:
            check_config_compatibility(metadata.config, config)

        # Load params
        try:
            restored = self._manager.restore(step)
        except Exception as e:
            raise CheckpointCorruptedError(f"Failed to restore checkpoint data: {e}") from e

        params = restored["params"]

        return params, loaded_config, metadata

    def latest_step(self) -> int | None:
        """Get latest checkpoint step.

        Returns:
            Latest step number, or None if no checkpoints exist
        """
        return self._manager.latest_step()

    def all_steps(self) -> list[int]:
        """Get all available checkpoint steps.

        Returns:
            List of available step numbers
        """
        return list(self._manager.all_steps())


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------


def save_input_classifier(
    path: Path | str,
    params: dict[str, Any],
    config: InputClassifierConfig,
    step: int = 0,
) -> Path:
    """Save Input Classifier checkpoint.

    Convenience function for one-off saves without managing a checkpointer.

    Args:
        path: Checkpoint directory
        params: Model parameters
        config: Model configuration
        step: Training step (default: 0)

    Returns:
        Path to saved checkpoint
    """
    checkpointer = InputClassifierCheckpointer(path, max_to_keep=1)
    return checkpointer.save(step, params, config)


def load_input_classifier(
    path: Path | str,
    step: int | None = None,
) -> tuple[InputClassifier, dict[str, Any], InputClassifierConfig]:
    """Load Input Classifier from checkpoint.

    Args:
        path: Checkpoint directory
        step: Specific step to load (None for latest)

    Returns:
        Tuple of (model, params, config)

    Raises:
        CheckpointNotFoundError: If no checkpoint found
        CheckpointVersionError: If checkpoint version incompatible
        CheckpointCorruptedError: If checkpoint data is corrupted
    """
    checkpointer = InputClassifierCheckpointer(path)
    params, config, _ = checkpointer.load(step)

    model = InputClassifier(config=config)
    return model, params, config


def load_input_classifier_inference(
    path: Path | str,
    step: int | None = None,
    inference_config: InferenceConfig | None = None,
    tokenizer_name: str | None = None,
) -> InputClassifierInference:
    """Load Input Classifier inference pipeline from checkpoint.

    Reconstructs a complete inference pipeline ready for classification.

    Args:
        path: Checkpoint directory
        step: Specific step to load (None for latest)
        inference_config: Optional inference configuration
        tokenizer_name: Optional tokenizer name (uses config default if None)

    Returns:
        Ready-to-use InputClassifierInference instance

    Raises:
        CheckpointNotFoundError: If no checkpoint found
        CheckpointVersionError: If checkpoint version incompatible
        CheckpointCorruptedError: If checkpoint data is corrupted

    Example:
        >>> inference = load_input_classifier_inference("checkpoints/input")
        >>> result = inference.classify("Generate music like Drake")
        >>> print(result.decision)
    """
    model, params, config = load_input_classifier(path, step)

    # Load tokenizer
    tokenizer = get_tokenizer(tokenizer_name or config.pretrained_model_name)

    return InputClassifierInference(
        model=model,
        params=params,
        tokenizer=tokenizer,
        config=inference_config,
    )
