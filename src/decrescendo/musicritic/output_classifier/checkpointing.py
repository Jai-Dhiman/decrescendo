"""Checkpointing utilities for the Output Classifier.

Provides save/load functionality for trained Output Classifier models using Orbax,
with JSON metadata for human-readable configuration storage. Also includes
voice database persistence for protected voices.
"""

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from .config import (
    AudioEncoderConfig,
    OutputClassifierConfig,
    PreprocessingConfig,
    SpeakerConfig,
)
from .inference import OutputClassifierInference
from .model import OutputClassifierModel

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


class VoiceDatabaseError(Exception):
    """Base exception for voice database operations."""

    pass


class VoiceDatabaseNotFoundError(VoiceDatabaseError):
    """Raised when voice database does not exist."""

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


@dataclass
class VoiceEntry:
    """Single protected voice entry.

    Attributes:
        voice_id: Unique identifier for the voice
        name: Human-readable name (e.g., artist name)
        metadata: Additional metadata (e.g., genre, consent info)
    """

    voice_id: int
    name: str
    metadata: dict[str, Any]


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


def _reconstruct_output_classifier_config(data: dict[str, Any]) -> OutputClassifierConfig:
    """Reconstruct OutputClassifierConfig from dict.

    Handles nested PreprocessingConfig, AudioEncoderConfig, and SpeakerConfig.

    Args:
        data: Config dictionary from metadata

    Returns:
        OutputClassifierConfig instance
    """
    preprocessing = PreprocessingConfig(**data.get("preprocessing", {}))
    audio_encoder = AudioEncoderConfig(**data.get("audio_encoder", {}))
    speaker = SpeakerConfig(**data.get("speaker", {}))

    return OutputClassifierConfig(
        preprocessing=preprocessing,
        audio_encoder=audio_encoder,
        speaker=speaker,
        num_harm_categories=data.get("num_harm_categories", 7),
        classifier_hidden_dim=data.get("classifier_hidden_dim", 256),
        classifier_dropout=data.get("classifier_dropout", 0.1),
        aggregation_window=data.get("aggregation_window", 10),
        exponential_decay=data.get("exponential_decay", 0.9),
        block_threshold=data.get("block_threshold", 0.8),
        flag_threshold=data.get("flag_threshold", 0.5),
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
    model_config: OutputClassifierConfig,
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
        ("audio_encoder", "input_samples"),
        ("audio_encoder", "num_conv_layers"),
        ("audio_encoder", "embedding_dim"),
        ("speaker", "embedding_dim"),
        ("speaker", "num_conv_layers"),
        ("num_harm_categories",),
        ("classifier_hidden_dim",),
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


class OutputClassifierCheckpointer:
    """Manages saving and loading Output Classifier checkpoints.

    Uses Orbax for efficient array storage and JSON for human-readable metadata.
    Handles both params and batch_stats (for BatchNorm layers).

    Example:
        >>> checkpointer = OutputClassifierCheckpointer("checkpoints/output")
        >>> checkpointer.save(step=1000, variables=variables, config=config)
        >>> variables, config, metadata = checkpointer.load()
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
        variables: dict[str, Any],
        config: OutputClassifierConfig,
    ) -> Path:
        """Save checkpoint with params and batch_stats.

        Args:
            step: Training step number
            variables: Model variables dict with 'params' and optionally 'batch_stats'
            config: Model configuration

        Returns:
            Path to saved checkpoint directory
        """
        # Extract params and batch_stats
        checkpoint_data = {
            "params": variables["params"],
            "batch_stats": variables.get("batch_stats", {}),
        }

        # Save with Orbax
        self._manager.save(
            step,
            args=ocp.args.StandardSave(checkpoint_data),
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
        config: OutputClassifierConfig | None = None,
    ) -> tuple[dict[str, Any], OutputClassifierConfig, CheckpointMetadata]:
        """Load checkpoint.

        Args:
            step: Step to load (None for latest)
            config: Optional config to verify compatibility against

        Returns:
            Tuple of (variables_dict, config, metadata)
            variables_dict contains 'params' and 'batch_stats'

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
        loaded_config = _reconstruct_output_classifier_config(metadata.config)

        # Check config compatibility if provided
        if config is not None:
            check_config_compatibility(metadata.config, config)

        # Load checkpoint data
        try:
            restored = self._manager.restore(step)
        except Exception as e:
            raise CheckpointCorruptedError(f"Failed to restore checkpoint data: {e}") from e

        variables = {
            "params": restored["params"],
            "batch_stats": restored.get("batch_stats", {}),
        }

        return variables, loaded_config, metadata

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
# Voice Database Functions
# -----------------------------------------------------------------------------


def save_voice_database(
    path: Path | str,
    embeddings: np.ndarray | jnp.ndarray,
    entries: list[VoiceEntry],
) -> Path:
    """Save protected voice database.

    Stores voice embeddings in NPZ format and metadata in JSON.

    Args:
        path: Directory to save voice database
        embeddings: Voice embeddings array (num_voices, embedding_dim)
        entries: List of voice entries with metadata

    Returns:
        Path to saved database directory

    Example:
        >>> embeddings = np.random.randn(5, 192).astype(np.float32)
        >>> entries = [VoiceEntry(i, f"Artist{i}", {}) for i in range(5)]
        >>> save_voice_database("voices/", embeddings, entries)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Convert to numpy if JAX array
    embeddings_np = np.asarray(embeddings)

    # Save embeddings as NPZ
    np.savez(path / "voices.npz", embeddings=embeddings_np)

    # Save manifest as JSON
    manifest = {
        "version": LIBRARY_VERSION,
        "num_voices": len(entries),
        "embedding_dim": embeddings_np.shape[1] if len(embeddings_np.shape) > 1 else 0,
        "voices": [
            {
                "voice_id": e.voice_id,
                "name": e.name,
                "metadata": e.metadata,
            }
            for e in entries
        ],
    }
    with open(path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return path


def load_voice_database(
    path: Path | str,
) -> tuple[jnp.ndarray, list[VoiceEntry]]:
    """Load protected voice database.

    Args:
        path: Directory containing voice database

    Returns:
        Tuple of (embeddings as JAX array, list of VoiceEntry)

    Raises:
        VoiceDatabaseNotFoundError: If database files not found
        VoiceDatabaseError: If database is corrupted

    Example:
        >>> embeddings, entries = load_voice_database("voices/")
        >>> print(f"Loaded {len(entries)} protected voices")
    """
    path = Path(path)

    # Check files exist
    embeddings_path = path / "voices.npz"
    manifest_path = path / "manifest.json"

    if not embeddings_path.exists():
        raise VoiceDatabaseNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not manifest_path.exists():
        raise VoiceDatabaseNotFoundError(f"Manifest file not found: {manifest_path}")

    # Load embeddings
    try:
        with np.load(embeddings_path) as data:
            embeddings = jnp.array(data["embeddings"])
    except Exception as e:
        raise VoiceDatabaseError(f"Failed to load embeddings: {e}") from e

    # Load manifest
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        raise VoiceDatabaseError(f"Invalid JSON in manifest: {e}") from e

    entries = [
        VoiceEntry(
            voice_id=v["voice_id"],
            name=v["name"],
            metadata=v.get("metadata", {}),
        )
        for v in manifest.get("voices", [])
    ]

    return embeddings, entries


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------


def save_output_classifier(
    path: Path | str,
    variables: dict[str, Any],
    config: OutputClassifierConfig,
    step: int = 0,
) -> Path:
    """Save Output Classifier checkpoint.

    Convenience function for one-off saves without managing a checkpointer.

    Args:
        path: Checkpoint directory
        variables: Model variables with 'params' and 'batch_stats'
        config: Model configuration
        step: Training step (default: 0)

    Returns:
        Path to saved checkpoint
    """
    checkpointer = OutputClassifierCheckpointer(path, max_to_keep=1)
    return checkpointer.save(step, variables, config)


def load_output_classifier(
    path: Path | str,
    step: int | None = None,
) -> tuple[OutputClassifierModel, dict[str, Any], OutputClassifierConfig]:
    """Load Output Classifier from checkpoint.

    Args:
        path: Checkpoint directory
        step: Specific step to load (None for latest)

    Returns:
        Tuple of (model, variables, config)

    Raises:
        CheckpointNotFoundError: If no checkpoint found
        CheckpointVersionError: If checkpoint version incompatible
        CheckpointCorruptedError: If checkpoint data is corrupted
    """
    checkpointer = OutputClassifierCheckpointer(path)
    variables, config, _ = checkpointer.load(step)

    model = OutputClassifierModel(config=config)
    return model, variables, config


def load_output_classifier_inference(
    path: Path | str,
    step: int | None = None,
    protected_voices: jnp.ndarray | None = None,
    protected_voice_names: list[str] | None = None,
    voice_database_path: Path | str | None = None,
) -> OutputClassifierInference:
    """Load Output Classifier inference pipeline from checkpoint.

    Reconstructs a complete inference pipeline ready for classification.
    Can optionally load protected voices from a voice database.

    Args:
        path: Checkpoint directory
        step: Specific step to load (None for latest)
        protected_voices: Optional protected voice embeddings
        protected_voice_names: Optional protected voice names
        voice_database_path: Optional path to voice database (overrides above if provided)

    Returns:
        Ready-to-use OutputClassifierInference instance

    Raises:
        CheckpointNotFoundError: If no checkpoint found
        CheckpointVersionError: If checkpoint version incompatible
        CheckpointCorruptedError: If checkpoint data is corrupted

    Example:
        >>> # Load with separate voice database
        >>> inference = load_output_classifier_inference(
        ...     "checkpoints/output",
        ...     voice_database_path="voices/",
        ... )
        >>> result = inference.classify_file("audio.wav")
        >>> print(result.decision)
    """
    model, variables, config = load_output_classifier(path, step)

    # Load voice database if path provided
    if voice_database_path is not None:
        embeddings, entries = load_voice_database(voice_database_path)
        protected_voices = embeddings
        protected_voice_names = [e.name for e in entries]

    return OutputClassifierInference(
        model=model,
        variables=variables,
        config=config,
        protected_voices=protected_voices,
        protected_voice_names=protected_voice_names,
    )
