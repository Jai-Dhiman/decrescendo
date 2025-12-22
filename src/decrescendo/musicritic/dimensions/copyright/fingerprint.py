"""Audio fingerprinting for copyright detection.

This module provides:
- ChromaprintEncoder: Generate audio fingerprints using Chromaprint
- FingerprintDatabase: Store and search fingerprints for similarity matching
"""

from __future__ import annotations

import hashlib
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import FingerprintConfig
from .exceptions import (
    DatabaseError,
    DatabaseNotFoundError,
    FingerprintError,
    FingerprintNotAvailableError,
)

# Optional dependency
try:
    import acoustid

    CHROMAPRINT_AVAILABLE = True
except ImportError:
    CHROMAPRINT_AVAILABLE = False


@dataclass
class FingerprintMatch:
    """Result of a fingerprint match.

    Attributes:
        entry_id: ID of the matched entry in the database.
        name: Name/identifier of the matched content.
        similarity: Similarity score (0-1), where 1 is exact match.
        metadata: Additional metadata about the matched content.
    """

    entry_id: int
    name: str
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FingerprintEntry:
    """Entry in the fingerprint database.

    Attributes:
        entry_id: Unique identifier.
        name: Human-readable name (e.g., song title).
        fingerprint: The Chromaprint fingerprint string.
        duration: Audio duration in seconds.
        metadata: Additional metadata (artist, album, etc.).
    """

    entry_id: int
    name: str
    fingerprint: str
    duration: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ChromaprintEncoder:
    """Encoder for generating Chromaprint audio fingerprints.

    Uses the Chromaprint library (via pyacoustid) to generate
    fingerprints that can be used for audio identification.

    Example:
        >>> encoder = ChromaprintEncoder()
        >>> fingerprint, duration = encoder.encode(audio, sample_rate=44100)
        >>> print(f"Fingerprint length: {len(fingerprint)}")
    """

    def __init__(self, config: FingerprintConfig | None = None) -> None:
        """Initialize the encoder.

        Args:
            config: Fingerprint configuration.

        Raises:
            FingerprintNotAvailableError: If Chromaprint is not available.
        """
        self.config = config or FingerprintConfig()
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if Chromaprint is available."""
        if not CHROMAPRINT_AVAILABLE:
            raise FingerprintNotAvailableError()

    def encode(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[str, float]:
        """Generate a fingerprint from audio.

        Args:
            audio: Audio waveform (mono, float32, normalized).
            sample_rate: Sample rate of the audio.

        Returns:
            Tuple of (fingerprint string, duration in seconds).

        Raises:
            FingerprintError: If fingerprint generation fails.
        """
        duration = len(audio) / sample_rate

        if duration < self.config.min_duration:
            raise FingerprintError(
                f"Audio too short for fingerprinting: {duration:.2f}s < {self.config.min_duration}s"
            )

        # Chromaprint expects 16-bit PCM audio
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write to temporary WAV file (acoustid.fingerprint_file expects a file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with wave.open(tmp_path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(sample_rate)
                wav.writeframes(audio_int16.tobytes())

            # Generate fingerprint
            result = acoustid.fingerprint_file(tmp_path)
            fingerprint_duration, fingerprint = result

            return fingerprint, float(fingerprint_duration)

        except Exception as e:
            raise FingerprintError(f"Failed to generate fingerprint: {e}") from e
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    def encode_from_file(self, path: str | Path) -> tuple[str, float]:
        """Generate a fingerprint from an audio file.

        Args:
            path: Path to the audio file.

        Returns:
            Tuple of (fingerprint string, duration in seconds).

        Raises:
            FingerprintError: If fingerprint generation fails.
        """
        try:
            result = acoustid.fingerprint_file(str(path))
            duration, fingerprint = result
            return fingerprint, float(duration)
        except Exception as e:
            raise FingerprintError(f"Failed to fingerprint file: {e}") from e

    @staticmethod
    def compare_fingerprints(fp1: str, fp2: str) -> float:
        """Compare two fingerprints and return similarity.

        Uses a simple hash-based comparison for now.
        For more accurate comparison, use acoustid.compare.

        Args:
            fp1: First fingerprint string.
            fp2: Second fingerprint string.

        Returns:
            Similarity score (0-1).
        """
        # Handle empty fingerprints
        if not fp1 or not fp2:
            return 0.0

        if fp1 == fp2:
            return 1.0

        # Use a sliding window comparison
        # Fingerprints are base64-encoded, so we can compare them directly
        min_len = min(len(fp1), len(fp2))
        if min_len == 0:
            return 0.0

        matches = sum(c1 == c2 for c1, c2 in zip(fp1, fp2))
        similarity = matches / max(len(fp1), len(fp2))

        return similarity


class FingerprintDatabase:
    """Database for storing and searching audio fingerprints.

    Stores fingerprint entries in memory with optional file-based persistence.
    Provides methods for adding, searching, and managing fingerprints.

    Example:
        >>> db = FingerprintDatabase()
        >>> db.add("song_1", fingerprint, duration=180.0, metadata={"artist": "..."})
        >>> matches = db.search(query_fingerprint, top_k=5)
        >>> db.save("fingerprints/")
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        """Initialize empty fingerprint database.

        Args:
            similarity_threshold: Default threshold for similarity matching.
        """
        self._similarity_threshold = similarity_threshold

        # Storage
        self._entries: dict[int, FingerprintEntry] = {}
        self._next_id: int = 0

        # Name to ID mapping for quick lookup
        self._name_to_id: dict[str, int] = {}

    @property
    def similarity_threshold(self) -> float:
        """Default similarity threshold for matching."""
        return self._similarity_threshold

    def __len__(self) -> int:
        """Return number of entries in the database."""
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        """Check if an entry with the given name exists."""
        return name in self._name_to_id

    def add(
        self,
        name: str,
        fingerprint: str,
        duration: float,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a fingerprint entry to the database.

        Args:
            name: Human-readable name for the entry.
            fingerprint: Chromaprint fingerprint string.
            duration: Audio duration in seconds.
            metadata: Optional metadata.

        Returns:
            Assigned entry ID.
        """
        entry_id = self._next_id
        self._next_id += 1

        entry = FingerprintEntry(
            entry_id=entry_id,
            name=name,
            fingerprint=fingerprint,
            duration=duration,
            metadata=metadata or {},
        )

        self._entries[entry_id] = entry
        self._name_to_id[name] = entry_id

        return entry_id

    def remove(self, entry_id: int) -> FingerprintEntry:
        """Remove an entry from the database.

        Args:
            entry_id: ID of the entry to remove.

        Returns:
            The removed entry.

        Raises:
            DatabaseError: If entry_id doesn't exist.
        """
        if entry_id not in self._entries:
            raise DatabaseError(f"Entry with ID {entry_id} not found")

        entry = self._entries.pop(entry_id)
        self._name_to_id.pop(entry.name, None)

        return entry

    def get(self, entry_id: int) -> FingerprintEntry:
        """Get an entry by ID.

        Args:
            entry_id: ID of the entry.

        Returns:
            The entry.

        Raises:
            DatabaseError: If entry_id doesn't exist.
        """
        if entry_id not in self._entries:
            raise DatabaseError(f"Entry with ID {entry_id} not found")

        return self._entries[entry_id]

    def get_by_name(self, name: str) -> FingerprintEntry:
        """Get an entry by name.

        Args:
            name: Name of the entry.

        Returns:
            The entry.

        Raises:
            DatabaseError: If name doesn't exist.
        """
        if name not in self._name_to_id:
            raise DatabaseError(f"Entry with name '{name}' not found")

        entry_id = self._name_to_id[name]
        return self._entries[entry_id]

    def list_entries(self) -> list[FingerprintEntry]:
        """List all entries.

        Returns:
            List of all entries.
        """
        return list(self._entries.values())

    def search(
        self,
        query_fingerprint: str,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[FingerprintMatch]:
        """Search for similar fingerprints.

        Args:
            query_fingerprint: Query fingerprint string.
            top_k: Maximum number of results.
            threshold: Minimum similarity threshold (uses default if None).

        Returns:
            List of FingerprintMatch sorted by similarity (descending).
        """
        if len(self._entries) == 0:
            return []

        threshold = threshold if threshold is not None else self._similarity_threshold

        # Compute similarities
        similarities = []
        for entry_id, entry in self._entries.items():
            sim = ChromaprintEncoder.compare_fingerprints(
                query_fingerprint, entry.fingerprint
            )
            if sim >= threshold:
                similarities.append((entry_id, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for entry_id, sim in similarities[:top_k]:
            entry = self._entries[entry_id]
            results.append(
                FingerprintMatch(
                    entry_id=entry_id,
                    name=entry.name,
                    similarity=sim,
                    metadata=entry.metadata,
                )
            )

        return results

    def find_match(
        self,
        query_fingerprint: str,
        threshold: float | None = None,
    ) -> FingerprintMatch | None:
        """Find the best matching fingerprint above threshold.

        Args:
            query_fingerprint: Query fingerprint.
            threshold: Minimum similarity for a match.

        Returns:
            FingerprintMatch if found, None otherwise.
        """
        results = self.search(query_fingerprint, top_k=1, threshold=threshold)
        return results[0] if results else None

    def save(self, path: Path | str) -> Path:
        """Save database to disk.

        Args:
            path: Directory to save database.

        Returns:
            Path to saved database.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save as simple JSON-like format using numpy
        entries_data = []
        for entry in self._entries.values():
            entries_data.append(
                {
                    "entry_id": entry.entry_id,
                    "name": entry.name,
                    "fingerprint": entry.fingerprint,
                    "duration": entry.duration,
                    "metadata": entry.metadata,
                }
            )

        import json

        db_file = path / "fingerprints.json"
        with open(db_file, "w") as f:
            json.dump(
                {
                    "similarity_threshold": self._similarity_threshold,
                    "next_id": self._next_id,
                    "entries": entries_data,
                },
                f,
                indent=2,
            )

        return db_file

    @classmethod
    def load(cls, path: Path | str) -> "FingerprintDatabase":
        """Load database from disk.

        Args:
            path: Directory containing saved database.

        Returns:
            FingerprintDatabase instance.

        Raises:
            DatabaseNotFoundError: If database files not found.
        """
        path = Path(path)
        db_file = path / "fingerprints.json"

        if not db_file.exists():
            raise DatabaseNotFoundError(f"Database file not found: {db_file}")

        import json

        with open(db_file) as f:
            data = json.load(f)

        db = cls(similarity_threshold=data.get("similarity_threshold", 0.85))
        db._next_id = data.get("next_id", 0)

        for entry_data in data.get("entries", []):
            entry = FingerprintEntry(
                entry_id=entry_data["entry_id"],
                name=entry_data["name"],
                fingerprint=entry_data["fingerprint"],
                duration=entry_data["duration"],
                metadata=entry_data.get("metadata", {}),
            )
            db._entries[entry.entry_id] = entry
            db._name_to_id[entry.name] = entry.entry_id

        return db

    def clear(self) -> None:
        """Remove all entries from the database."""
        self._entries.clear()
        self._name_to_id.clear()
        self._next_id = 0


def is_chromaprint_available() -> bool:
    """Check if Chromaprint is available.

    Returns:
        True if Chromaprint can be used, False otherwise.
    """
    return CHROMAPRINT_AVAILABLE
