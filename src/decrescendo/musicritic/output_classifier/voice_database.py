"""Voice database for managing protected voices.

Provides a VoiceDatabase class for storing, searching, and managing
protected voice embeddings. Supports file-based persistence and
optional FAISS integration for large-scale similarity search.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from .checkpointing import (
    VoiceEntry,
    VoiceDatabaseError,
    VoiceDatabaseNotFoundError,
    load_voice_database,
    save_voice_database,
)
from .model import compare_against_protected_voices


class VoiceNotFoundError(VoiceDatabaseError):
    """Raised when a voice is not found in the database."""

    pass


class VoiceDuplicateError(VoiceDatabaseError):
    """Raised when attempting to add a duplicate voice."""

    pass


@dataclass
class SimilarityResult:
    """Result of a similarity search.

    Attributes:
        voice_id: ID of the matched voice
        name: Name of the matched voice
        similarity: Cosine similarity score (0-1)
        metadata: Additional metadata for the matched voice
    """

    voice_id: int
    name: str
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VoiceDatabase:
    """Database for managing protected voice embeddings.

    Stores voice embeddings in memory with file-based persistence.
    Provides methods for adding, removing, searching, and listing voices.

    Example:
        >>> db = VoiceDatabase(embedding_dim=192)
        >>> db.add_voice("artist_1", embedding, metadata={"genre": "pop"})
        >>> results = db.search(query_embedding, top_k=5)
        >>> db.save("voices/")

        >>> # Load existing database
        >>> db = VoiceDatabase.load("voices/")
        >>> print(f"Loaded {len(db)} voices")
    """

    def __init__(
        self,
        embedding_dim: int = 192,
        similarity_threshold: float = 0.85,
    ) -> None:
        """Initialize empty voice database.

        Args:
            embedding_dim: Dimension of voice embeddings
            similarity_threshold: Default threshold for similarity matching
        """
        self._embedding_dim = embedding_dim
        self._similarity_threshold = similarity_threshold

        # Storage
        self._entries: dict[int, VoiceEntry] = {}
        self._embeddings: dict[int, np.ndarray] = {}
        self._next_id: int = 0

        # Name to ID mapping for quick lookup
        self._name_to_id: dict[str, int] = {}

    @property
    def embedding_dim(self) -> int:
        """Dimension of voice embeddings."""
        return self._embedding_dim

    @property
    def similarity_threshold(self) -> float:
        """Default similarity threshold for matching."""
        return self._similarity_threshold

    def __len__(self) -> int:
        """Return number of voices in the database."""
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        """Check if a voice with the given name exists."""
        return name in self._name_to_id

    def add_voice(
        self,
        name: str,
        embedding: np.ndarray | jnp.ndarray,
        metadata: dict[str, Any] | None = None,
        allow_duplicate_name: bool = False,
    ) -> int:
        """Add a new protected voice to the database.

        Args:
            name: Human-readable name for the voice (e.g., artist name)
            embedding: Voice embedding vector (embedding_dim,)
            metadata: Optional metadata (e.g., genre, consent info)
            allow_duplicate_name: If False, raises error for duplicate names

        Returns:
            Assigned voice ID

        Raises:
            VoiceDuplicateError: If name already exists and allow_duplicate_name is False
            ValueError: If embedding dimension doesn't match
        """
        # Validate embedding
        embedding = np.asarray(embedding).flatten()
        if embedding.shape[0] != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                f"got {embedding.shape[0]}"
            )

        # Check for duplicate name
        if not allow_duplicate_name and name in self._name_to_id:
            raise VoiceDuplicateError(f"Voice with name '{name}' already exists")

        # L2 normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Assign ID and store
        voice_id = self._next_id
        self._next_id += 1

        entry = VoiceEntry(
            voice_id=voice_id,
            name=name,
            metadata=metadata or {},
        )
        self._entries[voice_id] = entry
        self._embeddings[voice_id] = embedding.astype(np.float32)
        self._name_to_id[name] = voice_id

        return voice_id

    def remove_voice(self, voice_id: int) -> VoiceEntry:
        """Remove a voice from the database.

        Args:
            voice_id: ID of the voice to remove

        Returns:
            The removed VoiceEntry

        Raises:
            VoiceNotFoundError: If voice_id doesn't exist
        """
        if voice_id not in self._entries:
            raise VoiceNotFoundError(f"Voice with ID {voice_id} not found")

        entry = self._entries.pop(voice_id)
        self._embeddings.pop(voice_id)
        self._name_to_id.pop(entry.name, None)

        return entry

    def remove_voice_by_name(self, name: str) -> VoiceEntry:
        """Remove a voice by name.

        Args:
            name: Name of the voice to remove

        Returns:
            The removed VoiceEntry

        Raises:
            VoiceNotFoundError: If name doesn't exist
        """
        if name not in self._name_to_id:
            raise VoiceNotFoundError(f"Voice with name '{name}' not found")

        voice_id = self._name_to_id[name]
        return self.remove_voice(voice_id)

    def get_voice(self, voice_id: int) -> tuple[VoiceEntry, np.ndarray]:
        """Get a voice entry and its embedding.

        Args:
            voice_id: ID of the voice

        Returns:
            Tuple of (VoiceEntry, embedding)

        Raises:
            VoiceNotFoundError: If voice_id doesn't exist
        """
        if voice_id not in self._entries:
            raise VoiceNotFoundError(f"Voice with ID {voice_id} not found")

        return self._entries[voice_id], self._embeddings[voice_id]

    def get_voice_by_name(self, name: str) -> tuple[VoiceEntry, np.ndarray]:
        """Get a voice entry by name.

        Args:
            name: Name of the voice

        Returns:
            Tuple of (VoiceEntry, embedding)

        Raises:
            VoiceNotFoundError: If name doesn't exist
        """
        if name not in self._name_to_id:
            raise VoiceNotFoundError(f"Voice with name '{name}' not found")

        voice_id = self._name_to_id[name]
        return self.get_voice(voice_id)

    def list_voices(self) -> list[VoiceEntry]:
        """List all voice entries.

        Returns:
            List of VoiceEntry objects
        """
        return list(self._entries.values())

    def search(
        self,
        query_embedding: np.ndarray | jnp.ndarray,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[SimilarityResult]:
        """Search for similar voices.

        Args:
            query_embedding: Query voice embedding (embedding_dim,)
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold (uses default if None)

        Returns:
            List of SimilarityResult sorted by similarity (descending)
        """
        if len(self._entries) == 0:
            return []

        threshold = threshold if threshold is not None else self._similarity_threshold

        # Prepare query embedding
        query = np.asarray(query_embedding).flatten()
        if query.shape[0] != self._embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self._embedding_dim}, "
                f"got {query.shape[0]}"
            )

        # L2 normalize
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Compute similarities against all voices
        voice_ids = list(self._embeddings.keys())
        embeddings_matrix = np.stack([self._embeddings[vid] for vid in voice_ids])

        # Cosine similarity (embeddings are already normalized)
        similarities = np.dot(embeddings_matrix, query)

        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices[:top_k]:
            sim = float(similarities[idx])
            if sim < threshold:
                break

            voice_id = voice_ids[idx]
            entry = self._entries[voice_id]
            results.append(
                SimilarityResult(
                    voice_id=voice_id,
                    name=entry.name,
                    similarity=sim,
                    metadata=entry.metadata,
                )
            )

        return results

    def find_match(
        self,
        query_embedding: np.ndarray | jnp.ndarray,
        threshold: float | None = None,
    ) -> SimilarityResult | None:
        """Find the best matching voice above threshold.

        Args:
            query_embedding: Query voice embedding
            threshold: Minimum similarity for a match (uses default if None)

        Returns:
            SimilarityResult if a match is found, None otherwise
        """
        results = self.search(query_embedding, top_k=1, threshold=threshold)
        return results[0] if results else None

    def check_duplicate(
        self,
        embedding: np.ndarray | jnp.ndarray,
        threshold: float = 0.95,
    ) -> SimilarityResult | None:
        """Check if an embedding is a duplicate of an existing voice.

        Uses a high threshold to detect near-duplicates.

        Args:
            embedding: Voice embedding to check
            threshold: Similarity threshold for duplicate detection

        Returns:
            SimilarityResult of the duplicate if found, None otherwise
        """
        return self.find_match(embedding, threshold=threshold)

    def get_all_embeddings(self) -> tuple[jnp.ndarray, list[str]]:
        """Get all embeddings as a JAX array for batch processing.

        Returns:
            Tuple of (embeddings array, list of names)
            embeddings: (num_voices, embedding_dim)
        """
        if len(self._entries) == 0:
            return jnp.zeros((0, self._embedding_dim)), []

        voice_ids = sorted(self._embeddings.keys())
        embeddings = np.stack([self._embeddings[vid] for vid in voice_ids])
        names = [self._entries[vid].name for vid in voice_ids]

        return jnp.array(embeddings), names

    def batch_search(
        self,
        query_embeddings: np.ndarray | jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Batch similarity search using JAX.

        Efficient for comparing multiple queries against the database.

        Args:
            query_embeddings: Query embeddings (num_queries, embedding_dim)

        Returns:
            Tuple of (max_similarities, best_match_indices)
            Each has shape (num_queries,)
        """
        if len(self._entries) == 0:
            num_queries = query_embeddings.shape[0] if query_embeddings.ndim > 1 else 1
            return jnp.zeros(num_queries), jnp.zeros(num_queries, dtype=jnp.int32)

        protected_embeddings, _ = self.get_all_embeddings()
        return compare_against_protected_voices(query_embeddings, protected_embeddings)

    def save(self, path: Path | str) -> Path:
        """Save database to disk.

        Args:
            path: Directory to save database

        Returns:
            Path to saved database
        """
        if len(self._entries) == 0:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            # Save empty database
            return save_voice_database(path, np.zeros((0, self._embedding_dim)), [])

        # Prepare data in consistent order
        voice_ids = sorted(self._entries.keys())
        entries = [self._entries[vid] for vid in voice_ids]
        embeddings = np.stack([self._embeddings[vid] for vid in voice_ids])

        return save_voice_database(path, embeddings, entries)

    @classmethod
    def load(cls, path: Path | str) -> "VoiceDatabase":
        """Load database from disk.

        Args:
            path: Directory containing saved database

        Returns:
            VoiceDatabase instance

        Raises:
            VoiceDatabaseNotFoundError: If database files not found
            VoiceDatabaseError: If database is corrupted
        """
        embeddings, entries = load_voice_database(path)

        # Determine embedding dimension
        if len(embeddings) > 0:
            embedding_dim = embeddings.shape[1]
        else:
            embedding_dim = 192  # Default

        db = cls(embedding_dim=embedding_dim)

        # Restore entries and embeddings
        for entry, embedding in zip(entries, np.asarray(embeddings)):
            db._entries[entry.voice_id] = entry
            db._embeddings[entry.voice_id] = embedding.astype(np.float32)
            db._name_to_id[entry.name] = entry.voice_id

            # Track next ID
            if entry.voice_id >= db._next_id:
                db._next_id = entry.voice_id + 1

        return db

    def update_metadata(
        self,
        voice_id: int,
        metadata: dict[str, Any],
        merge: bool = True,
    ) -> VoiceEntry:
        """Update metadata for a voice.

        Args:
            voice_id: ID of the voice to update
            metadata: New metadata
            merge: If True, merge with existing metadata; if False, replace

        Returns:
            Updated VoiceEntry

        Raises:
            VoiceNotFoundError: If voice_id doesn't exist
        """
        if voice_id not in self._entries:
            raise VoiceNotFoundError(f"Voice with ID {voice_id} not found")

        entry = self._entries[voice_id]

        if merge:
            new_metadata = {**entry.metadata, **metadata}
        else:
            new_metadata = metadata

        updated_entry = VoiceEntry(
            voice_id=entry.voice_id,
            name=entry.name,
            metadata=new_metadata,
        )
        self._entries[voice_id] = updated_entry

        return updated_entry

    def clear(self) -> None:
        """Remove all voices from the database."""
        self._entries.clear()
        self._embeddings.clear()
        self._name_to_id.clear()
        self._next_id = 0
