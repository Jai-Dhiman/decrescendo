"""Tests for fingerprint module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.copyright.fingerprint import (
    ChromaprintEncoder,
    FingerprintDatabase,
    FingerprintEntry,
    FingerprintMatch,
    is_chromaprint_available,
)
from decrescendo.musicritic.dimensions.copyright.exceptions import (
    DatabaseError,
    DatabaseNotFoundError,
    FingerprintError,
    FingerprintNotAvailableError,
)
from decrescendo.musicritic.dimensions.copyright.config import FingerprintConfig


class TestIsChromaprintAvailable:
    """Tests for is_chromaprint_available function."""

    def test_returns_boolean(self):
        """Should return a boolean."""
        result = is_chromaprint_available()
        assert isinstance(result, bool)


class TestFingerprintConfig:
    """Tests for FingerprintConfig."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = FingerprintConfig()
        assert config.min_duration == 1.0
        assert config.target_sample_rate == 11025

    def test_frozen(self):
        """Config should be frozen (immutable)."""
        config = FingerprintConfig()
        with pytest.raises(AttributeError):
            config.min_duration = 2.0  # type: ignore


class TestFingerprintEntry:
    """Tests for FingerprintEntry dataclass."""

    def test_create_entry(self):
        """Should create entry with all fields."""
        entry = FingerprintEntry(
            entry_id=1,
            name="test_song",
            fingerprint="AQAA...",
            duration=180.0,
            metadata={"artist": "Test Artist"},
        )
        assert entry.entry_id == 1
        assert entry.name == "test_song"
        assert entry.fingerprint == "AQAA..."
        assert entry.duration == 180.0
        assert entry.metadata["artist"] == "Test Artist"

    def test_default_metadata(self):
        """Should have empty metadata by default."""
        entry = FingerprintEntry(
            entry_id=1,
            name="test",
            fingerprint="fp",
            duration=10.0,
        )
        assert entry.metadata == {}


class TestFingerprintMatch:
    """Tests for FingerprintMatch dataclass."""

    def test_create_match(self):
        """Should create match with all fields."""
        match = FingerprintMatch(
            entry_id=1,
            name="matched_song",
            similarity=0.95,
            metadata={"source": "db"},
        )
        assert match.entry_id == 1
        assert match.name == "matched_song"
        assert match.similarity == 0.95
        assert match.metadata["source"] == "db"


class TestFingerprintDatabase:
    """Tests for FingerprintDatabase."""

    def test_init_empty(self):
        """Should initialize empty database."""
        db = FingerprintDatabase()
        assert len(db) == 0
        assert db.similarity_threshold == 0.85

    def test_init_custom_threshold(self):
        """Should accept custom similarity threshold."""
        db = FingerprintDatabase(similarity_threshold=0.9)
        assert db.similarity_threshold == 0.9

    def test_add_entry(self):
        """Should add entry and return ID."""
        db = FingerprintDatabase()
        entry_id = db.add("song1", "fp1", duration=100.0)
        assert entry_id == 0
        assert len(db) == 1
        assert "song1" in db

    def test_add_multiple_entries(self):
        """Should assign unique IDs to entries."""
        db = FingerprintDatabase()
        id1 = db.add("song1", "fp1", duration=100.0)
        id2 = db.add("song2", "fp2", duration=200.0)
        assert id1 != id2
        assert len(db) == 2

    def test_add_with_metadata(self):
        """Should store metadata with entry."""
        db = FingerprintDatabase()
        db.add("song1", "fp1", duration=100.0, metadata={"artist": "Test"})
        entry = db.get_by_name("song1")
        assert entry.metadata["artist"] == "Test"

    def test_get_entry(self):
        """Should retrieve entry by ID."""
        db = FingerprintDatabase()
        entry_id = db.add("song1", "fp1", duration=100.0)
        entry = db.get(entry_id)
        assert entry.name == "song1"
        assert entry.fingerprint == "fp1"

    def test_get_nonexistent_raises(self):
        """Should raise DatabaseError for nonexistent ID."""
        db = FingerprintDatabase()
        with pytest.raises(DatabaseError):
            db.get(999)

    def test_get_by_name(self):
        """Should retrieve entry by name."""
        db = FingerprintDatabase()
        db.add("song1", "fp1", duration=100.0)
        entry = db.get_by_name("song1")
        assert entry.name == "song1"

    def test_get_by_name_nonexistent_raises(self):
        """Should raise DatabaseError for nonexistent name."""
        db = FingerprintDatabase()
        with pytest.raises(DatabaseError):
            db.get_by_name("nonexistent")

    def test_remove_entry(self):
        """Should remove entry by ID."""
        db = FingerprintDatabase()
        entry_id = db.add("song1", "fp1", duration=100.0)
        removed = db.remove(entry_id)
        assert removed.name == "song1"
        assert len(db) == 0
        assert "song1" not in db

    def test_remove_nonexistent_raises(self):
        """Should raise DatabaseError for nonexistent ID."""
        db = FingerprintDatabase()
        with pytest.raises(DatabaseError):
            db.remove(999)

    def test_list_entries(self):
        """Should list all entries."""
        db = FingerprintDatabase()
        db.add("song1", "fp1", duration=100.0)
        db.add("song2", "fp2", duration=200.0)
        entries = db.list_entries()
        assert len(entries) == 2
        names = {e.name for e in entries}
        assert names == {"song1", "song2"}

    def test_search_empty_db(self):
        """Should return empty list for empty database."""
        db = FingerprintDatabase()
        results = db.search("query_fp")
        assert results == []

    def test_search_exact_match(self):
        """Should find exact fingerprint match."""
        db = FingerprintDatabase()
        db.add("song1", "AQAA123", duration=100.0)
        results = db.search("AQAA123", threshold=0.9)
        assert len(results) == 1
        assert results[0].name == "song1"
        assert results[0].similarity == 1.0

    def test_search_top_k(self):
        """Should respect top_k limit."""
        db = FingerprintDatabase()
        for i in range(10):
            db.add(f"song{i}", f"fp{i}", duration=100.0)
        results = db.search("fp5", top_k=3, threshold=0.0)
        assert len(results) <= 3

    def test_search_threshold(self):
        """Should filter by threshold."""
        db = FingerprintDatabase()
        db.add("song1", "AAAA", duration=100.0)
        db.add("song2", "BBBB", duration=100.0)
        results = db.search("AAAA", threshold=0.99)
        # Only exact match should pass high threshold
        assert len(results) <= 1

    def test_find_match(self):
        """Should find best match."""
        db = FingerprintDatabase()
        db.add("song1", "test_fp", duration=100.0)
        match = db.find_match("test_fp")
        assert match is not None
        assert match.name == "song1"

    def test_find_match_none(self):
        """Should return None if no match above threshold."""
        db = FingerprintDatabase()
        db.add("song1", "AAAA", duration=100.0)
        match = db.find_match("ZZZZ", threshold=0.99)
        assert match is None

    def test_contains(self):
        """Should support 'in' operator."""
        db = FingerprintDatabase()
        db.add("song1", "fp1", duration=100.0)
        assert "song1" in db
        assert "song2" not in db

    def test_clear(self):
        """Should clear all entries."""
        db = FingerprintDatabase()
        db.add("song1", "fp1", duration=100.0)
        db.add("song2", "fp2", duration=200.0)
        db.clear()
        assert len(db) == 0

    def test_save_and_load(self):
        """Should save and load database."""
        db = FingerprintDatabase()
        db.add("song1", "fp1", duration=100.0, metadata={"artist": "Test"})
        db.add("song2", "fp2", duration=200.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            db.save(tmpdir)
            loaded_db = FingerprintDatabase.load(tmpdir)

        assert len(loaded_db) == 2
        assert "song1" in loaded_db
        assert "song2" in loaded_db
        entry = loaded_db.get_by_name("song1")
        assert entry.metadata["artist"] == "Test"

    def test_load_nonexistent_raises(self):
        """Should raise DatabaseNotFoundError for missing file."""
        with pytest.raises(DatabaseNotFoundError):
            FingerprintDatabase.load("/nonexistent/path")


@pytest.mark.skipif(
    not is_chromaprint_available(),
    reason="Chromaprint not available",
)
class TestChromaprintEncoder:
    """Tests for ChromaprintEncoder (requires Chromaprint)."""

    def test_init(self):
        """Should initialize encoder."""
        encoder = ChromaprintEncoder()
        assert encoder.config is not None

    def test_encode_audio(self, sine_440hz: np.ndarray, sample_rate: int):
        """Should encode audio to fingerprint."""
        encoder = ChromaprintEncoder()
        fp, duration = encoder.encode(sine_440hz, sample_rate)
        assert isinstance(fp, str)
        assert len(fp) > 0
        assert duration > 0

    def test_encode_short_audio_raises(self, short_audio: np.ndarray, sample_rate: int):
        """Should raise for audio shorter than min_duration."""
        encoder = ChromaprintEncoder()
        with pytest.raises(FingerprintError):
            encoder.encode(short_audio, sample_rate)

    def test_compare_identical(self, sine_440hz: np.ndarray, sample_rate: int):
        """Identical audio should have similarity 1.0."""
        encoder = ChromaprintEncoder()
        fp1, _ = encoder.encode(sine_440hz, sample_rate)
        similarity = ChromaprintEncoder.compare_fingerprints(fp1, fp1)
        assert similarity == 1.0

    def test_compare_different(
        self, sine_440hz: np.ndarray, sine_880hz: np.ndarray, sample_rate: int
    ):
        """Different audio should have lower similarity."""
        encoder = ChromaprintEncoder()
        fp1, _ = encoder.encode(sine_440hz, sample_rate)
        fp2, _ = encoder.encode(sine_880hz, sample_rate)
        similarity = ChromaprintEncoder.compare_fingerprints(fp1, fp2)
        assert 0.0 <= similarity < 1.0


class TestChromaprintEncoderUnavailable:
    """Tests for ChromaprintEncoder when Chromaprint is not available."""

    @pytest.mark.skipif(
        is_chromaprint_available(),
        reason="Chromaprint is available",
    )
    def test_init_raises_when_unavailable(self):
        """Should raise FingerprintNotAvailableError if Chromaprint not installed."""
        with pytest.raises(FingerprintNotAvailableError):
            ChromaprintEncoder()


class TestCompareFingerprints:
    """Tests for fingerprint comparison."""

    def test_compare_empty_fingerprints(self):
        """Empty fingerprints should return 0 similarity."""
        assert ChromaprintEncoder.compare_fingerprints("", "") == 0.0
        assert ChromaprintEncoder.compare_fingerprints("abc", "") == 0.0
        assert ChromaprintEncoder.compare_fingerprints("", "abc") == 0.0

    def test_compare_identical(self):
        """Identical fingerprints should return 1.0."""
        assert ChromaprintEncoder.compare_fingerprints("AQAA", "AQAA") == 1.0

    def test_compare_similar(self):
        """Similar fingerprints should return high similarity."""
        sim = ChromaprintEncoder.compare_fingerprints("AAAA", "AAAB")
        assert 0.5 < sim < 1.0

    def test_compare_different(self):
        """Different fingerprints should return low similarity."""
        sim = ChromaprintEncoder.compare_fingerprints("AAAA", "ZZZZ")
        assert sim < 0.5
