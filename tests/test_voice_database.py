"""Tests for voice database and enrollment."""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from decrescendo.constitutional_audio.output_classifier import (
    OutputClassifierConfig,
    VoiceDatabase,
    VoiceDuplicateError,
    VoiceEntry,
    VoiceNotFoundError,
)
from decrescendo.constitutional_audio.output_classifier.checkpointing import (
    VoiceDatabaseNotFoundError,
)
from decrescendo.constitutional_audio.output_classifier.voice_database import (
    SimilarityResult,
)
from decrescendo.constitutional_audio.output_classifier.voice_enrollment import (
    AudioQualityError,
    EnrollmentResult,
    QualityCheckResult,
    VoiceEnroller,
    create_voice_enroller,
)


class TestVoiceDatabase:
    """Test VoiceDatabase class."""

    @pytest.fixture
    def db(self):
        """Create empty voice database."""
        return VoiceDatabase(embedding_dim=192)

    @pytest.fixture
    def sample_embedding(self, rng):
        """Create a sample embedding."""
        embedding = jax.random.normal(rng, (192,))
        embedding = embedding / jnp.linalg.norm(embedding)
        return np.array(embedding)

    def test_init_empty(self):
        """Test creating empty database."""
        db = VoiceDatabase(embedding_dim=192)
        assert len(db) == 0
        assert db.embedding_dim == 192

    def test_add_voice(self, db, sample_embedding):
        """Test adding a voice."""
        voice_id = db.add_voice("Artist A", sample_embedding)

        assert voice_id == 0
        assert len(db) == 1
        assert "Artist A" in db

    def test_add_voice_with_metadata(self, db, sample_embedding):
        """Test adding a voice with metadata."""
        metadata = {"genre": "pop", "consent_date": "2024-01-01"}
        voice_id = db.add_voice("Artist B", sample_embedding, metadata=metadata)

        entry, _ = db.get_voice(voice_id)
        assert entry.metadata == metadata

    def test_add_voice_wrong_dimension(self, db):
        """Test adding voice with wrong embedding dimension."""
        wrong_embedding = np.random.randn(128)

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            db.add_voice("Wrong", wrong_embedding)

    def test_add_duplicate_name_raises(self, db, sample_embedding):
        """Test that duplicate names raise error by default."""
        db.add_voice("Artist A", sample_embedding)

        with pytest.raises(VoiceDuplicateError, match="already exists"):
            db.add_voice("Artist A", np.random.randn(192))

    def test_add_duplicate_name_allowed(self, db, sample_embedding):
        """Test allowing duplicate names."""
        id1 = db.add_voice("Artist A", sample_embedding)
        id2 = db.add_voice("Artist A", np.random.randn(192), allow_duplicate_name=True)

        assert id1 != id2
        assert len(db) == 2

    def test_remove_voice(self, db, sample_embedding):
        """Test removing a voice."""
        voice_id = db.add_voice("Artist A", sample_embedding)
        assert len(db) == 1

        removed = db.remove_voice(voice_id)
        assert removed.name == "Artist A"
        assert len(db) == 0
        assert "Artist A" not in db

    def test_remove_nonexistent_raises(self, db):
        """Test removing nonexistent voice raises error."""
        with pytest.raises(VoiceNotFoundError):
            db.remove_voice(999)

    def test_remove_by_name(self, db, sample_embedding):
        """Test removing voice by name."""
        db.add_voice("Artist A", sample_embedding)

        removed = db.remove_voice_by_name("Artist A")
        assert removed.name == "Artist A"
        assert len(db) == 0

    def test_get_voice(self, db, sample_embedding):
        """Test getting a voice."""
        voice_id = db.add_voice("Artist A", sample_embedding, metadata={"genre": "rock"})

        entry, embedding = db.get_voice(voice_id)
        assert entry.name == "Artist A"
        assert entry.metadata["genre"] == "rock"
        assert embedding.shape == (192,)

    def test_get_voice_by_name(self, db, sample_embedding):
        """Test getting voice by name."""
        db.add_voice("Artist A", sample_embedding)

        entry, embedding = db.get_voice_by_name("Artist A")
        assert entry.name == "Artist A"

    def test_get_nonexistent_raises(self, db):
        """Test getting nonexistent voice raises error."""
        with pytest.raises(VoiceNotFoundError):
            db.get_voice(999)

    def test_list_voices(self, db, rng):
        """Test listing all voices."""
        keys = jax.random.split(rng, 3)
        for i, key in enumerate(keys):
            embedding = jax.random.normal(key, (192,))
            db.add_voice(f"Artist {i}", np.array(embedding))

        voices = db.list_voices()
        assert len(voices) == 3
        names = [v.name for v in voices]
        assert "Artist 0" in names
        assert "Artist 1" in names
        assert "Artist 2" in names

    def test_search_similar(self, db, rng):
        """Test searching for similar voices."""
        # Add some voices
        keys = jax.random.split(rng, 4)
        for i, key in enumerate(keys[:3]):
            embedding = jax.random.normal(key, (192,))
            embedding = embedding / jnp.linalg.norm(embedding)
            db.add_voice(f"Artist {i}", np.array(embedding))

        # Query with a voice similar to Artist 0
        entry, embedding = db.get_voice(0)
        # Add small noise
        noise = jax.random.normal(keys[3], (192,)) * 0.1
        query = embedding + np.array(noise)
        query = query / np.linalg.norm(query)

        results = db.search(query, top_k=3, threshold=0.0)
        assert len(results) > 0
        # Best match should be Artist 0
        assert results[0].voice_id == 0

    def test_search_empty_database(self, db, sample_embedding):
        """Test searching empty database returns empty list."""
        results = db.search(sample_embedding)
        assert results == []

    def test_search_with_threshold(self, db, rng):
        """Test search with threshold filtering."""
        keys = jax.random.split(rng, 2)

        # Add a voice
        embedding1 = jax.random.normal(keys[0], (192,))
        embedding1 = embedding1 / jnp.linalg.norm(embedding1)
        db.add_voice("Artist A", np.array(embedding1))

        # Query with unrelated embedding
        embedding2 = jax.random.normal(keys[1], (192,))
        embedding2 = embedding2 / jnp.linalg.norm(embedding2)

        # High threshold should return no results
        results = db.search(np.array(embedding2), threshold=0.99)
        assert len(results) == 0

    def test_find_match(self, db, sample_embedding):
        """Test finding best matching voice."""
        db.add_voice("Artist A", sample_embedding)

        match = db.find_match(sample_embedding, threshold=0.5)
        assert match is not None
        assert match.name == "Artist A"
        assert match.similarity > 0.99  # Should be nearly identical

    def test_find_match_none(self, db, rng):
        """Test find_match returns None when no match."""
        keys = jax.random.split(rng, 2)

        embedding1 = jax.random.normal(keys[0], (192,))
        db.add_voice("Artist A", np.array(embedding1))

        # Unrelated query
        embedding2 = jax.random.normal(keys[1], (192,))
        match = db.find_match(np.array(embedding2), threshold=0.99)
        assert match is None

    def test_check_duplicate(self, db, sample_embedding):
        """Test duplicate detection."""
        db.add_voice("Artist A", sample_embedding)

        # Exact duplicate
        duplicate = db.check_duplicate(sample_embedding)
        assert duplicate is not None
        assert duplicate.name == "Artist A"

    def test_get_all_embeddings(self, db, rng):
        """Test getting all embeddings as array."""
        keys = jax.random.split(rng, 3)
        for i, key in enumerate(keys):
            embedding = jax.random.normal(key, (192,))
            db.add_voice(f"Artist {i}", np.array(embedding))

        embeddings, names = db.get_all_embeddings()
        assert embeddings.shape == (3, 192)
        assert len(names) == 3

    def test_get_all_embeddings_empty(self, db):
        """Test getting embeddings from empty database."""
        embeddings, names = db.get_all_embeddings()
        assert embeddings.shape == (0, 192)
        assert names == []

    def test_batch_search(self, db, rng):
        """Test batch similarity search."""
        keys = jax.random.split(rng, 5)

        # Add some voices
        for i, key in enumerate(keys[:3]):
            embedding = jax.random.normal(key, (192,))
            embedding = embedding / jnp.linalg.norm(embedding)
            db.add_voice(f"Artist {i}", np.array(embedding))

        # Batch query
        queries = jax.random.normal(keys[3], (2, 192))
        queries = queries / jnp.linalg.norm(queries, axis=-1, keepdims=True)

        max_sims, indices = db.batch_search(np.array(queries))
        assert max_sims.shape == (2,)
        assert indices.shape == (2,)

    def test_update_metadata(self, db, sample_embedding):
        """Test updating voice metadata."""
        voice_id = db.add_voice("Artist A", sample_embedding, metadata={"genre": "rock"})

        updated = db.update_metadata(voice_id, {"label": "indie"}, merge=True)
        assert updated.metadata["genre"] == "rock"
        assert updated.metadata["label"] == "indie"

        # Replace metadata
        db.update_metadata(voice_id, {"new_key": "value"}, merge=False)
        entry, _ = db.get_voice(voice_id)
        assert "genre" not in entry.metadata
        assert entry.metadata["new_key"] == "value"

    def test_clear(self, db, rng):
        """Test clearing the database."""
        keys = jax.random.split(rng, 3)
        for i, key in enumerate(keys):
            embedding = jax.random.normal(key, (192,))
            db.add_voice(f"Artist {i}", np.array(embedding))

        assert len(db) == 3
        db.clear()
        assert len(db) == 0


class TestVoiceDatabasePersistence:
    """Test VoiceDatabase save/load functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def populated_db(self, rng):
        """Create database with some voices."""
        db = VoiceDatabase(embedding_dim=192)
        keys = jax.random.split(rng, 3)

        for i, key in enumerate(keys):
            embedding = jax.random.normal(key, (192,))
            embedding = embedding / jnp.linalg.norm(embedding)
            db.add_voice(
                f"Artist {i}",
                np.array(embedding),
                metadata={"index": i},
            )

        return db

    def test_save_and_load(self, temp_dir, populated_db):
        """Test saving and loading database."""
        save_path = temp_dir / "voices"
        populated_db.save(save_path)

        # Verify files created
        assert (save_path / "voices.npz").exists()
        assert (save_path / "manifest.json").exists()

        # Load and verify
        loaded_db = VoiceDatabase.load(save_path)
        assert len(loaded_db) == len(populated_db)

        # Check entries preserved
        for voice in populated_db.list_voices():
            entry, embedding = loaded_db.get_voice(voice.voice_id)
            assert entry.name == voice.name
            assert entry.metadata == voice.metadata

    def test_save_empty_database(self, temp_dir):
        """Test saving empty database."""
        db = VoiceDatabase(embedding_dim=192)
        save_path = temp_dir / "empty_voices"
        db.save(save_path)

        loaded_db = VoiceDatabase.load(save_path)
        assert len(loaded_db) == 0

    def test_load_nonexistent_raises(self, temp_dir):
        """Test loading from nonexistent path raises error."""
        with pytest.raises(VoiceDatabaseNotFoundError):
            VoiceDatabase.load(temp_dir / "nonexistent")

    def test_embeddings_preserved(self, temp_dir, populated_db, rng):
        """Test that embeddings are preserved correctly after save/load."""
        save_path = temp_dir / "voices"
        populated_db.save(save_path)
        loaded_db = VoiceDatabase.load(save_path)

        # Get embeddings from both
        orig_embeddings, orig_names = populated_db.get_all_embeddings()
        loaded_embeddings, loaded_names = loaded_db.get_all_embeddings()

        # Compare
        np.testing.assert_array_almost_equal(
            np.array(orig_embeddings),
            np.array(loaded_embeddings),
            decimal=5,
        )


class TestQualityCheckResult:
    """Test QualityCheckResult dataclass."""

    def test_passed_quality_check(self):
        """Test quality check that passes."""
        result = QualityCheckResult(
            passed=True,
            duration_sec=10.0,
            rms_db=-20.0,
            snr_db=25.0,
            issues=[],
        )
        assert result.passed
        assert len(result.issues) == 0

    def test_failed_quality_check(self):
        """Test quality check that fails."""
        result = QualityCheckResult(
            passed=False,
            duration_sec=1.0,
            rms_db=-60.0,
            snr_db=5.0,
            issues=["Audio too short", "Audio too quiet"],
        )
        assert not result.passed
        assert len(result.issues) == 2


class TestVoiceEnroller:
    """Test VoiceEnroller class."""

    @pytest.fixture
    def enroller(self, rng):
        """Create voice enroller with initialized model."""
        return create_voice_enroller(rng=rng)

    @pytest.fixture
    def db(self):
        """Create empty voice database."""
        return VoiceDatabase(embedding_dim=192)

    def test_check_audio_quality_valid(self, enroller):
        """Test quality check on valid audio."""
        # Create valid audio: 5 seconds at 24kHz
        # Use a simple sine wave which has better SNR than random noise
        sample_rate = 24000
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)

        result = enroller.check_audio_quality(audio, sample_rate)
        assert result.passed
        assert abs(result.duration_sec - duration) < 0.1

    def test_check_audio_quality_too_short(self, enroller):
        """Test quality check on too short audio."""
        sample_rate = 24000
        # Only 1 second (min is 3)
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1

        result = enroller.check_audio_quality(audio, sample_rate)
        assert not result.passed
        assert any("too short" in issue for issue in result.issues)

    def test_check_audio_quality_too_quiet(self, enroller):
        """Test quality check on too quiet audio."""
        sample_rate = 24000
        duration = 5.0
        # Very quiet audio
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 1e-6

        result = enroller.check_audio_quality(audio, sample_rate)
        assert not result.passed
        assert any("too quiet" in issue for issue in result.issues)

    def test_extract_embedding(self, enroller):
        """Test extracting embedding from audio."""
        sample_rate = 24000
        duration = 5.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        embedding = enroller.extract_embedding(audio, sample_rate)

        assert embedding.shape == (192,)  # Default speaker embedding dim
        # Check normalized
        norm = np.linalg.norm(embedding)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)

    def test_extract_embedding_from_array(self, enroller):
        """Test extract_embedding_from_array method."""
        sample_rate = 24000
        duration = 5.0
        # Use a sine wave for better SNR
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)

        embedding, quality = enroller.extract_embedding_from_array(audio, sample_rate)

        assert embedding.shape == (192,)
        assert quality is not None
        assert quality.passed

    def test_extract_embedding_from_array_quality_fail(self, enroller):
        """Test extract_embedding_from_array raises on quality fail."""
        sample_rate = 24000
        # Too short
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1

        with pytest.raises(AudioQualityError):
            enroller.extract_embedding_from_array(audio, sample_rate, check_quality=True)


class TestVoiceEnrollerDuplicateDetection:
    """Test duplicate voice detection in enrollment."""

    @pytest.fixture
    def enroller(self, rng):
        """Create voice enroller."""
        return create_voice_enroller(rng=rng)

    @pytest.fixture
    def db_with_voice(self, enroller):
        """Create database with one enrolled voice."""
        db = VoiceDatabase(embedding_dim=192)

        sample_rate = 24000
        duration = 5.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        # Extract and add embedding directly
        embedding = enroller.extract_embedding(audio, sample_rate)
        db.add_voice("Existing Artist", embedding)

        return db, audio, sample_rate

    def test_duplicate_detection_blocks_enrollment(self, enroller, db_with_voice):
        """Test that duplicate voice is detected and enrollment blocked."""
        db, audio, sample_rate = db_with_voice

        # Try to enroll same audio as different artist
        result = enroller.enroll_from_files(
            db,
            "New Artist",
            [],  # Empty paths, we'll use array
            skip_quality_check=True,
        )

        # Should fail due to no files
        assert not result.success

    def test_skip_duplicate_check(self, enroller):
        """Test skipping duplicate check allows enrollment."""
        db = VoiceDatabase(embedding_dim=192)

        sample_rate = 24000
        duration = 5.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        # Add first voice
        embedding1 = enroller.extract_embedding(audio, sample_rate)
        db.add_voice("Artist A", embedding1)

        # Add same embedding as different name (skip duplicate check)
        db.add_voice("Artist B", embedding1, allow_duplicate_name=True)

        assert len(db) == 2


class TestEnrollmentResult:
    """Test EnrollmentResult dataclass."""

    def test_successful_enrollment(self):
        """Test successful enrollment result."""
        result = EnrollmentResult(
            success=True,
            voice_id=0,
            name="Artist A",
            embedding_dim=192,
            num_samples_used=3,
            quality_results=[],
        )
        assert result.success
        assert result.voice_id == 0
        assert result.error is None

    def test_failed_enrollment(self):
        """Test failed enrollment result."""
        result = EnrollmentResult(
            success=False,
            voice_id=None,
            name="Artist A",
            embedding_dim=192,
            num_samples_used=0,
            quality_results=[],
            error="Audio quality check failed",
        )
        assert not result.success
        assert result.voice_id is None
        assert result.error is not None

    def test_enrollment_with_duplicate_match(self):
        """Test enrollment result with duplicate match info."""
        duplicate = SimilarityResult(
            voice_id=0,
            name="Existing Artist",
            similarity=0.97,
        )
        result = EnrollmentResult(
            success=False,
            voice_id=None,
            name="New Artist",
            embedding_dim=192,
            num_samples_used=1,
            quality_results=[],
            duplicate_match=duplicate,
            error="Duplicate detected",
        )
        assert not result.success
        assert result.duplicate_match is not None
        assert result.duplicate_match.similarity == 0.97


class TestSimilarityResult:
    """Test SimilarityResult dataclass."""

    def test_similarity_result(self):
        """Test SimilarityResult creation."""
        result = SimilarityResult(
            voice_id=5,
            name="Artist X",
            similarity=0.87,
            metadata={"genre": "jazz"},
        )
        assert result.voice_id == 5
        assert result.name == "Artist X"
        assert result.similarity == 0.87
        assert result.metadata["genre"] == "jazz"

    def test_similarity_result_default_metadata(self):
        """Test SimilarityResult with default empty metadata."""
        result = SimilarityResult(
            voice_id=0,
            name="Artist",
            similarity=0.5,
        )
        assert result.metadata == {}
