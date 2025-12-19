"""Voice enrollment pipeline for protected voices.

Provides a VoiceEnroller class for extracting speaker embeddings from audio
files and enrolling them into a VoiceDatabase with quality checks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .audio_preprocessing import AudioLoadError, AudioPreprocessor
from .config import OutputClassifierConfig, PreprocessingConfig
from .inference import OutputClassifierInference, initialize_output_classifier
from .model import OutputClassifierModel
from .voice_database import SimilarityResult, VoiceDatabase, VoiceDuplicateError


class EnrollmentError(Exception):
    """Base exception for enrollment errors."""

    pass


class AudioQualityError(EnrollmentError):
    """Raised when audio fails quality checks."""

    pass


class DuplicateVoiceError(EnrollmentError):
    """Raised when attempting to enroll a duplicate voice."""

    pass


@dataclass
class QualityCheckResult:
    """Result of audio quality checks.

    Attributes:
        passed: Whether all quality checks passed
        duration_sec: Audio duration in seconds
        rms_db: RMS level in decibels
        snr_db: Estimated signal-to-noise ratio (if computable)
        issues: List of quality issues found
    """

    passed: bool
    duration_sec: float
    rms_db: float
    snr_db: float | None
    issues: list[str]


@dataclass
class EnrollmentResult:
    """Result of voice enrollment.

    Attributes:
        success: Whether enrollment succeeded
        voice_id: Assigned voice ID (if successful)
        name: Voice name
        embedding_dim: Dimension of the embedding
        num_samples_used: Number of audio samples used for embedding
        quality_results: Quality check results for each file
        duplicate_match: If a duplicate was detected, the similar voice info
        error: Error message if enrollment failed
    """

    success: bool
    voice_id: int | None
    name: str
    embedding_dim: int
    num_samples_used: int
    quality_results: list[QualityCheckResult]
    duplicate_match: SimilarityResult | None = None
    error: str | None = None


class VoiceEnroller:
    """Pipeline for enrolling protected voices.

    Extracts speaker embeddings from audio files, performs quality checks,
    and enrolls voices into a VoiceDatabase.

    Example:
        >>> # Initialize with existing model
        >>> enroller = VoiceEnroller(model, variables, config)
        >>> db = VoiceDatabase()
        >>>
        >>> # Enroll from a single file
        >>> result = enroller.enroll_from_file(
        ...     db, "artist_name", "voice_sample.wav"
        ... )
        >>>
        >>> # Enroll from multiple samples (recommended for better embedding)
        >>> result = enroller.enroll_from_files(
        ...     db, "artist_name", ["sample1.wav", "sample2.wav", "sample3.wav"]
        ... )
    """

    def __init__(
        self,
        model: OutputClassifierModel,
        variables: dict[str, Any],
        config: OutputClassifierConfig | None = None,
        min_duration_sec: float = 3.0,
        max_duration_sec: float = 300.0,
        min_rms_db: float = -50.0,
        duplicate_threshold: float = 0.95,
    ) -> None:
        """Initialize voice enroller.

        Args:
            model: OutputClassifierModel instance
            variables: Model variables containing 'params' and 'batch_stats'
            config: Output classifier configuration
            min_duration_sec: Minimum audio duration in seconds
            max_duration_sec: Maximum audio duration in seconds
            min_rms_db: Minimum RMS level in dB (for detecting silence)
            duplicate_threshold: Similarity threshold for duplicate detection
        """
        self.model = model
        self.variables = variables
        self.config = config or OutputClassifierConfig()

        # Quality thresholds
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.min_rms_db = min_rms_db
        self.duplicate_threshold = duplicate_threshold

        # Preprocessor for loading audio
        self.preprocessor = AudioPreprocessor(self.config.preprocessing)

        # JIT compile the forward pass
        self._forward = jax.jit(self._forward_fn)

    def _forward_fn(self, audio: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """JIT-compiled forward pass."""
        return self.model.apply(
            self.variables,
            audio,
            train=False,
        )

    def check_audio_quality(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> QualityCheckResult:
        """Check audio quality for voice enrollment.

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            QualityCheckResult with quality metrics and issues
        """
        issues = []

        # Duration check
        duration_sec = len(audio) / sample_rate
        if duration_sec < self.min_duration_sec:
            issues.append(
                f"Audio too short: {duration_sec:.1f}s (min: {self.min_duration_sec}s)"
            )
        if duration_sec > self.max_duration_sec:
            issues.append(
                f"Audio too long: {duration_sec:.1f}s (max: {self.max_duration_sec}s)"
            )

        # RMS level check (detect silence/very quiet audio)
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms + 1e-10)
        if rms_db < self.min_rms_db:
            issues.append(
                f"Audio too quiet: {rms_db:.1f}dB (min: {self.min_rms_db}dB)"
            )

        # Simple SNR estimation (ratio of signal to noise floor)
        # This is a rough estimate using the top 10% vs bottom 10% of energy
        # Note: SNR is informational only and doesn't fail the quality check
        snr_db = None
        try:
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)  # 10ms hop

            # Compute frame energies
            num_frames = (len(audio) - frame_length) // hop_length + 1
            if num_frames > 10:
                energies = []
                for i in range(num_frames):
                    start = i * hop_length
                    frame = audio[start : start + frame_length]
                    energy = np.mean(frame**2)
                    energies.append(energy)

                energies = np.array(energies)
                sorted_energies = np.sort(energies)

                # Top 10% = signal, bottom 10% = noise floor
                n = len(sorted_energies)
                signal_energy = np.mean(sorted_energies[int(0.9 * n) :])
                noise_energy = np.mean(sorted_energies[: int(0.1 * n)])

                if noise_energy > 1e-12:
                    snr_db = 10 * np.log10(signal_energy / noise_energy)
                    # SNR is informational - we don't fail quality check for low SNR
                    # as it may be a valid clean audio with consistent energy
        except Exception:
            pass  # SNR estimation failed, continue without it

        passed = len(issues) == 0

        return QualityCheckResult(
            passed=passed,
            duration_sec=duration_sec,
            rms_db=rms_db,
            snr_db=snr_db,
            issues=issues,
        )

    def extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Extract speaker embedding from audio.

        Processes audio in chunks and averages the speaker embeddings.

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            Speaker embedding (embedding_dim,)
        """
        # Preprocess audio
        audio = self.preprocessor.preprocess(audio, sample_rate)

        # Process in chunks and collect embeddings
        embeddings = []
        for chunk in self.preprocessor.chunk_audio(audio):
            chunk_jax = jnp.array(chunk, dtype=jnp.float32)

            # Add batch dimension
            chunk_batch = chunk_jax[None, :]

            # Extract speaker embedding
            outputs = self._forward(chunk_batch)
            speaker_embedding = outputs["speaker_embeddings"][0]

            embeddings.append(np.array(speaker_embedding))

        if len(embeddings) == 0:
            raise EnrollmentError("No valid audio chunks to process")

        # Average embeddings across chunks
        avg_embedding = np.mean(embeddings, axis=0)

        # L2 normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        return avg_embedding

    def extract_embedding_from_file(
        self,
        path: Path | str,
    ) -> tuple[np.ndarray, QualityCheckResult]:
        """Extract speaker embedding from an audio file.

        Args:
            path: Path to audio file

        Returns:
            Tuple of (embedding, quality_result)

        Raises:
            AudioLoadError: If file cannot be loaded
            AudioQualityError: If audio fails quality checks
        """
        # Load audio
        audio, sr = self.preprocessor.load_audio(path)

        # Check quality
        quality = self.check_audio_quality(audio, sr)

        if not quality.passed:
            raise AudioQualityError(
                f"Audio quality check failed for {path}: {quality.issues}"
            )

        # Extract embedding
        embedding = self.extract_embedding(audio, sr)

        return embedding, quality

    def enroll_from_file(
        self,
        database: VoiceDatabase,
        name: str,
        path: Path | str,
        metadata: dict[str, Any] | None = None,
        skip_quality_check: bool = False,
        skip_duplicate_check: bool = False,
    ) -> EnrollmentResult:
        """Enroll a voice from a single audio file.

        Args:
            database: VoiceDatabase to enroll into
            name: Name for the voice (e.g., artist name)
            path: Path to audio file
            metadata: Optional metadata to store
            skip_quality_check: Skip audio quality checks
            skip_duplicate_check: Skip duplicate detection

        Returns:
            EnrollmentResult with enrollment status
        """
        return self.enroll_from_files(
            database=database,
            name=name,
            paths=[path],
            metadata=metadata,
            skip_quality_check=skip_quality_check,
            skip_duplicate_check=skip_duplicate_check,
        )

    def enroll_from_files(
        self,
        database: VoiceDatabase,
        name: str,
        paths: list[Path | str],
        metadata: dict[str, Any] | None = None,
        skip_quality_check: bool = False,
        skip_duplicate_check: bool = False,
    ) -> EnrollmentResult:
        """Enroll a voice from multiple audio files.

        Extracts embeddings from all files and averages them for a more
        robust voice representation.

        Args:
            database: VoiceDatabase to enroll into
            name: Name for the voice (e.g., artist name)
            paths: List of paths to audio files
            metadata: Optional metadata to store
            skip_quality_check: Skip audio quality checks
            skip_duplicate_check: Skip duplicate detection

        Returns:
            EnrollmentResult with enrollment status
        """
        if len(paths) == 0:
            return EnrollmentResult(
                success=False,
                voice_id=None,
                name=name,
                embedding_dim=database.embedding_dim,
                num_samples_used=0,
                quality_results=[],
                error="No audio files provided",
            )

        embeddings = []
        quality_results = []
        failed_files = []

        for path in paths:
            try:
                # Load audio
                audio, sr = self.preprocessor.load_audio(path)

                # Check quality
                quality = self.check_audio_quality(audio, sr)
                quality_results.append(quality)

                if not skip_quality_check and not quality.passed:
                    failed_files.append((path, quality.issues))
                    continue

                # Extract embedding
                embedding = self.extract_embedding(audio, sr)
                embeddings.append(embedding)

            except AudioLoadError as e:
                failed_files.append((path, [str(e)]))
                quality_results.append(
                    QualityCheckResult(
                        passed=False,
                        duration_sec=0,
                        rms_db=-100,
                        snr_db=None,
                        issues=[str(e)],
                    )
                )

        if len(embeddings) == 0:
            error_details = "; ".join(
                f"{p}: {issues}" for p, issues in failed_files
            )
            return EnrollmentResult(
                success=False,
                voice_id=None,
                name=name,
                embedding_dim=database.embedding_dim,
                num_samples_used=0,
                quality_results=quality_results,
                error=f"No valid audio files: {error_details}",
            )

        # Average embeddings across all files
        final_embedding = np.mean(embeddings, axis=0)

        # L2 normalize
        norm = np.linalg.norm(final_embedding)
        if norm > 0:
            final_embedding = final_embedding / norm

        # Check for duplicates
        duplicate_match = None
        if not skip_duplicate_check and len(database) > 0:
            duplicate_match = database.check_duplicate(
                final_embedding, threshold=self.duplicate_threshold
            )
            if duplicate_match is not None:
                return EnrollmentResult(
                    success=False,
                    voice_id=None,
                    name=name,
                    embedding_dim=database.embedding_dim,
                    num_samples_used=len(embeddings),
                    quality_results=quality_results,
                    duplicate_match=duplicate_match,
                    error=f"Duplicate voice detected: similar to '{duplicate_match.name}' "
                    f"(similarity: {duplicate_match.similarity:.3f})",
                )

        # Add to database
        try:
            voice_id = database.add_voice(
                name=name,
                embedding=final_embedding,
                metadata=metadata,
            )
        except VoiceDuplicateError as e:
            return EnrollmentResult(
                success=False,
                voice_id=None,
                name=name,
                embedding_dim=database.embedding_dim,
                num_samples_used=len(embeddings),
                quality_results=quality_results,
                error=str(e),
            )

        return EnrollmentResult(
            success=True,
            voice_id=voice_id,
            name=name,
            embedding_dim=final_embedding.shape[0],
            num_samples_used=len(embeddings),
            quality_results=quality_results,
        )

    def extract_embedding_from_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        check_quality: bool = True,
    ) -> tuple[np.ndarray, QualityCheckResult | None]:
        """Extract speaker embedding from audio array.

        Args:
            audio: Audio array
            sample_rate: Sample rate
            check_quality: Whether to perform quality checks

        Returns:
            Tuple of (embedding, quality_result or None)

        Raises:
            AudioQualityError: If audio fails quality checks (when check_quality=True)
        """
        quality = None

        if check_quality:
            quality = self.check_audio_quality(audio, sample_rate)
            if not quality.passed:
                raise AudioQualityError(
                    f"Audio quality check failed: {quality.issues}"
                )

        embedding = self.extract_embedding(audio, sample_rate)
        return embedding, quality


def create_voice_enroller(
    config: OutputClassifierConfig | None = None,
    rng: jax.Array | None = None,
) -> VoiceEnroller:
    """Create a VoiceEnroller with freshly initialized model.

    This is useful for creating an enroller without loading a checkpoint.
    Note that the embeddings will not be meaningful without a trained model.

    Args:
        config: Model configuration
        rng: JAX random key

    Returns:
        VoiceEnroller instance
    """
    model, variables = initialize_output_classifier(config, rng)
    return VoiceEnroller(model, variables, config)


def create_voice_enroller_from_inference(
    inference: OutputClassifierInference,
) -> VoiceEnroller:
    """Create a VoiceEnroller from an existing inference pipeline.

    Args:
        inference: OutputClassifierInference instance

    Returns:
        VoiceEnroller instance sharing the model and variables
    """
    return VoiceEnroller(
        model=inference.model,
        variables=inference.variables,
        config=inference.config,
    )
