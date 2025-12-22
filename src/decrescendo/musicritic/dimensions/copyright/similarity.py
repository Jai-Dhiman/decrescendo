"""Melody and rhythm similarity analysis for copyright detection.

This module provides:
- MelodyExtractor: Extract pitch contours from audio
- RhythmExtractor: Extract onset/beat patterns
- SimilarityMatcher: Compare extracted features for similarity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import librosa
import numpy as np
from scipy import signal
from scipy.spatial.distance import cosine

from .config import MelodySimilarityConfig, RhythmSimilarityConfig
from .exceptions import MelodySimilarityError, RhythmSimilarityError


@dataclass
class MelodyReport:
    """Report from melody extraction.

    Attributes:
        pitch_contour: Array of pitch values (Hz), NaN for unvoiced frames.
        voiced_mask: Boolean mask indicating voiced frames.
        pitch_confidence: Confidence values for each pitch estimate.
        duration: Audio duration in seconds.
        hop_time: Time between frames in seconds.
    """

    pitch_contour: np.ndarray
    voiced_mask: np.ndarray
    pitch_confidence: np.ndarray
    duration: float
    hop_time: float

    @property
    def voiced_ratio(self) -> float:
        """Ratio of voiced frames."""
        if len(self.voiced_mask) == 0:
            return 0.0
        return float(np.mean(self.voiced_mask))

    @property
    def mean_pitch(self) -> float:
        """Mean pitch of voiced frames in Hz."""
        if not np.any(self.voiced_mask):
            return 0.0
        return float(np.nanmean(self.pitch_contour[self.voiced_mask]))


@dataclass
class RhythmReport:
    """Report from rhythm extraction.

    Attributes:
        onset_times: Array of onset times in seconds.
        onset_strengths: Strength of each onset.
        tempo: Estimated tempo in BPM.
        beat_times: Array of beat times in seconds.
        duration: Audio duration in seconds.
    """

    onset_times: np.ndarray
    onset_strengths: np.ndarray
    tempo: float
    beat_times: np.ndarray
    duration: float

    @property
    def onset_density(self) -> float:
        """Average onsets per second."""
        if self.duration <= 0:
            return 0.0
        return len(self.onset_times) / self.duration


@dataclass
class SimilarityReport:
    """Report from similarity comparison.

    Attributes:
        melody_similarity: Melody contour similarity (0-1).
        rhythm_similarity: Rhythm pattern similarity (0-1).
        harmony_similarity: Harmonic content similarity (0-1).
        overall_similarity: Combined similarity score (0-1).
        matched_sections: List of (start, end) tuples of similar sections.
        metadata: Additional analysis metadata.
    """

    melody_similarity: float
    rhythm_similarity: float
    harmony_similarity: float
    overall_similarity: float
    matched_sections: list[tuple[float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MelodyExtractor:
    """Extract melody (pitch contour) from audio.

    Uses librosa's pyin algorithm for robust pitch tracking.

    Example:
        >>> extractor = MelodyExtractor()
        >>> report = extractor.extract(audio, sample_rate=44100)
        >>> print(f"Mean pitch: {report.mean_pitch:.1f} Hz")
    """

    def __init__(self, config: MelodySimilarityConfig | None = None) -> None:
        """Initialize the extractor.

        Args:
            config: Melody extraction configuration.
        """
        self.config = config or MelodySimilarityConfig()

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> MelodyReport:
        """Extract pitch contour from audio.

        Args:
            audio: Audio waveform (mono, float32).
            sample_rate: Sample rate of the audio.

        Returns:
            MelodyReport with extracted pitch information.

        Raises:
            MelodySimilarityError: If extraction fails.
        """
        try:
            # Resample if needed
            if sample_rate != self.config.target_sample_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=self.config.target_sample_rate,
                )
                sample_rate = self.config.target_sample_rate

            duration = len(audio) / sample_rate

            # Extract pitch using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
                sr=sample_rate,
                hop_length=self.config.hop_length,
                frame_length=self.config.frame_length,
            )

            hop_time = self.config.hop_length / sample_rate

            return MelodyReport(
                pitch_contour=f0,
                voiced_mask=voiced_flag,
                pitch_confidence=voiced_probs,
                duration=duration,
                hop_time=hop_time,
            )

        except Exception as e:
            raise MelodySimilarityError(f"Melody extraction failed: {e}") from e

    def compute_pitch_histogram(
        self,
        report: MelodyReport,
        bins: int = 12,
    ) -> np.ndarray:
        """Compute a pitch class histogram from the melody.

        Args:
            report: MelodyReport from extract().
            bins: Number of pitch classes (12 for chromatic).

        Returns:
            Normalized histogram of pitch classes.
        """
        voiced_pitches = report.pitch_contour[report.voiced_mask]
        if len(voiced_pitches) == 0:
            return np.zeros(bins)

        # Convert Hz to MIDI note numbers, then to pitch class
        midi_notes = librosa.hz_to_midi(voiced_pitches)
        pitch_classes = np.mod(midi_notes, 12)

        # Create histogram
        hist, _ = np.histogram(pitch_classes, bins=bins, range=(0, 12))
        hist = hist.astype(np.float32)

        # Normalize
        total = hist.sum()
        if total > 0:
            hist /= total

        return hist


class RhythmExtractor:
    """Extract rhythm patterns from audio.

    Uses librosa for onset detection and beat tracking.

    Example:
        >>> extractor = RhythmExtractor()
        >>> report = extractor.extract(audio, sample_rate=44100)
        >>> print(f"Tempo: {report.tempo:.1f} BPM")
    """

    def __init__(self, config: RhythmSimilarityConfig | None = None) -> None:
        """Initialize the extractor.

        Args:
            config: Rhythm extraction configuration.
        """
        self.config = config or RhythmSimilarityConfig()

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> RhythmReport:
        """Extract rhythm information from audio.

        Args:
            audio: Audio waveform (mono, float32).
            sample_rate: Sample rate of the audio.

        Returns:
            RhythmReport with extracted rhythm information.

        Raises:
            RhythmSimilarityError: If extraction fails.
        """
        try:
            # Resample if needed
            if sample_rate != self.config.target_sample_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=self.config.target_sample_rate,
                )
                sample_rate = self.config.target_sample_rate

            duration = len(audio) / sample_rate

            # Compute onset strength envelope
            onset_env = librosa.onset.onset_strength(
                y=audio,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            # Detect onsets
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )
            onset_times = librosa.frames_to_time(
                onset_frames,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            # Get onset strengths at detected positions
            onset_strengths = onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])

            # Estimate tempo and beat positions
            tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )
            beat_times = librosa.frames_to_time(
                beat_frames,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            # Handle tempo which may be an array in newer librosa versions
            tempo_value = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)

            return RhythmReport(
                onset_times=onset_times,
                onset_strengths=onset_strengths,
                tempo=tempo_value,
                beat_times=beat_times,
                duration=duration,
            )

        except Exception as e:
            raise RhythmSimilarityError(f"Rhythm extraction failed: {e}") from e

    def compute_onset_histogram(
        self,
        report: RhythmReport,
        bins: int = 16,
    ) -> np.ndarray:
        """Compute a histogram of onset intervals.

        Args:
            report: RhythmReport from extract().
            bins: Number of histogram bins.

        Returns:
            Normalized histogram of inter-onset intervals.
        """
        if len(report.onset_times) < 2:
            return np.zeros(bins)

        # Compute inter-onset intervals
        ioi = np.diff(report.onset_times)

        # Create histogram (0 to 1 second range)
        hist, _ = np.histogram(ioi, bins=bins, range=(0, 1))
        hist = hist.astype(np.float32)

        # Normalize
        total = hist.sum()
        if total > 0:
            hist /= total

        return hist


class SimilarityMatcher:
    """Compare audio features for similarity detection.

    Combines melody, rhythm, and harmony similarity into an overall score.

    Example:
        >>> matcher = SimilarityMatcher()
        >>> report = matcher.compare(audio1, audio2, sample_rate=44100)
        >>> print(f"Overall similarity: {report.overall_similarity:.2%}")
    """

    def __init__(
        self,
        melody_config: MelodySimilarityConfig | None = None,
        rhythm_config: RhythmSimilarityConfig | None = None,
        melody_weight: float = 0.5,
        rhythm_weight: float = 0.3,
        harmony_weight: float = 0.2,
    ) -> None:
        """Initialize the matcher.

        Args:
            melody_config: Configuration for melody extraction.
            rhythm_config: Configuration for rhythm extraction.
            melody_weight: Weight for melody similarity (0-1).
            rhythm_weight: Weight for rhythm similarity (0-1).
            harmony_weight: Weight for harmony similarity (0-1).
        """
        self.melody_extractor = MelodyExtractor(melody_config)
        self.rhythm_extractor = RhythmExtractor(rhythm_config)
        self.melody_weight = melody_weight
        self.rhythm_weight = rhythm_weight
        self.harmony_weight = harmony_weight

    def compare(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        sample_rate: int,
    ) -> SimilarityReport:
        """Compare two audio samples for similarity.

        Args:
            audio1: First audio waveform (mono, float32).
            audio2: Second audio waveform (mono, float32).
            sample_rate: Sample rate of both audio samples.

        Returns:
            SimilarityReport with detailed comparison results.
        """
        # Extract features from both samples
        melody1 = self.melody_extractor.extract(audio1, sample_rate)
        melody2 = self.melody_extractor.extract(audio2, sample_rate)

        rhythm1 = self.rhythm_extractor.extract(audio1, sample_rate)
        rhythm2 = self.rhythm_extractor.extract(audio2, sample_rate)

        # Compute melody similarity using pitch histograms
        melody_sim = self._compare_melody(melody1, melody2)

        # Compute rhythm similarity using onset patterns
        rhythm_sim = self._compare_rhythm(rhythm1, rhythm2)

        # Compute harmony similarity using chroma
        harmony_sim = self._compare_harmony(audio1, audio2, sample_rate)

        # Weighted combination
        overall = (
            self.melody_weight * melody_sim
            + self.rhythm_weight * rhythm_sim
            + self.harmony_weight * harmony_sim
        )

        return SimilarityReport(
            melody_similarity=melody_sim,
            rhythm_similarity=rhythm_sim,
            harmony_similarity=harmony_sim,
            overall_similarity=overall,
            metadata={
                "melody1_voiced_ratio": melody1.voiced_ratio,
                "melody2_voiced_ratio": melody2.voiced_ratio,
                "tempo1": rhythm1.tempo,
                "tempo2": rhythm2.tempo,
            },
        )

    def _compare_melody(
        self,
        melody1: MelodyReport,
        melody2: MelodyReport,
    ) -> float:
        """Compare two melody reports for similarity.

        Args:
            melody1: First melody report.
            melody2: Second melody report.

        Returns:
            Similarity score (0-1).
        """
        # Compute pitch class histograms
        hist1 = self.melody_extractor.compute_pitch_histogram(melody1)
        hist2 = self.melody_extractor.compute_pitch_histogram(melody2)

        # Handle empty histograms
        if hist1.sum() == 0 or hist2.sum() == 0:
            return 0.0

        # Cosine similarity
        similarity = 1.0 - cosine(hist1, hist2)
        return max(0.0, min(1.0, similarity))

    def _compare_rhythm(
        self,
        rhythm1: RhythmReport,
        rhythm2: RhythmReport,
    ) -> float:
        """Compare two rhythm reports for similarity.

        Args:
            rhythm1: First rhythm report.
            rhythm2: Second rhythm report.

        Returns:
            Similarity score (0-1).
        """
        # Compute onset interval histograms
        hist1 = self.rhythm_extractor.compute_onset_histogram(rhythm1)
        hist2 = self.rhythm_extractor.compute_onset_histogram(rhythm2)

        # Handle empty histograms
        if hist1.sum() == 0 or hist2.sum() == 0:
            return 0.0

        # Cosine similarity for rhythm patterns
        rhythm_pattern_sim = 1.0 - cosine(hist1, hist2)

        # Tempo similarity (within 10% is considered similar)
        if rhythm1.tempo > 0 and rhythm2.tempo > 0:
            tempo_ratio = min(rhythm1.tempo, rhythm2.tempo) / max(rhythm1.tempo, rhythm2.tempo)
            # Also check for double/half tempo
            double_ratio = min(rhythm1.tempo, rhythm2.tempo * 2) / max(rhythm1.tempo, rhythm2.tempo * 2)
            half_ratio = min(rhythm1.tempo * 2, rhythm2.tempo) / max(rhythm1.tempo * 2, rhythm2.tempo)
            tempo_sim = max(tempo_ratio, double_ratio, half_ratio)
        else:
            tempo_sim = 0.0

        # Combine pattern and tempo similarity
        similarity = 0.7 * rhythm_pattern_sim + 0.3 * tempo_sim
        return max(0.0, min(1.0, similarity))

    def _compare_harmony(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Compare harmonic content of two audio samples.

        Args:
            audio1: First audio waveform.
            audio2: Second audio waveform.
            sample_rate: Sample rate.

        Returns:
            Similarity score (0-1).
        """
        # Compute chroma features
        chroma1 = librosa.feature.chroma_cqt(y=audio1, sr=sample_rate)
        chroma2 = librosa.feature.chroma_cqt(y=audio2, sr=sample_rate)

        # Compute mean chroma (pitch class profile)
        profile1 = chroma1.mean(axis=1)
        profile2 = chroma2.mean(axis=1)

        # Normalize
        norm1 = np.linalg.norm(profile1)
        norm2 = np.linalg.norm(profile2)

        if norm1 > 0:
            profile1 /= norm1
        if norm2 > 0:
            profile2 /= norm2

        # Cosine similarity
        similarity = float(np.dot(profile1, profile2))
        return max(0.0, min(1.0, similarity))

    def compute_similarity_score(
        self,
        audio: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Compute overall similarity between query and reference audio.

        Convenience method that returns just the overall similarity score.

        Args:
            audio: Query audio waveform.
            reference: Reference audio waveform.
            sample_rate: Sample rate.

        Returns:
            Overall similarity score (0-1).
        """
        report = self.compare(audio, reference, sample_rate)
        return report.overall_similarity
