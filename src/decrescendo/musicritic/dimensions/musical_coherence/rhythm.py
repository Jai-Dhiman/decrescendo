"""Rhythm analysis for Musical Coherence dimension.

This module analyzes rhythmic content including beat tracking,
tempo detection, and tempo stability measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from .config import RhythmConfig
from .exceptions import DependencyNotAvailableError, RhythmAnalysisError


@dataclass
class RhythmReport:
    """Report of rhythm analysis.

    Attributes:
        tempo_bpm: Detected tempo in beats per minute.
        tempo_confidence: Confidence in tempo detection (0.0-1.0).
        beat_timestamps: Timestamps of detected beats in seconds.
        beat_count: Number of detected beats.
        tempo_stability: How stable the tempo is throughout (0.0-1.0).
            Higher values indicate more consistent tempo.
        beat_strength: Average beat strength/prominence (0.0-1.0).
        downbeat_timestamps: Timestamps of detected downbeats in seconds.
    """

    tempo_bpm: float = 0.0
    tempo_confidence: float = 0.0
    beat_timestamps: list[float] = field(default_factory=list)
    beat_count: int = 0
    tempo_stability: float = 0.0
    beat_strength: float = 0.0
    downbeat_timestamps: list[float] = field(default_factory=list)


class RhythmAnalyzer:
    """Analyzes rhythmic content including beat tracking and tempo.

    Prefers madmom DBNBeatTracker for high-accuracy beat tracking,
    but uses librosa as the default (madmom is optional).

    Example:
        >>> analyzer = RhythmAnalyzer()
        >>> report = analyzer.analyze(audio, sample_rate=22050)
        >>> print(f"Tempo: {report.tempo_bpm:.1f} BPM")
        >>> print(f"Tempo stability: {report.tempo_stability:.2%}")
    """

    def __init__(self, config: RhythmConfig | None = None) -> None:
        """Initialize the rhythm analyzer.

        Args:
            config: Rhythm analysis configuration. Uses defaults if None.
        """
        self.config = config or RhythmConfig()
        self._madmom_available: bool | None = None

    @property
    def madmom_available(self) -> bool:
        """Check if madmom is available."""
        if self._madmom_available is None:
            try:
                import madmom  # noqa: F401

                self._madmom_available = True
            except ImportError:
                self._madmom_available = False
        return self._madmom_available

    def detect_beats_librosa(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, float]:
        """Detect beats using librosa beat tracking.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (beat_times, tempo).

        Raises:
            RhythmAnalysisError: If beat detection fails.
        """
        try:
            # Detect tempo and beats
            tempo, beat_frames = librosa.beat.beat_track(
                y=audio,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            # Handle tempo as array or scalar
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
            else:
                tempo = float(tempo)

            # Convert frames to times
            beat_times = librosa.frames_to_time(
                beat_frames,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            return beat_times, tempo

        except Exception as e:
            raise RhythmAnalysisError(f"Librosa beat detection failed: {e}") from e

    def detect_beats_madmom(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, float]:
        """Detect beats using madmom DBNBeatTracker.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (beat_times, tempo).

        Raises:
            DependencyNotAvailableError: If madmom is not installed.
            RhythmAnalysisError: If beat detection fails.
        """
        if not self.madmom_available:
            raise DependencyNotAvailableError("madmom", "beat tracking with DBNBeatTracker")

        try:
            from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

            # Process audio
            proc = RNNBeatProcessor()
            activations = proc(audio)

            # Track beats
            beat_proc = DBNBeatTrackingProcessor(
                min_bpm=self.config.min_tempo,
                max_bpm=self.config.max_tempo,
                fps=100,
            )
            beat_times = beat_proc(activations)

            # Calculate tempo from beat intervals
            if len(beat_times) >= 2:
                intervals = np.diff(beat_times)
                median_interval = np.median(intervals)
                tempo = 60.0 / median_interval if median_interval > 0 else 0.0
            else:
                tempo = 0.0

            return beat_times, tempo

        except Exception as e:
            raise RhythmAnalysisError(f"Madmom beat detection failed: {e}") from e

    def detect_beats(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, float]:
        """Detect beats using configured method.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (beat_times, tempo).

        Raises:
            DependencyNotAvailableError: If madmom is requested but unavailable.
            RhythmAnalysisError: If beat detection fails.
        """
        if self.config.use_madmom:
            if not self.madmom_available:
                raise DependencyNotAvailableError("madmom", "beat tracking with DBNBeatTracker")
            return self.detect_beats_madmom(audio, sample_rate)
        else:
            return self.detect_beats_librosa(audio, sample_rate)

    def compute_tempo_stability(
        self,
        beat_timestamps: np.ndarray,
    ) -> float:
        """Compute tempo stability from beat intervals.

        Measures how consistent the inter-beat intervals are.
        A perfectly steady tempo would have zero variance.

        Args:
            beat_timestamps: Array of beat times in seconds.

        Returns:
            Stability score from 0.0 to 1.0.
            Higher values indicate more consistent tempo.
        """
        if len(beat_timestamps) < 3:
            return 0.0

        # Compute inter-beat intervals
        intervals = np.diff(beat_timestamps)

        if len(intervals) == 0:
            return 0.0

        # Use coefficient of variation (CV) as stability measure
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if mean_interval == 0:
            return 0.0

        cv = std_interval / mean_interval

        # Convert CV to stability score
        # CV of 0 = perfect stability (1.0)
        # CV of 0.2 = moderate instability (0.5)
        # CV of 0.4+ = poor stability (0.0)
        stability = max(0.0, 1.0 - cv * 2.5)

        return float(stability)

    def compute_beat_strength(
        self,
        audio: np.ndarray,
        sample_rate: int,
        beat_timestamps: np.ndarray,
    ) -> float:
        """Compute average beat strength/prominence.

        Measures how prominent/clear the detected beats are.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.
            beat_timestamps: Array of beat times in seconds.

        Returns:
            Beat strength score from 0.0 to 1.0.
            Higher values indicate clearer beats.
        """
        if len(beat_timestamps) == 0:
            return 0.0

        try:
            # Compute onset envelope
            onset_env = librosa.onset.onset_strength(
                y=audio,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            if len(onset_env) == 0:
                return 0.0

            # Normalize onset envelope
            max_onset = onset_env.max()
            if max_onset > 0:
                onset_env = onset_env / max_onset

            # Get onset values at beat positions
            beat_frames = librosa.time_to_frames(
                beat_timestamps,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            # Filter valid frame indices
            valid_frames = beat_frames[(beat_frames >= 0) & (beat_frames < len(onset_env))]

            if len(valid_frames) == 0:
                return 0.0

            beat_onset_values = onset_env[valid_frames]

            # Beat strength is average onset value at beats
            # vs average onset value overall
            mean_at_beats = np.mean(beat_onset_values)
            mean_overall = np.mean(onset_env)

            if mean_overall == 0:
                return 0.0

            # Ratio: beats should be stronger than average
            ratio = mean_at_beats / mean_overall

            # Convert to 0-1 score (ratio of 2.0+ = perfect)
            strength = min(1.0, (ratio - 1.0) / 1.0)
            strength = max(0.0, strength)

            return float(strength)

        except Exception:
            return 0.0

    def estimate_tempo_confidence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        detected_tempo: float,
    ) -> float:
        """Estimate confidence in the detected tempo.

        Uses tempogram analysis to verify the detected tempo.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.
            detected_tempo: The tempo that was detected.

        Returns:
            Confidence score from 0.0 to 1.0.
        """
        if detected_tempo <= 0:
            return 0.0

        try:
            # Compute tempogram
            onset_env = librosa.onset.onset_strength(
                y=audio,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            tempogram = librosa.feature.tempogram(
                onset_envelope=onset_env,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            # Get tempo axis
            tempo_axis = librosa.tempo_frequencies(
                tempogram.shape[0],
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            # Find the closest tempo bin to detected tempo
            tempo_idx = np.argmin(np.abs(tempo_axis - detected_tempo))

            # Get the energy at that tempo
            tempo_energy = np.mean(tempogram[tempo_idx, :])

            # Normalize by total energy
            total_energy = np.mean(tempogram)

            if total_energy == 0:
                return 0.0

            confidence = min(1.0, tempo_energy / (total_energy * 2))

            return float(confidence)

        except Exception:
            return 0.5  # Default moderate confidence if analysis fails

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> RhythmReport:
        """Run complete rhythm analysis.

        Args:
            audio: Audio samples (mono, float32, normalized).
            sample_rate: Sample rate in Hz.

        Returns:
            RhythmReport with all rhythm analysis results.

        Raises:
            RhythmAnalysisError: If analysis fails.
        """
        # Detect beats
        beat_times, tempo = self.detect_beats(audio, sample_rate)

        # Validate tempo range
        if tempo < self.config.min_tempo or tempo > self.config.max_tempo:
            # Tempo outside expected range, reduce confidence
            tempo_confidence = 0.3
        else:
            tempo_confidence = self.estimate_tempo_confidence(audio, sample_rate, tempo)

        # Compute stability
        tempo_stability = self.compute_tempo_stability(beat_times)

        # Compute beat strength
        beat_strength = self.compute_beat_strength(audio, sample_rate, beat_times)

        return RhythmReport(
            tempo_bpm=tempo,
            tempo_confidence=tempo_confidence,
            beat_timestamps=beat_times.tolist() if len(beat_times) > 0 else [],
            beat_count=len(beat_times),
            tempo_stability=tempo_stability,
            beat_strength=beat_strength,
            downbeat_timestamps=[],  # Downbeat detection not implemented yet
        )

    def compute_score(self, report: RhythmReport) -> float:
        """Convert rhythm report to quality score.

        Score is based on:
        - Tempo stability (45% weight)
        - Beat strength (30% weight)
        - Tempo validity (25% weight)

        Args:
            report: RhythmReport from analyze().

        Returns:
            Quality score from 0.0 to 1.0 (higher = better rhythm).
        """
        # Tempo validity: is tempo in reasonable range?
        if self.config.min_tempo <= report.tempo_bpm <= self.config.max_tempo:
            tempo_validity = 1.0
        elif report.tempo_bpm <= 0:
            tempo_validity = 0.0
        else:
            # Outside range but detectable
            tempo_validity = 0.5

        # Weighted combination
        score = (
            0.45 * report.tempo_stability
            + 0.30 * report.beat_strength
            + 0.25 * tempo_validity * report.tempo_confidence
        )

        return float(np.clip(score, 0.0, 1.0))
