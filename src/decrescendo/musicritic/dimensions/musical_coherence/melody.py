"""Melody analysis for Musical Coherence dimension.

This module analyzes melodic content including pitch tracking,
phrase detection, and melodic contour analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from .config import MelodyConfig
from .exceptions import MelodyAnalysisError


@dataclass
class MelodyReport:
    """Report of melody analysis.

    Attributes:
        pitch_contour: Pitch values over time in Hz (0 for unvoiced).
        pitch_timestamps: Timestamps for pitch values in seconds.
        voiced_ratio: Ratio of voiced frames (0.0-1.0).
        pitch_range_hz: Range of pitches detected in Hz.
        pitch_mean_hz: Mean pitch in Hz.
        phrase_count: Number of melodic phrases detected.
        phrase_boundaries: Timestamps of phrase boundaries in seconds.
        contour_complexity: Melodic contour complexity (0.0-1.0).
            Higher values indicate more varied melodies.
        pitch_stability: How stable pitches are within phrases (0.0-1.0).
    """

    pitch_contour: np.ndarray = field(default_factory=lambda: np.array([]))
    pitch_timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    voiced_ratio: float = 0.0
    pitch_range_hz: float = 0.0
    pitch_mean_hz: float = 0.0
    phrase_count: int = 0
    phrase_boundaries: list[float] = field(default_factory=list)
    contour_complexity: float = 0.0
    pitch_stability: float = 0.0


class MelodyAnalyzer:
    """Analyzes melodic content using pitch tracking.

    Uses librosa pyin for robust pitch tracking. Analyzes melodic
    contour, phrase structure, and pitch coherence.

    Example:
        >>> analyzer = MelodyAnalyzer()
        >>> report = analyzer.analyze(audio, sample_rate=22050)
        >>> print(f"Voiced ratio: {report.voiced_ratio:.2%}")
        >>> print(f"Phrase count: {report.phrase_count}")
    """

    def __init__(self, config: MelodyConfig | None = None) -> None:
        """Initialize the melody analyzer.

        Args:
            config: Melody analysis configuration. Uses defaults if None.
        """
        self.config = config or MelodyConfig()

    def extract_pitch(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract pitch contour using pYIN algorithm.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (f0, voiced_flag, voiced_probs):
            - f0: Fundamental frequency in Hz (np.nan for unvoiced)
            - voiced_flag: Boolean array of voiced frames
            - voiced_probs: Probability of voicing

        Raises:
            MelodyAnalysisError: If pitch extraction fails.
        """
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
                sr=sample_rate,
                frame_length=self.config.frame_length,
                hop_length=self.config.hop_length,
            )

            return f0, voiced_flag, voiced_probs

        except Exception as e:
            raise MelodyAnalysisError(f"Pitch extraction failed: {e}") from e

    def compute_timestamps(
        self,
        n_frames: int,
        sample_rate: int,
    ) -> np.ndarray:
        """Compute timestamps for pitch frames.

        Args:
            n_frames: Number of frames.
            sample_rate: Sample rate in Hz.

        Returns:
            Array of timestamps in seconds.
        """
        return librosa.frames_to_time(
            np.arange(n_frames),
            sr=sample_rate,
            hop_length=self.config.hop_length,
        )

    def detect_phrase_boundaries(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[float]:
        """Detect phrase boundaries from pitch contour.

        Phrases are separated by:
        - Gaps in voicing (silence/unvoiced regions)
        - Large pitch jumps (> 1 octave)

        Args:
            f0: Fundamental frequency array.
            voiced_flag: Boolean array of voiced frames.
            timestamps: Timestamps for each frame.

        Returns:
            List of phrase boundary timestamps.
        """
        if len(f0) == 0 or not np.any(voiced_flag):
            return []

        boundaries = []
        min_gap_frames = 5  # Minimum frames of silence for phrase break

        # Find gaps in voicing
        unvoiced = ~voiced_flag
        diff = np.diff(unvoiced.astype(int))

        # Start of unvoiced region
        gap_starts = np.where(diff == 1)[0] + 1
        gap_ends = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if unvoiced[0]:
            gap_starts = np.insert(gap_starts, 0, 0)
        if unvoiced[-1]:
            gap_ends = np.append(gap_ends, len(unvoiced))

        for start, end in zip(gap_starts, gap_ends):
            gap_length = end - start
            if gap_length >= min_gap_frames and start < len(timestamps):
                boundaries.append(float(timestamps[start]))

        # Also detect large pitch jumps in voiced regions
        voiced_indices = np.where(voiced_flag)[0]
        if len(voiced_indices) >= 2:
            voiced_f0 = f0[voiced_indices]
            # Calculate semitone intervals
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.abs(np.diff(np.log2(voiced_f0)))
                # More than 1 octave = 1.0 in log2 terms
                large_jumps = np.where(ratios > 0.8)[0]

            for jump_idx in large_jumps:
                if jump_idx < len(voiced_indices) - 1:
                    frame_idx = voiced_indices[jump_idx + 1]
                    if frame_idx < len(timestamps):
                        boundary_time = float(timestamps[frame_idx])
                        if boundary_time not in boundaries:
                            boundaries.append(boundary_time)

        return sorted(boundaries)

    def compute_contour_complexity(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
    ) -> float:
        """Compute melodic contour complexity.

        Complexity is based on:
        - Interval variety (number of different intervals used)
        - Direction changes (how often melody changes direction)
        - Range usage (how much of the pitch range is used)

        Args:
            f0: Fundamental frequency array.
            voiced_flag: Boolean array of voiced frames.

        Returns:
            Complexity score from 0.0 to 1.0.
        """
        if not np.any(voiced_flag):
            return 0.0

        # Get voiced pitches only
        voiced_f0 = f0[voiced_flag]
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        if len(voiced_f0) < 3:
            return 0.0

        # Convert to semitones for interval analysis
        with np.errstate(divide="ignore", invalid="ignore"):
            semitones = 12 * np.log2(voiced_f0 / voiced_f0[0])

        semitones = semitones[~np.isnan(semitones)]

        if len(semitones) < 3:
            return 0.0

        # Feature 1: Interval variety
        intervals = np.diff(semitones)
        # Round to nearest semitone
        intervals_rounded = np.round(intervals)
        unique_intervals = len(np.unique(intervals_rounded))
        # Normalize: 12 unique intervals = high variety
        interval_variety = min(1.0, unique_intervals / 12.0)

        # Feature 2: Direction changes
        directions = np.sign(intervals)
        direction_changes = np.sum(np.abs(np.diff(directions)) > 0)
        # Normalize by length
        direction_change_rate = direction_changes / (len(directions) - 1 + 1e-10)

        # Feature 3: Range usage
        pitch_range = np.ptp(semitones)  # Peak-to-peak (range)
        # Normalize: 24 semitones (2 octaves) = full range
        range_score = min(1.0, pitch_range / 24.0)

        # Combine features
        complexity = 0.4 * interval_variety + 0.3 * direction_change_rate + 0.3 * range_score

        return float(np.clip(complexity, 0.0, 1.0))

    def compute_pitch_stability(
        self,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
    ) -> float:
        """Compute pitch stability within voiced regions.

        Measures how stable/consistent pitches are, which indicates
        intentional melodic content vs noise.

        Args:
            f0: Fundamental frequency array.
            voiced_flag: Boolean array of voiced frames.

        Returns:
            Stability score from 0.0 to 1.0.
        """
        if not np.any(voiced_flag):
            return 0.0

        voiced_f0 = f0[voiced_flag]
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        if len(voiced_f0) < 2:
            return 0.0

        # Convert to semitones
        with np.errstate(divide="ignore", invalid="ignore"):
            semitones = 12 * np.log2(voiced_f0 / voiced_f0[0])

        semitones = semitones[~np.isnan(semitones)]

        if len(semitones) < 2:
            return 0.0

        # Compute frame-to-frame variation
        frame_diff = np.abs(np.diff(semitones))

        # Stable melody: small frame-to-frame changes
        # Average change should be < 0.5 semitones for stable
        mean_change = np.mean(frame_diff)

        # Convert to stability score
        # mean_change of 0 = perfect stability (1.0)
        # mean_change of 1 = moderate (0.5)
        # mean_change of 2+ = poor (0.0)
        stability = max(0.0, 1.0 - mean_change / 2.0)

        return float(stability)

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> MelodyReport:
        """Run complete melody analysis.

        Args:
            audio: Audio samples (mono, float32, normalized).
            sample_rate: Sample rate in Hz.

        Returns:
            MelodyReport with all melody analysis results.

        Raises:
            MelodyAnalysisError: If analysis fails.
        """
        # Extract pitch
        f0, voiced_flag, voiced_probs = self.extract_pitch(audio, sample_rate)

        # Compute timestamps
        timestamps = self.compute_timestamps(len(f0), sample_rate)

        # Calculate voiced ratio
        voiced_ratio = np.mean(voiced_flag) if len(voiced_flag) > 0 else 0.0

        # Get voiced pitches for statistics
        voiced_f0 = f0[voiced_flag] if np.any(voiced_flag) else np.array([])
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        # Calculate pitch range and mean
        if len(voiced_f0) > 0:
            pitch_range_hz = float(np.ptp(voiced_f0))
            pitch_mean_hz = float(np.mean(voiced_f0))
        else:
            pitch_range_hz = 0.0
            pitch_mean_hz = 0.0

        # Detect phrase boundaries
        phrase_boundaries = self.detect_phrase_boundaries(f0, voiced_flag, timestamps)
        phrase_count = len(phrase_boundaries) + 1 if voiced_ratio > 0 else 0

        # Compute contour complexity
        contour_complexity = self.compute_contour_complexity(f0, voiced_flag)

        # Compute pitch stability
        pitch_stability = self.compute_pitch_stability(f0, voiced_flag)

        # Replace NaN with 0 for output
        pitch_contour = np.nan_to_num(f0, nan=0.0)

        return MelodyReport(
            pitch_contour=pitch_contour,
            pitch_timestamps=timestamps,
            voiced_ratio=float(voiced_ratio),
            pitch_range_hz=pitch_range_hz,
            pitch_mean_hz=pitch_mean_hz,
            phrase_count=phrase_count,
            phrase_boundaries=phrase_boundaries,
            contour_complexity=contour_complexity,
            pitch_stability=pitch_stability,
        )

    def compute_score(
        self,
        report: MelodyReport,
        audio_duration: float = 30.0,
    ) -> float:
        """Convert melody report to quality score.

        Score is based on:
        - Voiced ratio (35% weight) - should have melodic content
        - Contour complexity (30% weight) - interesting melodies
        - Pitch stability (35% weight) - coherent, not noisy

        Args:
            report: MelodyReport from analyze().
            audio_duration: Audio duration in seconds.

        Returns:
            Quality score from 0.0 to 1.0 (higher = better melody).
        """
        # Voiced ratio score
        # Too little voiced content = no melody
        # Too much (>0.8) is fine
        if report.voiced_ratio < self.config.min_voiced_ratio:
            voiced_score = 0.0
        else:
            # Normalize: 0.3 voiced = good (1.0)
            voiced_score = min(1.0, report.voiced_ratio / 0.3)

        # Contour complexity score
        # Some complexity is good, but not required
        # Score 0-1 directly
        complexity_score = report.contour_complexity

        # Pitch stability score
        # Stability is important for coherent melody
        stability_score = report.pitch_stability

        # Weighted combination
        score = 0.35 * voiced_score + 0.30 * complexity_score + 0.35 * stability_score

        return float(np.clip(score, 0.0, 1.0))
