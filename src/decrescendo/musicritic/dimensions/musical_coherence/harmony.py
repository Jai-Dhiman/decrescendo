"""Harmony analysis for Musical Coherence dimension.

This module analyzes harmonic content including key detection,
chord progression analysis, and harmonic consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from .config import HarmonyConfig
from .exceptions import DependencyNotAvailableError, HarmonyAnalysisError


# Key profiles for Krumhansl-Schmuckler algorithm
# Correlation weights for major and minor keys
KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Pitch class names
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class HarmonyReport:
    """Report of harmony analysis.

    Attributes:
        detected_key: Detected musical key (e.g., "C major").
        key_confidence: Confidence in key detection (0.0-1.0).
        chord_sequence: List of (start_time, end_time, chord_name) tuples.
        chord_count: Number of chords detected.
        unique_chord_count: Number of unique chords.
        key_consistency: How consistently the audio stays in key (0.0-1.0).
        progression_quality: Quality of chord progressions (0.0-1.0).
        chroma_features: Average chroma vector (12 values).
    """

    detected_key: str = ""
    key_confidence: float = 0.0
    chord_sequence: list[tuple[float, float, str]] = field(default_factory=list)
    chord_count: int = 0
    unique_chord_count: int = 0
    key_consistency: float = 0.0
    progression_quality: float = 0.0
    chroma_features: np.ndarray = field(default_factory=lambda: np.zeros(12))


class HarmonyAnalyzer:
    """Analyzes harmonic content including chords and key.

    Uses librosa for key detection and optionally Essentia for
    chord detection. Falls back to chroma-based chord estimation
    when Essentia is not available.

    Example:
        >>> analyzer = HarmonyAnalyzer()
        >>> report = analyzer.analyze(audio, sample_rate=22050)
        >>> print(f"Detected key: {report.detected_key}")
        >>> print(f"Key confidence: {report.key_confidence:.2%}")
    """

    def __init__(self, config: HarmonyConfig | None = None) -> None:
        """Initialize the harmony analyzer.

        Args:
            config: Harmony analysis configuration. Uses defaults if None.
        """
        self.config = config or HarmonyConfig()
        self._essentia_available: bool | None = None

    @property
    def essentia_available(self) -> bool:
        """Check if Essentia is available."""
        if self._essentia_available is None:
            try:
                import essentia.standard  # noqa: F401

                self._essentia_available = True
            except ImportError:
                self._essentia_available = False
        return self._essentia_available

    def compute_chroma(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Compute chroma features from audio.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Chroma features (12 x n_frames).

        Raises:
            HarmonyAnalysisError: If chroma computation fails.
        """
        try:
            chroma = librosa.feature.chroma_cqt(
                y=audio,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )
            return chroma

        except Exception as e:
            raise HarmonyAnalysisError(f"Chroma computation failed: {e}") from e

    def detect_key(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[str, float]:
        """Detect key using Krumhansl-Schmuckler algorithm.

        Uses chroma features and correlates with major/minor key profiles.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (key_name, confidence).

        Raises:
            HarmonyAnalysisError: If key detection fails.
        """
        try:
            # Compute chroma
            chroma = self.compute_chroma(audio, sample_rate)

            # Average chroma across time
            chroma_avg = np.mean(chroma, axis=1)

            # Normalize
            if np.max(chroma_avg) > 0:
                chroma_avg = chroma_avg / np.max(chroma_avg)

            best_key = ""
            best_correlation = -1.0
            correlations = []

            # Test all 24 keys (12 major + 12 minor)
            for shift in range(12):
                # Shift chroma to test each root note
                shifted = np.roll(chroma_avg, -shift)

                # Correlate with major profile
                major_corr = np.corrcoef(shifted, KRUMHANSL_MAJOR)[0, 1]
                correlations.append(major_corr)

                if major_corr > best_correlation:
                    best_correlation = major_corr
                    best_key = f"{PITCH_CLASSES[shift]} major"

                # Correlate with minor profile
                minor_corr = np.corrcoef(shifted, KRUMHANSL_MINOR)[0, 1]
                correlations.append(minor_corr)

                if minor_corr > best_correlation:
                    best_correlation = minor_corr
                    best_key = f"{PITCH_CLASSES[shift]} minor"

            # Compute confidence based on how much better the best is
            correlations = np.array(correlations)
            correlations = correlations[~np.isnan(correlations)]

            if len(correlations) > 1:
                sorted_corr = np.sort(correlations)[::-1]
                # Confidence based on margin over second best
                margin = sorted_corr[0] - sorted_corr[1]
                confidence = min(1.0, 0.5 + margin * 2)
            else:
                confidence = 0.5

            return best_key, float(confidence)

        except Exception as e:
            raise HarmonyAnalysisError(f"Key detection failed: {e}") from e

    def detect_chords_chroma(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[tuple[float, float, str]]:
        """Detect chords using chroma-based template matching.

        This is a simplified chord detection that matches chroma
        patterns to basic chord templates.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            List of (start_time, end_time, chord_name) tuples.

        Raises:
            HarmonyAnalysisError: If chord detection fails.
        """
        try:
            # Compute chroma
            chroma = self.compute_chroma(audio, sample_rate)

            # Define chord templates (root position major/minor)
            chord_templates = {}
            for i, root in enumerate(PITCH_CLASSES):
                # Major chord: root, major third, perfect fifth
                major = np.zeros(12)
                major[i] = 1.0
                major[(i + 4) % 12] = 1.0  # Major third
                major[(i + 7) % 12] = 1.0  # Perfect fifth
                chord_templates[f"{root}"] = major

                # Minor chord
                minor = np.zeros(12)
                minor[i] = 1.0
                minor[(i + 3) % 12] = 1.0  # Minor third
                minor[(i + 7) % 12] = 1.0  # Perfect fifth
                chord_templates[f"{root}m"] = minor

            # Segment chroma into chord-length windows
            frames_per_chord = max(
                1,
                int(
                    self.config.min_chord_duration
                    * sample_rate
                    / self.config.hop_length
                ),
            )

            chords = []
            n_frames = chroma.shape[1]

            for frame_start in range(0, n_frames, frames_per_chord):
                frame_end = min(frame_start + frames_per_chord, n_frames)

                # Average chroma in this window
                window_chroma = np.mean(chroma[:, frame_start:frame_end], axis=1)

                # Normalize
                if np.max(window_chroma) > 0:
                    window_chroma = window_chroma / np.max(window_chroma)

                # Find best matching chord
                best_chord = "N"  # No chord
                best_score = 0.3  # Minimum threshold

                for chord_name, template in chord_templates.items():
                    score = np.dot(window_chroma, template)
                    if score > best_score:
                        best_score = score
                        best_chord = chord_name

                # Convert frames to time
                start_time = librosa.frames_to_time(
                    frame_start, sr=sample_rate, hop_length=self.config.hop_length
                )
                end_time = librosa.frames_to_time(
                    frame_end, sr=sample_rate, hop_length=self.config.hop_length
                )

                chords.append((float(start_time), float(end_time), best_chord))

            # Merge consecutive same chords
            merged_chords = []
            for chord in chords:
                if merged_chords and merged_chords[-1][2] == chord[2]:
                    # Extend previous chord
                    merged_chords[-1] = (
                        merged_chords[-1][0],
                        chord[1],
                        chord[2],
                    )
                else:
                    merged_chords.append(chord)

            return merged_chords

        except Exception as e:
            raise HarmonyAnalysisError(f"Chord detection failed: {e}") from e

    def detect_chords_essentia(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[tuple[float, float, str]]:
        """Detect chords using Essentia ChordsDetection.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            List of (start_time, end_time, chord_name) tuples.

        Raises:
            DependencyNotAvailableError: If Essentia is not installed.
            HarmonyAnalysisError: If chord detection fails.
        """
        if not self.essentia_available:
            raise DependencyNotAvailableError(
                "essentia", "chord detection with Essentia"
            )

        try:
            import essentia.standard as es

            # Essentia expects specific sample rate
            if sample_rate != 44100:
                # Resample to 44100
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=44100
                )
                sample_rate = 44100

            # Compute HPCP (Harmonic Pitch Class Profile)
            frame_size = 4096
            hop_size = 2048

            spectrum_algo = es.Spectrum(size=frame_size)
            spectral_peaks = es.SpectralPeaks(
                sampleRate=sample_rate,
                maxPeaks=60,
                minFrequency=60,
                maxFrequency=5000,
            )
            hpcp_algo = es.HPCP(size=12, referenceFrequency=440)
            chords_algo = es.ChordsDetection(hopSize=hop_size, sampleRate=sample_rate)

            # Process frames
            hpcps = []
            for frame in es.FrameGenerator(
                audio, frameSize=frame_size, hopSize=hop_size
            ):
                spectrum = spectrum_algo(frame)
                freqs, mags = spectral_peaks(spectrum)
                hpcp = hpcp_algo(freqs, mags)
                hpcps.append(hpcp)

            if len(hpcps) == 0:
                return []

            hpcps = np.array(hpcps)

            # Detect chords
            chords, strength = chords_algo(hpcps)

            # Convert to output format
            result = []
            frame_duration = hop_size / sample_rate

            for i, (chord, s) in enumerate(zip(chords, strength)):
                start_time = i * frame_duration
                end_time = (i + 1) * frame_duration
                result.append((start_time, end_time, chord))

            # Merge consecutive same chords
            merged = []
            for chord in result:
                if merged and merged[-1][2] == chord[2]:
                    merged[-1] = (merged[-1][0], chord[1], chord[2])
                else:
                    merged.append(chord)

            return merged

        except Exception as e:
            raise HarmonyAnalysisError(f"Essentia chord detection failed: {e}") from e

    def detect_chords(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[tuple[float, float, str]]:
        """Detect chords using configured method.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            List of (start_time, end_time, chord_name) tuples.

        Raises:
            DependencyNotAvailableError: If essentia is requested but unavailable.
            HarmonyAnalysisError: If chord detection fails.
        """
        if self.config.use_essentia:
            if not self.essentia_available:
                raise DependencyNotAvailableError(
                    "essentia", "chord detection with Essentia"
                )
            return self.detect_chords_essentia(audio, sample_rate)
        else:
            return self.detect_chords_chroma(audio, sample_rate)

    def compute_key_consistency(
        self,
        chroma: np.ndarray,
        detected_key: str,
    ) -> float:
        """Compute how consistently the audio stays in the detected key.

        Measures frame-by-frame correlation with the detected key profile.

        Args:
            chroma: Chroma features (12 x n_frames).
            detected_key: The detected key string.

        Returns:
            Consistency score from 0.0 to 1.0.
        """
        if chroma.shape[1] == 0 or not detected_key:
            return 0.0

        # Parse key
        parts = detected_key.split()
        if len(parts) != 2:
            return 0.0

        root = parts[0]
        mode = parts[1]

        if root not in PITCH_CLASSES:
            return 0.0

        root_idx = PITCH_CLASSES.index(root)

        # Get the appropriate profile
        if mode == "major":
            profile = np.roll(KRUMHANSL_MAJOR, root_idx)
        else:
            profile = np.roll(KRUMHANSL_MINOR, root_idx)

        # Compute correlation for each frame
        correlations = []
        for i in range(chroma.shape[1]):
            frame_chroma = chroma[:, i]
            if np.max(frame_chroma) > 0:
                frame_chroma = frame_chroma / np.max(frame_chroma)
                corr = np.corrcoef(frame_chroma, profile)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if len(correlations) == 0:
            return 0.0

        # Average correlation (shifted to 0-1 range)
        avg_corr = np.mean(correlations)
        consistency = (avg_corr + 1.0) / 2.0  # Map from [-1,1] to [0,1]

        return float(np.clip(consistency, 0.0, 1.0))

    def compute_progression_quality(
        self,
        chord_sequence: list[tuple[float, float, str]],
    ) -> float:
        """Evaluate chord progression quality.

        Measures how musically sensible the chord progressions are
        based on common patterns in tonal music.

        Args:
            chord_sequence: List of (start, end, chord) tuples.

        Returns:
            Quality score from 0.0 to 1.0.
        """
        if len(chord_sequence) < 2:
            return 0.5  # Neutral if no progression

        chords = [c[2] for c in chord_sequence]

        # Remove "N" (no chord) entries
        chords = [c for c in chords if c != "N"]

        if len(chords) < 2:
            return 0.3

        # Common progression patterns (relative to root)
        # Score based on interval patterns
        quality_score = 0.5  # Base score

        # Count transitions
        n_transitions = len(chords) - 1
        good_transitions = 0

        for i in range(n_transitions):
            curr = chords[i]
            next_chord = chords[i + 1]

            # Get root notes
            curr_root = curr.replace("m", "")
            next_root = next_chord.replace("m", "")

            if curr_root in PITCH_CLASSES and next_root in PITCH_CLASSES:
                curr_idx = PITCH_CLASSES.index(curr_root)
                next_idx = PITCH_CLASSES.index(next_root)

                interval = (next_idx - curr_idx) % 12

                # Common good intervals: perfect 4th (5), perfect 5th (7),
                # step up (2), step down (10 = -2), same (0)
                good_intervals = {0, 2, 5, 7, 10}

                if interval in good_intervals:
                    good_transitions += 1

        # Score based on ratio of good transitions
        if n_transitions > 0:
            quality_score = 0.3 + 0.7 * (good_transitions / n_transitions)

        # Variety bonus (not just repeating same chord)
        unique_chords = len(set(chords))
        if unique_chords >= 3:
            quality_score = min(1.0, quality_score + 0.1)

        return float(np.clip(quality_score, 0.0, 1.0))

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> HarmonyReport:
        """Run complete harmony analysis.

        Args:
            audio: Audio samples (mono, float32, normalized).
            sample_rate: Sample rate in Hz.

        Returns:
            HarmonyReport with all harmony analysis results.

        Raises:
            HarmonyAnalysisError: If analysis fails.
        """
        # Detect key
        detected_key, key_confidence = self.detect_key(audio, sample_rate)

        # Compute chroma
        chroma = self.compute_chroma(audio, sample_rate)
        chroma_avg = np.mean(chroma, axis=1)

        # Detect chords
        chord_sequence = self.detect_chords(audio, sample_rate)

        # Count chords
        chord_count = len(chord_sequence)
        unique_chords = set(c[2] for c in chord_sequence if c[2] != "N")
        unique_chord_count = len(unique_chords)

        # Compute key consistency
        key_consistency = self.compute_key_consistency(chroma, detected_key)

        # Compute progression quality
        progression_quality = self.compute_progression_quality(chord_sequence)

        return HarmonyReport(
            detected_key=detected_key,
            key_confidence=key_confidence,
            chord_sequence=chord_sequence,
            chord_count=chord_count,
            unique_chord_count=unique_chord_count,
            key_consistency=key_consistency,
            progression_quality=progression_quality,
            chroma_features=chroma_avg,
        )

    def compute_score(self, report: HarmonyReport) -> float:
        """Convert harmony report to quality score.

        Score is based on:
        - Key consistency (40% weight) - staying in key
        - Progression quality (35% weight) - good chord progressions
        - Key confidence (25% weight) - clear tonal center

        Args:
            report: HarmonyReport from analyze().

        Returns:
            Quality score from 0.0 to 1.0 (higher = better harmony).
        """
        # Weighted combination
        score = (
            0.40 * report.key_consistency
            + 0.35 * report.progression_quality
            + 0.25 * report.key_confidence
        )

        return float(np.clip(score, 0.0, 1.0))
