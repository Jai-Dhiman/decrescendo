"""Artifact detection for Audio Quality dimension.

This module detects audio artifacts including clicks, clipping,
and AI-generated audio fingerprints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy import signal

from .config import ArtifactDetectionConfig
from .exceptions import ArtifactDetectionError


@dataclass
class ArtifactReport:
    """Report of detected artifacts.

    Attributes:
        click_count: Number of detected click/pop artifacts.
        click_timestamps: Timestamps of clicks in seconds.
        clipping_count: Number of clipping events detected.
        clipping_timestamps: Timestamps of clipping events in seconds.
        clipping_severity: Ratio of clipped samples (0.0-1.0).
        ai_artifact_score: AI generation fingerprint score (0.0-1.0).
            Higher values indicate more AI-like characteristics.
    """

    click_count: int = 0
    click_timestamps: list[float] = field(default_factory=list)
    clipping_count: int = 0
    clipping_timestamps: list[float] = field(default_factory=list)
    clipping_severity: float = 0.0
    ai_artifact_score: float = 0.0


class ArtifactDetector:
    """Detects audio artifacts including clicks, clipping, and AI fingerprints.

    Example:
        >>> detector = ArtifactDetector()
        >>> report = detector.analyze(audio, sample_rate=44100)
        >>> print(f"Clicks detected: {report.click_count}")
        >>> print(f"Clipping severity: {report.clipping_severity:.2%}")
    """

    def __init__(self, config: ArtifactDetectionConfig | None = None) -> None:
        """Initialize the artifact detector.

        Args:
            config: Artifact detection configuration. Uses defaults if None.
        """
        self.config = config or ArtifactDetectionConfig()

    def detect_clicks(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[int, list[float]]:
        """Detect click and pop artifacts using spectral flux.

        Clicks are characterized by sudden, broadband energy spikes.
        Uses onset detection with high sensitivity to catch transients.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (click_count, click_timestamps).

        Raises:
            ArtifactDetectionError: If detection fails.
        """
        try:
            # Compute onset envelope using spectral flux
            onset_env = librosa.onset.onset_strength(
                y=audio,
                sr=sample_rate,
                hop_length=512,
                aggregate=np.median,
            )

            # Normalize
            if onset_env.max() > 0:
                onset_env = onset_env / onset_env.max()

            # Find peaks that exceed threshold
            # Clicks are sudden spikes, so we look for outliers
            mean_env = np.mean(onset_env)
            std_env = np.std(onset_env)

            if std_env == 0:
                return 0, []

            # Threshold: peaks more than 3 std above mean
            threshold = mean_env + 3 * std_env * (1.0 - self.config.click_threshold)

            # Find peak locations
            peaks, properties = signal.find_peaks(
                onset_env,
                height=threshold,
                distance=int(0.05 * sample_rate / 512),  # Min 50ms between clicks
            )

            # Convert frame indices to timestamps
            timestamps = librosa.frames_to_time(
                peaks,
                sr=sample_rate,
                hop_length=512,
            ).tolist()

            return len(peaks), timestamps

        except Exception as e:
            raise ArtifactDetectionError(f"Click detection failed: {e}") from e

    def detect_clipping(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[int, list[float], float]:
        """Detect clipping by finding consecutive max-amplitude samples.

        Clipping occurs when audio exceeds the dynamic range and
        consecutive samples are at maximum amplitude.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (clip_count, clip_timestamps, severity).
            Severity is the ratio of clipped samples to total.

        Raises:
            ArtifactDetectionError: If detection fails.
        """
        try:
            threshold = self.config.clipping_threshold
            min_samples = self.config.min_clipping_samples

            # Find samples at or above clipping threshold
            clipped = np.abs(audio) >= threshold

            # Find runs of consecutive clipped samples
            clip_events = []
            clip_count = 0
            total_clipped_samples = 0

            # Use diff to find transitions
            diff = np.diff(clipped.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1

            # Handle edge cases
            if clipped[0]:
                starts = np.insert(starts, 0, 0)
            if clipped[-1]:
                ends = np.append(ends, len(clipped))

            for start, end in zip(starts, ends):
                run_length = end - start
                if run_length >= min_samples:
                    clip_count += 1
                    timestamp = start / sample_rate
                    clip_events.append(timestamp)
                    total_clipped_samples += run_length

            severity = total_clipped_samples / len(audio) if len(audio) > 0 else 0.0

            return clip_count, clip_events, severity

        except Exception as e:
            raise ArtifactDetectionError(f"Clipping detection failed: {e}") from e

    def detect_ai_artifacts(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Detect AI generation artifacts via spectral analysis.

        AI-generated audio often exhibits:
        - Unnaturally smooth spectral envelopes
        - Reduced spectral variation over time
        - Unusual harmonic patterns

        This is a heuristic approach based on spectral characteristics.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            AI artifact score from 0.0 to 1.0.
            Higher values indicate more AI-like characteristics.

        Raises:
            ArtifactDetectionError: If detection fails.
        """
        try:
            # Compute spectrogram
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)

            if magnitude.size == 0:
                return 0.0

            # Feature 1: Spectral smoothness
            # AI audio tends to have smoother spectral envelopes
            spectral_diff = np.diff(magnitude, axis=0)
            spectral_roughness = np.mean(np.abs(spectral_diff))
            max_roughness = np.max(np.abs(spectral_diff)) + 1e-10

            # Normalize: lower roughness = more AI-like
            smoothness_score = 1.0 - (spectral_roughness / max_roughness)

            # Feature 2: Temporal consistency
            # AI audio often has unnaturally consistent spectra over time
            temporal_diff = np.diff(magnitude, axis=1)
            temporal_variation = np.std(temporal_diff)
            mean_magnitude = np.mean(magnitude) + 1e-10

            # Normalize: lower variation = more AI-like
            consistency_score = 1.0 - min(1.0, temporal_variation / mean_magnitude)

            # Feature 3: Spectral flatness consistency
            # AI audio tends to have more consistent spectral flatness
            flatness = librosa.feature.spectral_flatness(S=magnitude)
            flatness_std = np.std(flatness)

            # Normalize: lower std = more AI-like
            flatness_consistency_score = 1.0 - min(1.0, flatness_std * 10)

            # Combine features (weighted average)
            ai_score = (
                0.3 * smoothness_score + 0.4 * consistency_score + 0.3 * flatness_consistency_score
            )

            # Apply a threshold to reduce false positives
            # Only flag if score is notably high
            if ai_score < 0.5:
                ai_score = ai_score * 0.5  # Reduce low scores further

            return float(np.clip(ai_score, 0.0, 1.0))

        except Exception as e:
            raise ArtifactDetectionError(f"AI artifact detection failed: {e}") from e

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> ArtifactReport:
        """Run all artifact detection and return comprehensive report.

        Args:
            audio: Audio samples (mono, float32, normalized).
            sample_rate: Sample rate in Hz.

        Returns:
            ArtifactReport with all detected artifacts.

        Raises:
            ArtifactDetectionError: If analysis fails.
        """
        click_count, click_timestamps = self.detect_clicks(audio, sample_rate)
        clip_count, clip_timestamps, clip_severity = self.detect_clipping(audio, sample_rate)
        ai_score = self.detect_ai_artifacts(audio, sample_rate)

        return ArtifactReport(
            click_count=click_count,
            click_timestamps=click_timestamps,
            clipping_count=clip_count,
            clipping_timestamps=clip_timestamps,
            clipping_severity=clip_severity,
            ai_artifact_score=ai_score,
        )

    def compute_score(
        self,
        report: ArtifactReport,
        audio_duration: float = 30.0,
    ) -> float:
        """Convert artifact report to quality score.

        Score is based on:
        - Click count (30% weight, normalized by duration)
        - Clipping severity (40% weight)
        - AI artifacts (30% weight)

        Args:
            report: ArtifactReport from analyze().
            audio_duration: Audio duration in seconds (for normalization).

        Returns:
            Quality score from 0.0 to 1.0 (higher = fewer artifacts).
        """
        # Click score: penalize clicks (normalized by duration)
        # Expect no more than 1 click per 10 seconds
        clicks_per_10s = report.click_count / (audio_duration / 10.0)
        click_penalty = min(1.0, clicks_per_10s * 0.2)
        click_score = 1.0 - click_penalty

        # Clipping score: penalize based on severity
        # Any clipping is bad, severity makes it worse
        if report.clipping_count == 0:
            clipping_score = 1.0
        else:
            # Base penalty for any clipping, plus severity
            clipping_score = max(0.0, 0.7 - report.clipping_severity * 2.0)

        # AI artifact score: invert (lower AI score = better)
        ai_score = 1.0 - report.ai_artifact_score

        # Weighted combination
        score = 0.30 * click_score + 0.40 * clipping_score + 0.30 * ai_score

        return float(np.clip(score, 0.0, 1.0))
