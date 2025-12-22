"""Loudness analysis for Audio Quality dimension.

This module provides ITU-R BS.1770-4 compliant loudness measurements
including integrated LUFS, Loudness Range (LRA), and True Peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
from scipy import signal

from .config import LoudnessConfig
from .exceptions import AudioTooShortError, LoudnessAnalysisError


@dataclass
class LoudnessReport:
    """Report of loudness measurements.

    Attributes:
        integrated_lufs: Integrated loudness in LUFS (Loudness Units Full Scale).
        loudness_range_lu: Loudness Range in LU (Loudness Units).
        true_peak_dbtp: True Peak in dBTP (decibels True Peak).
        short_term_lufs: Short-term loudness values over time (optional).
        streaming_compliant: Whether audio meets streaming platform requirements.
        true_peak_compliant: Whether True Peak is below threshold.
        dynamic_range_appropriate: Whether LRA is in acceptable range.
    """

    integrated_lufs: float
    loudness_range_lu: float
    true_peak_dbtp: float
    short_term_lufs: np.ndarray | None = None
    streaming_compliant: bool = False
    true_peak_compliant: bool = False
    dynamic_range_appropriate: bool = False


class LoudnessAnalyzer:
    """Analyzes loudness metrics per ITU-R BS.1770-4.

    Uses pyloudnorm for standards-compliant measurements.

    Example:
        >>> analyzer = LoudnessAnalyzer()
        >>> report = analyzer.analyze(audio, sample_rate=44100)
        >>> print(f"LUFS: {report.integrated_lufs:.1f}")
        >>> print(f"Streaming compliant: {report.streaming_compliant}")
    """

    def __init__(self, config: LoudnessConfig | None = None) -> None:
        """Initialize the loudness analyzer.

        Args:
            config: Loudness configuration. Uses defaults if None.
        """
        self.config = config or LoudnessConfig()
        self._meter: pyln.Meter | None = None
        self._current_sample_rate: int | None = None

    def _get_meter(self, sample_rate: int) -> pyln.Meter:
        """Get or create a loudness meter for the given sample rate.

        Args:
            sample_rate: Audio sample rate in Hz.

        Returns:
            pyloudnorm Meter configured for the sample rate.
        """
        if self._meter is None or self._current_sample_rate != sample_rate:
            self._meter = pyln.Meter(sample_rate, block_size=self.config.block_size)
            self._current_sample_rate = sample_rate
        return self._meter

    def measure_integrated_loudness(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Measure integrated LUFS.

        Args:
            audio: Audio samples (mono or stereo).
            sample_rate: Sample rate in Hz.

        Returns:
            Integrated loudness in LUFS.

        Raises:
            LoudnessAnalysisError: If measurement fails.
        """
        try:
            meter = self._get_meter(sample_rate)
            loudness = meter.integrated_loudness(audio)
            # Handle -inf for silence
            if np.isinf(loudness):
                return -70.0  # Return a very low value for silence
            return float(loudness)
        except Exception as e:
            raise LoudnessAnalysisError(
                f"Failed to measure integrated loudness: {e}"
            ) from e

    def measure_loudness_range(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Measure Loudness Range (LRA).

        LRA measures the variation in loudness over time, excluding
        the loudest and quietest 5% of segments.

        Args:
            audio: Audio samples (mono or stereo).
            sample_rate: Sample rate in Hz.

        Returns:
            Loudness Range in LU.

        Raises:
            LoudnessAnalysisError: If measurement fails.
        """
        try:
            # Calculate short-term loudness (3s windows)
            meter = self._get_meter(sample_rate)
            block_size_samples = int(3.0 * sample_rate)  # 3 second blocks
            hop_size_samples = int(0.1 * sample_rate)  # 100ms hop

            if len(audio) < block_size_samples:
                # Audio too short for LRA, return 0
                return 0.0

            short_term_values = []
            for i in range(0, len(audio) - block_size_samples + 1, hop_size_samples):
                block = audio[i : i + block_size_samples]
                st_loudness = meter.integrated_loudness(block)
                if not np.isinf(st_loudness):
                    short_term_values.append(st_loudness)

            if len(short_term_values) < 2:
                return 0.0

            # Sort and exclude top/bottom 5%
            sorted_values = np.sort(short_term_values)
            n = len(sorted_values)
            low_idx = int(n * 0.05)
            high_idx = int(n * 0.95)

            if high_idx <= low_idx:
                return 0.0

            trimmed = sorted_values[low_idx:high_idx]
            lra = float(trimmed.max() - trimmed.min())
            return lra

        except Exception as e:
            raise LoudnessAnalysisError(
                f"Failed to measure loudness range: {e}"
            ) from e

    def measure_true_peak(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Measure True Peak using oversampling.

        True Peak detects inter-sample peaks that may cause clipping
        during D/A conversion. Uses 4x oversampling per ITU-R BS.1770-4.

        Args:
            audio: Audio samples (mono or stereo).
            sample_rate: Sample rate in Hz.

        Returns:
            True Peak in dBTP.

        Raises:
            LoudnessAnalysisError: If measurement fails.
        """
        try:
            # Handle silence or very small signals
            max_sample = np.max(np.abs(audio))
            if max_sample < 1e-10:
                return -70.0

            # 4x oversampling per ITU-R BS.1770-4
            oversampling_factor = 4

            # Resample to 4x sample rate
            num_samples = len(audio)
            upsampled_length = num_samples * oversampling_factor

            # Use scipy resample for accurate interpolation
            upsampled = signal.resample(audio, upsampled_length)

            # Find the maximum absolute value
            peak_linear = np.max(np.abs(upsampled))

            # Convert to dBTP (dB True Peak)
            if peak_linear > 0:
                true_peak_dbtp = 20 * np.log10(peak_linear)
            else:
                true_peak_dbtp = -70.0

            return float(true_peak_dbtp)

        except Exception as e:
            raise LoudnessAnalysisError(
                f"Failed to measure true peak: {e}"
            ) from e

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> LoudnessReport:
        """Run complete loudness analysis.

        Args:
            audio: Audio samples (mono, float32, normalized).
            sample_rate: Sample rate in Hz.

        Returns:
            LoudnessReport with all measurements.

        Raises:
            AudioTooShortError: If audio is too short for analysis.
            LoudnessAnalysisError: If analysis fails.
        """
        # Check minimum duration
        duration = len(audio) / sample_rate
        min_duration = 0.4  # Minimum for LUFS measurement
        if duration < min_duration:
            raise AudioTooShortError(duration, min_duration)

        # Measure all components
        integrated_lufs = self.measure_integrated_loudness(audio, sample_rate)
        loudness_range = self.measure_loudness_range(audio, sample_rate)
        true_peak = self.measure_true_peak(audio, sample_rate)

        # Check compliance
        true_peak_compliant = true_peak <= self.config.max_true_peak_dbtp
        dynamic_range_appropriate = (
            self.config.min_lra <= loudness_range <= self.config.max_lra
        )

        # Overall streaming compliance:
        # - LUFS within +/- 2 of target
        # - True Peak below threshold
        lufs_compliant = abs(integrated_lufs - self.config.target_lufs) <= 2.0
        streaming_compliant = lufs_compliant and true_peak_compliant

        return LoudnessReport(
            integrated_lufs=integrated_lufs,
            loudness_range_lu=loudness_range,
            true_peak_dbtp=true_peak,
            streaming_compliant=streaming_compliant,
            true_peak_compliant=true_peak_compliant,
            dynamic_range_appropriate=dynamic_range_appropriate,
        )

    def compute_score(self, report: LoudnessReport) -> float:
        """Convert loudness report to quality score.

        Score is based on:
        - Deviation from target LUFS (40% weight)
        - True Peak compliance (40% weight)
        - LRA appropriateness (20% weight)

        Args:
            report: LoudnessReport from analyze().

        Returns:
            Quality score from 0.0 to 1.0 (higher is better).
        """
        # LUFS score: penalize deviation from target
        lufs_deviation = abs(report.integrated_lufs - self.config.target_lufs)
        # Score drops to 0 at 10 LUFS deviation
        lufs_score = max(0.0, 1.0 - lufs_deviation / 10.0)

        # True Peak score: penalize exceeding limit
        if report.true_peak_dbtp <= self.config.max_true_peak_dbtp:
            peak_score = 1.0
        else:
            # Score drops 0.2 per dB over limit
            overage = report.true_peak_dbtp - self.config.max_true_peak_dbtp
            peak_score = max(0.0, 1.0 - overage * 0.2)

        # LRA score: penalize outside acceptable range
        if self.config.min_lra <= report.loudness_range_lu <= self.config.max_lra:
            lra_score = 1.0
        elif report.loudness_range_lu < self.config.min_lra:
            # Too compressed
            deviation = self.config.min_lra - report.loudness_range_lu
            lra_score = max(0.0, 1.0 - deviation * 0.1)
        else:
            # Too dynamic
            deviation = report.loudness_range_lu - self.config.max_lra
            lra_score = max(0.0, 1.0 - deviation * 0.05)

        # Weighted combination
        score = 0.4 * lufs_score + 0.4 * peak_score + 0.2 * lra_score

        return float(np.clip(score, 0.0, 1.0))
