"""Perceptual quality analysis for Audio Quality dimension.

This module provides spectral quality metrics including spectral centroid,
spectral flatness, and frequency balance analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy import signal

from .config import PerceptualConfig
from .exceptions import PerceptualAnalysisError


@dataclass
class PerceptualReport:
    """Report of perceptual quality metrics.

    Attributes:
        spectral_centroid_mean: Mean spectral centroid in Hz.
            Indicates the "brightness" of the audio.
        spectral_centroid_std: Standard deviation of spectral centroid.
            Indicates spectral variation over time.
        spectral_flatness_mean: Mean spectral flatness (0.0-1.0).
            0 = tonal (pure tones), 1 = noisy (white noise).
        spectral_flatness_std: Standard deviation of spectral flatness.
        frequency_balance: Energy distribution across frequency bands.
        balance_deviation: Deviation from ideal frequency balance (0.0-1.0).
        bandwidth_utilization: How much of the frequency range is used (0.0-1.0).
    """

    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    spectral_flatness_mean: float = 0.0
    spectral_flatness_std: float = 0.0
    frequency_balance: dict[str, float] = field(default_factory=dict)
    balance_deviation: float = 0.0
    bandwidth_utilization: float = 0.0


class PerceptualAnalyzer:
    """Analyzes perceptual audio quality metrics.

    Example:
        >>> analyzer = PerceptualAnalyzer()
        >>> report = analyzer.analyze(audio, sample_rate=44100)
        >>> print(f"Centroid: {report.spectral_centroid_mean:.0f} Hz")
        >>> print(f"Balance deviation: {report.balance_deviation:.2f}")
    """

    def __init__(self, config: PerceptualConfig | None = None) -> None:
        """Initialize the perceptual analyzer.

        Args:
            config: Perceptual configuration. Uses defaults if None.
        """
        self.config = config or PerceptualConfig()

    def compute_spectral_centroid(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[float, float]:
        """Compute mean and standard deviation of spectral centroid.

        The spectral centroid indicates the "center of mass" of the spectrum.
        Higher values indicate brighter audio.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (mean_centroid_hz, std_centroid_hz).

        Raises:
            PerceptualAnalysisError: If computation fails.
        """
        try:
            centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=sample_rate,
                n_fft=2048,
                hop_length=512,
            )

            mean_centroid = float(np.mean(centroid))
            std_centroid = float(np.std(centroid))

            return mean_centroid, std_centroid

        except Exception as e:
            raise PerceptualAnalysisError(f"Spectral centroid computation failed: {e}") from e

    def compute_spectral_flatness(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[float, float]:
        """Compute mean and standard deviation of spectral flatness.

        Spectral flatness measures the "tonality" of audio:
        - Values near 0 indicate tonal content (musical notes)
        - Values near 1 indicate noise-like content

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (mean_flatness, std_flatness).

        Raises:
            PerceptualAnalysisError: If computation fails.
        """
        try:
            flatness = librosa.feature.spectral_flatness(
                y=audio,
                n_fft=2048,
                hop_length=512,
            )

            mean_flatness = float(np.mean(flatness))
            std_flatness = float(np.std(flatness))

            return mean_flatness, std_flatness

        except Exception as e:
            raise PerceptualAnalysisError(f"Spectral flatness computation failed: {e}") from e

    def compute_frequency_balance(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float]:
        """Compute energy distribution across frequency bands.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Dictionary mapping band names to energy ratios (sum to 1.0).

        Raises:
            PerceptualAnalysisError: If computation fails.
        """
        try:
            # Compute power spectrum
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            power = np.abs(stft) ** 2

            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

            # Calculate energy per band
            band_names = ["bass", "mids", "upper_mids", "highs"]
            band_energies = []

            for low, high in self.config.frequency_bands:
                # Find bins in this frequency range
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    band_power = np.sum(power[mask, :])
                else:
                    band_power = 0.0
                band_energies.append(band_power)

            # Normalize to ratios
            total_energy = sum(band_energies)
            if total_energy > 0:
                ratios = [e / total_energy for e in band_energies]
            else:
                ratios = [0.25] * len(band_names)  # Default to equal

            return dict(zip(band_names, ratios))

        except Exception as e:
            raise PerceptualAnalysisError(f"Frequency balance computation failed: {e}") from e

    def compute_bandwidth_utilization(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Compute how much of the frequency range is utilized.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Bandwidth utilization from 0.0 to 1.0.

        Raises:
            PerceptualAnalysisError: If computation fails.
        """
        try:
            # Compute spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=sample_rate,
                n_fft=2048,
                hop_length=512,
            )

            mean_bandwidth = np.mean(bandwidth)

            # Normalize by Nyquist frequency
            nyquist = sample_rate / 2
            utilization = mean_bandwidth / nyquist

            return float(np.clip(utilization, 0.0, 1.0))

        except Exception as e:
            raise PerceptualAnalysisError(f"Bandwidth utilization computation failed: {e}") from e

    def _compute_balance_deviation(
        self,
        frequency_balance: dict[str, float],
    ) -> float:
        """Compute deviation from ideal frequency balance.

        Args:
            frequency_balance: Actual energy distribution.

        Returns:
            Deviation from 0.0 to 1.0 (0 = perfect balance).
        """
        band_names = ["bass", "mids", "upper_mids", "highs"]
        ideal = self.config.ideal_balance

        # Compute L1 distance from ideal
        deviation = 0.0
        for i, name in enumerate(band_names):
            actual = frequency_balance.get(name, 0.25)
            deviation += abs(actual - ideal[i])

        # Normalize (max deviation is 2.0 for L1)
        normalized = deviation / 2.0

        return float(np.clip(normalized, 0.0, 1.0))

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> PerceptualReport:
        """Run complete perceptual analysis.

        Args:
            audio: Audio samples (mono, float32, normalized).
            sample_rate: Sample rate in Hz.

        Returns:
            PerceptualReport with all metrics.

        Raises:
            PerceptualAnalysisError: If analysis fails.
        """
        centroid_mean, centroid_std = self.compute_spectral_centroid(audio, sample_rate)
        flatness_mean, flatness_std = self.compute_spectral_flatness(audio, sample_rate)
        frequency_balance = self.compute_frequency_balance(audio, sample_rate)
        bandwidth = self.compute_bandwidth_utilization(audio, sample_rate)
        balance_deviation = self._compute_balance_deviation(frequency_balance)

        return PerceptualReport(
            spectral_centroid_mean=centroid_mean,
            spectral_centroid_std=centroid_std,
            spectral_flatness_mean=flatness_mean,
            spectral_flatness_std=flatness_std,
            frequency_balance=frequency_balance,
            balance_deviation=balance_deviation,
            bandwidth_utilization=bandwidth,
        )

    def compute_score(self, report: PerceptualReport) -> float:
        """Convert perceptual report to quality score.

        Score is based on:
        - Frequency balance (40% weight)
        - Spectral centroid appropriateness (30% weight)
        - Bandwidth utilization (30% weight)

        Args:
            report: PerceptualReport from analyze().

        Returns:
            Quality score from 0.0 to 1.0 (higher is better).
        """
        # Balance score: lower deviation = better
        balance_score = 1.0 - report.balance_deviation

        # Centroid score: check if in appropriate range for music
        min_centroid = self.config.min_centroid_hz
        max_centroid = self.config.max_centroid_hz
        centroid = report.spectral_centroid_mean

        if min_centroid <= centroid <= max_centroid:
            centroid_score = 1.0
        elif centroid < min_centroid:
            # Too dark/muddy
            deviation = (min_centroid - centroid) / min_centroid
            centroid_score = max(0.5, 1.0 - deviation)
        else:
            # Too bright/harsh
            deviation = (centroid - max_centroid) / max_centroid
            centroid_score = max(0.5, 1.0 - deviation)

        # Bandwidth score: more utilization is generally better
        # But not too much (pure noise would be 1.0)
        bandwidth = report.bandwidth_utilization
        if 0.2 <= bandwidth <= 0.8:
            bandwidth_score = 1.0
        elif bandwidth < 0.2:
            # Too narrow
            bandwidth_score = 0.5 + bandwidth * 2.5
        else:
            # Too wide (might be noisy)
            bandwidth_score = 0.5 + (1.0 - bandwidth) * 2.5

        # Weighted combination
        score = 0.40 * balance_score + 0.30 * centroid_score + 0.30 * bandwidth_score

        return float(np.clip(score, 0.0, 1.0))
