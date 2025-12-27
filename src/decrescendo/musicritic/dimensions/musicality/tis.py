"""Tonal Interval Space (TIS) analysis for Musicality dimension.

TIS is a geometric representation where chords are mapped to points
in a 12-dimensional space (one dimension per pitch class), allowing
computation of harmonic tension and complexity.

References:
- Bernardes, G., et al. (2016). A multi-level tonal interval space
  for modelling pitch relatedness and musical consonance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy import ndimage

from .config import TISConfig
from .exceptions import TISAnalysisError


@dataclass
class TISReport:
    """Report of Tonal Interval Space analysis.

    Attributes:
        cloud_diameter: Average harmonic complexity (0-1).
            Smaller values indicate more consonant, focused harmony.
        cloud_momentum: Rate of harmonic change (0-1).
            Higher values indicate more rapid harmonic movement.
        tensile_strain: Average deviation from tonal center (0-1).
            Higher values indicate more tension/distance from key.
        cloud_diameter_curve: Time-series of cloud diameter values.
        tensile_strain_curve: Time-series of tensile strain values.
        tonal_center: Estimated tonal center pitch class (0-11).
        frame_rate: Frame rate of the curves in Hz.
    """

    cloud_diameter: float = 0.0
    cloud_momentum: float = 0.0
    tensile_strain: float = 0.0
    cloud_diameter_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    tensile_strain_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    tonal_center: int = 0
    frame_rate: float = 0.0


class TISAnalyzer:
    """Computes Tonal Interval Space features from audio.

    TIS maps chroma frames to points in a 12-dimensional space,
    where each dimension represents a pitch class. This enables
    geometric analysis of harmonic content including:

    - Cloud diameter: Spread of harmonic content (complexity)
    - Cloud momentum: Rate of movement through TIS (change)
    - Tensile strain: Distance from tonal center (tension)

    Example:
        >>> analyzer = TISAnalyzer()
        >>> report = analyzer.analyze(audio, sample_rate=22050)
        >>> print(f"Harmonic complexity: {report.cloud_diameter:.2f}")
        >>> print(f"Harmonic tension: {report.tensile_strain:.2f}")
    """

    def __init__(self, config: TISConfig | None = None) -> None:
        """Initialize the TIS analyzer.

        Args:
            config: TIS computation configuration. Uses defaults if None.
        """
        self.config = config or TISConfig()

    def compute_chroma_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Compute normalized chroma features from audio.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Chroma features (12 x n_frames), L2-normalized per frame.

        Raises:
            TISAnalysisError: If chroma computation fails.
        """
        try:
            # Compute chroma using CQT for better low-frequency resolution
            chroma = librosa.feature.chroma_cqt(
                y=audio,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            # L2 normalize each frame (map to unit sphere in 12D)
            norms = np.linalg.norm(chroma, axis=0, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            chroma_normalized = chroma / norms

            return chroma_normalized

        except Exception as e:
            raise TISAnalysisError(f"Chroma computation failed: {e}") from e

    def estimate_tonal_center(self, chroma: np.ndarray) -> int:
        """Estimate the tonal center (key root) from chroma features.

        Uses the pitch class with highest average energy as the tonal center.

        Args:
            chroma: Chroma features (12 x n_frames).

        Returns:
            Pitch class index (0-11) of the tonal center.
        """
        # Average chroma energy per pitch class
        avg_chroma = np.mean(chroma, axis=1)

        # The pitch class with highest energy is likely the tonic
        tonal_center = int(np.argmax(avg_chroma))

        return tonal_center

    def compute_cloud_diameter(self, chroma_frame: np.ndarray) -> float:
        """Compute cloud diameter for a single chroma frame.

        Cloud diameter measures the spread of pitch classes in TIS,
        indicating harmonic complexity. A pure unison has diameter 0,
        while a chromatic cluster has maximum diameter.

        Args:
            chroma_frame: Single chroma frame (12,), L2-normalized.

        Returns:
            Cloud diameter value (0-1).
        """
        # Only consider active pitch classes (above threshold)
        active_mask = chroma_frame > self.config.min_chroma_energy

        if np.sum(active_mask) < 2:
            # Single note or silence - no spread
            return 0.0

        # Get indices of active pitch classes
        active_indices = np.where(active_mask)[0]

        # Compute pairwise distances in pitch class space
        # Using circular distance (semitones, mod 12)
        max_dist = 0.0
        for i in range(len(active_indices)):
            for j in range(i + 1, len(active_indices)):
                # Circular distance
                diff = abs(active_indices[i] - active_indices[j])
                dist = min(diff, 12 - diff)  # Shortest path around circle
                # Weight by energy of both pitch classes
                weight = chroma_frame[active_indices[i]] * chroma_frame[active_indices[j]]
                weighted_dist = dist * weight
                max_dist = max(max_dist, weighted_dist)

        # Normalize to 0-1 (max distance is 6 semitones = tritone)
        diameter = max_dist / 6.0

        return float(np.clip(diameter, 0.0, 1.0))

    def compute_tensile_strain(
        self,
        chroma_frame: np.ndarray,
        tonal_center: int,
    ) -> float:
        """Compute tensile strain for a single chroma frame.

        Tensile strain measures how far the current harmonic content
        is from the tonal center, indicating musical tension.

        Args:
            chroma_frame: Single chroma frame (12,), L2-normalized.
            tonal_center: Pitch class index of the tonal center (0-11).

        Returns:
            Tensile strain value (0-1).
        """
        # Weight each pitch class by its distance from tonal center
        total_strain = 0.0
        total_energy = 0.0

        for pc in range(12):
            energy = chroma_frame[pc]
            if energy > self.config.min_chroma_energy:
                # Circular distance from tonal center
                diff = abs(pc - tonal_center)
                dist = min(diff, 12 - diff)

                # Weight strain by energy
                total_strain += dist * energy
                total_energy += energy

        if total_energy < self.config.min_chroma_energy:
            return 0.0

        # Average strain, normalized (max is 6 semitones)
        avg_strain = total_strain / total_energy
        normalized_strain = avg_strain / 6.0

        return float(np.clip(normalized_strain, 0.0, 1.0))

    def compute_cloud_momentum(
        self,
        chroma: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute cloud momentum (rate of harmonic change).

        Momentum measures how fast the harmony moves through TIS space,
        computed as the frame-to-frame distance in chroma space.

        Args:
            chroma: Chroma features (12 x n_frames), L2-normalized.

        Returns:
            Tuple of (average_momentum, momentum_curve).
        """
        n_frames = chroma.shape[1]

        if n_frames < 2:
            return 0.0, np.array([0.0])

        # Compute frame-to-frame distances
        momentum_curve = np.zeros(n_frames - 1)

        for i in range(n_frames - 1):
            # Euclidean distance between consecutive frames
            diff = chroma[:, i + 1] - chroma[:, i]
            dist = np.linalg.norm(diff)
            momentum_curve[i] = dist

        # Normalize: max distance for normalized vectors is sqrt(2)
        momentum_curve = momentum_curve / np.sqrt(2.0)

        # Average momentum
        avg_momentum = float(np.mean(momentum_curve))

        return avg_momentum, momentum_curve

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> TISReport:
        """Run complete TIS analysis on audio.

        Args:
            audio: Audio samples (mono, float32, normalized).
            sample_rate: Sample rate in Hz.

        Returns:
            TISReport with all TIS analysis results.

        Raises:
            TISAnalysisError: If analysis fails.
        """
        try:
            # Compute chroma features
            chroma = self.compute_chroma_features(audio, sample_rate)
            n_frames = chroma.shape[1]

            if n_frames == 0:
                return TISReport()

            # Estimate tonal center
            tonal_center = self.estimate_tonal_center(chroma)

            # Compute per-frame metrics
            diameter_curve = np.zeros(n_frames)
            strain_curve = np.zeros(n_frames)

            for i in range(n_frames):
                frame = chroma[:, i]
                diameter_curve[i] = self.compute_cloud_diameter(frame)
                strain_curve[i] = self.compute_tensile_strain(frame, tonal_center)

            # Smooth curves
            if len(diameter_curve) >= self.config.smoothing_window:
                diameter_curve = ndimage.uniform_filter1d(
                    diameter_curve, size=self.config.smoothing_window
                )
                strain_curve = ndimage.uniform_filter1d(
                    strain_curve, size=self.config.smoothing_window
                )

            # Compute momentum
            cloud_momentum, _ = self.compute_cloud_momentum(chroma)

            # Compute frame rate
            frame_rate = sample_rate / self.config.hop_length

            return TISReport(
                cloud_diameter=float(np.mean(diameter_curve)),
                cloud_momentum=cloud_momentum,
                tensile_strain=float(np.mean(strain_curve)),
                cloud_diameter_curve=diameter_curve,
                tensile_strain_curve=strain_curve,
                tonal_center=tonal_center,
                frame_rate=frame_rate,
            )

        except TISAnalysisError:
            raise
        except Exception as e:
            raise TISAnalysisError(f"TIS analysis failed: {e}") from e

    def compute_score(self, report: TISReport) -> float:
        """Convert TIS report to musicality score.

        Higher scores for:
        - Moderate harmonic complexity (not too simple, not chaotic)
        - Appropriate momentum (variation without randomness)
        - Moderate tension that varies (not constant high or low)

        Args:
            report: TISReport from analyze().

        Returns:
            Musicality score from 0.0 to 1.0 (higher = better).
        """
        # Optimal values for musicality:
        # - Cloud diameter: moderate (0.3-0.6) - some complexity, not chaos
        # - Cloud momentum: moderate (0.2-0.5) - some change, not static/random
        # - Tensile strain: moderate (0.2-0.5) - some tension, not boring/harsh

        # Complexity score: penalize extremes (too simple or too complex)
        complexity_optimal = 0.45
        complexity_diff = abs(report.cloud_diameter - complexity_optimal)
        complexity_score = 1.0 - min(1.0, complexity_diff * 2.5)

        # Momentum score: penalize extremes (static or chaotic)
        momentum_optimal = 0.35
        momentum_diff = abs(report.cloud_momentum - momentum_optimal)
        momentum_score = 1.0 - min(1.0, momentum_diff * 2.5)

        # Tension score: moderate tension is good, but also reward variation
        tension_optimal = 0.35
        tension_diff = abs(report.tensile_strain - tension_optimal)
        tension_score = 1.0 - min(1.0, tension_diff * 2.0)

        # Variation bonus: reward tension that varies over time
        if len(report.tensile_strain_curve) > 1:
            strain_std = np.std(report.tensile_strain_curve)
            # Optimal variation around 0.1-0.2
            variation_score = min(1.0, strain_std * 5)
        else:
            variation_score = 0.0

        # Weighted combination
        score = (
            0.30 * complexity_score
            + 0.25 * momentum_score
            + 0.25 * tension_score
            + 0.20 * variation_score
        )

        return float(np.clip(score, 0.0, 1.0))
