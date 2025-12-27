"""Tension-resolution analysis for Musicality dimension.

Analyzes musical tension and resolution patterns to evaluate
the narrative arc and phrase structure of music.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .config import TensionConfig
from .exceptions import TensionAnalysisError
from .tis import TISReport


@dataclass
class TensionReport:
    """Report of tension-resolution analysis.

    Attributes:
        tension_curve: Time-series of tension values (0-1).
        resolution_points: Timestamps (in seconds) where tension resolves.
        resolution_count: Number of tension resolutions detected.
        resolution_strength: Average strength of resolutions (0-1).
        arc_quality: Quality of overall tension arc (0-1).
        average_tension: Mean tension level.
        tension_variance: Variance in tension levels.
        frame_rate: Frame rate of the tension curve in Hz.
    """

    tension_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    resolution_points: list[float] = field(default_factory=list)
    resolution_count: int = 0
    resolution_strength: float = 0.0
    arc_quality: float = 0.0
    average_tension: float = 0.0
    tension_variance: float = 0.0
    frame_rate: float = 0.0


class TensionAnalyzer:
    """Analyzes tension-resolution patterns in music.

    Uses TIS-derived features to compute a tension curve and
    detect resolution points where musical tension is released.

    Example:
        >>> analyzer = TensionAnalyzer()
        >>> tis_report = tis_analyzer.analyze(audio, sample_rate)
        >>> tension_report = analyzer.analyze(audio, sample_rate, tis_report)
        >>> print(f"Resolutions detected: {tension_report.resolution_count}")
    """

    def __init__(self, config: TensionConfig | None = None) -> None:
        """Initialize the tension analyzer.

        Args:
            config: Tension analysis configuration. Uses defaults if None.
        """
        self.config = config or TensionConfig()

    def compute_tension_curve(
        self,
        tis_report: TISReport,
    ) -> np.ndarray:
        """Compute tension curve from TIS features.

        Tension is derived from:
        - Tensile strain (distance from tonal center)
        - Cloud diameter (harmonic complexity)

        Args:
            tis_report: TISReport from TIS analysis.

        Returns:
            Tension curve array (0-1 values).

        Raises:
            TensionAnalysisError: If tension computation fails.
        """
        try:
            strain_curve = tis_report.tensile_strain_curve
            diameter_curve = tis_report.cloud_diameter_curve

            if len(strain_curve) == 0:
                return np.array([0.0])

            # Ensure same length
            min_len = min(len(strain_curve), len(diameter_curve))
            strain_curve = strain_curve[:min_len]
            diameter_curve = diameter_curve[:min_len]

            # Combine strain and diameter for tension
            # Higher strain = more tension, higher diameter = more tension
            tension_curve = 0.6 * strain_curve + 0.4 * diameter_curve

            # Smooth the curve
            if len(tension_curve) >= self.config.smoothing_window:
                tension_curve = ndimage.uniform_filter1d(
                    tension_curve, size=self.config.smoothing_window
                )

            return tension_curve

        except Exception as e:
            raise TensionAnalysisError(f"Tension curve computation failed: {e}") from e

    def detect_resolutions(
        self,
        tension_curve: np.ndarray,
        frame_rate: float,
    ) -> tuple[list[float], list[float]]:
        """Detect points where tension resolves.

        Resolution is detected as a significant drop in tension
        following a period of higher tension.

        Args:
            tension_curve: Tension values over time (0-1).
            frame_rate: Frame rate of the tension curve in Hz.

        Returns:
            Tuple of (resolution_timestamps, resolution_strengths).

        Raises:
            TensionAnalysisError: If resolution detection fails.
        """
        try:
            if len(tension_curve) < 3:
                return [], []

            resolution_times = []
            resolution_strengths = []

            # Window size in frames
            window_frames = max(1, int(self.config.cadence_window_sec * frame_rate))

            # Look for tension drops
            for i in range(window_frames, len(tension_curve) - 1):
                # Average tension before this point
                pre_tension = np.mean(tension_curve[i - window_frames : i])

                # Current tension
                curr_tension = tension_curve[i]

                # Check for significant drop
                drop = pre_tension - curr_tension

                if drop >= self.config.min_tension_drop:
                    # Also check that tension continues to stay low or rises gradually
                    post_frames = min(window_frames // 2, len(tension_curve) - i - 1)
                    if post_frames > 0:
                        post_tension = np.mean(tension_curve[i : i + post_frames])

                        # Resolution is valid if post-tension stays relatively low
                        if post_tension <= pre_tension - self.config.resolution_threshold:
                            time_sec = i / frame_rate
                            resolution_times.append(time_sec)
                            resolution_strengths.append(float(drop))

            # Merge nearby resolutions (within cadence window)
            merged_times = []
            merged_strengths = []

            for t, s in zip(resolution_times, resolution_strengths):
                if merged_times and (t - merged_times[-1]) < self.config.cadence_window_sec:
                    # Keep the stronger one
                    if s > merged_strengths[-1]:
                        merged_times[-1] = t
                        merged_strengths[-1] = s
                else:
                    merged_times.append(t)
                    merged_strengths.append(s)

            return merged_times, merged_strengths

        except Exception as e:
            raise TensionAnalysisError(f"Resolution detection failed: {e}") from e

    def compute_arc_quality(
        self,
        tension_curve: np.ndarray,
        resolution_points: list[float],
        duration: float,
    ) -> float:
        """Evaluate quality of the overall tension arc.

        Good arcs have:
        - Variation in tension (not flat)
        - Regular resolutions (phrasing)
        - Final resolution near the end
        - Build-up before resolutions

        Args:
            tension_curve: Tension values over time.
            resolution_points: Detected resolution timestamps.
            duration: Total audio duration in seconds.

        Returns:
            Arc quality score (0-1).
        """
        if len(tension_curve) < 2:
            return 0.5

        scores = []

        # 1. Variation score: reward appropriate variation
        tension_std = np.std(tension_curve)
        # Optimal std around 0.15-0.25
        variation_score = min(1.0, tension_std * 4) * min(1.0, 2.0 - tension_std * 4)
        scores.append(variation_score)

        # 2. Resolution regularity: reward regular phrase structure
        if len(resolution_points) >= 2:
            intervals = np.diff(resolution_points)
            interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
            # Lower CV = more regular = better
            regularity_score = 1.0 - min(1.0, interval_cv)
        elif len(resolution_points) == 1:
            regularity_score = 0.5
        else:
            regularity_score = 0.3  # No resolutions is not ideal
        scores.append(regularity_score)

        # 3. Final resolution: reward resolution near the end
        if resolution_points:
            last_resolution = resolution_points[-1]
            # How close to the end (within last 20% is good)
            end_proximity = last_resolution / duration
            if end_proximity >= 0.8:
                final_score = 1.0
            elif end_proximity >= 0.6:
                final_score = 0.7
            else:
                final_score = 0.4
        else:
            final_score = 0.3
        scores.append(final_score)

        # 4. Range score: reward appropriate range of tension
        tension_range = np.max(tension_curve) - np.min(tension_curve)
        # Optimal range around 0.3-0.6
        range_score = min(1.0, tension_range * 2.5) * min(1.0, 2.0 - tension_range * 2)
        scores.append(range_score)

        # Average all scores
        arc_quality = float(np.mean(scores))

        return np.clip(arc_quality, 0.0, 1.0)

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
        tis_report: TISReport,
    ) -> TensionReport:
        """Run complete tension-resolution analysis.

        Args:
            audio: Audio samples (mono, float32, normalized).
            sample_rate: Sample rate in Hz.
            tis_report: TISReport from TIS analysis.

        Returns:
            TensionReport with all tension analysis results.

        Raises:
            TensionAnalysisError: If analysis fails.
        """
        try:
            duration = len(audio) / sample_rate
            frame_rate = tis_report.frame_rate

            if frame_rate == 0:
                return TensionReport()

            # Compute tension curve
            tension_curve = self.compute_tension_curve(tis_report)

            # Detect resolutions
            resolution_times, resolution_strengths = self.detect_resolutions(
                tension_curve, frame_rate
            )

            # Compute statistics
            average_tension = float(np.mean(tension_curve))
            tension_variance = float(np.var(tension_curve))

            # Average resolution strength
            if resolution_strengths:
                avg_resolution_strength = float(np.mean(resolution_strengths))
            else:
                avg_resolution_strength = 0.0

            # Compute arc quality
            arc_quality = self.compute_arc_quality(
                tension_curve, resolution_times, duration
            )

            return TensionReport(
                tension_curve=tension_curve,
                resolution_points=resolution_times,
                resolution_count=len(resolution_times),
                resolution_strength=avg_resolution_strength,
                arc_quality=arc_quality,
                average_tension=average_tension,
                tension_variance=tension_variance,
                frame_rate=frame_rate,
            )

        except TensionAnalysisError:
            raise
        except Exception as e:
            raise TensionAnalysisError(f"Tension analysis failed: {e}") from e

    def compute_score(
        self,
        report: TensionReport,
        duration: float,
    ) -> float:
        """Convert tension report to musicality score.

        Higher scores for:
        - Regular resolutions (phrased structure)
        - Strong resolutions (clear tension release)
        - Good arc quality (narrative structure)
        - Appropriate average tension (not monotonic)

        Args:
            report: TensionReport from analyze().
            duration: Audio duration in seconds.

        Returns:
            Musicality score from 0.0 to 1.0 (higher = better).
        """
        # 1. Resolution frequency score
        # Expect roughly 1 resolution per 4-8 seconds for typical music
        expected_resolutions = duration / 6.0  # Every 6 seconds on average
        if expected_resolutions > 0:
            resolution_ratio = report.resolution_count / expected_resolutions
            # Best if ratio is around 1.0, penalize too few or too many
            resolution_score = 1.0 - min(1.0, abs(resolution_ratio - 1.0))
        else:
            resolution_score = 0.5

        # 2. Resolution strength score
        # Stronger resolutions are generally better
        strength_score = min(1.0, report.resolution_strength * 3)

        # 3. Arc quality score (already 0-1)
        arc_score = report.arc_quality

        # 4. Tension variance score
        # Reward appropriate variance (not flat, not chaotic)
        optimal_variance = 0.04  # ~0.2 std
        variance_diff = abs(report.tension_variance - optimal_variance)
        variance_score = 1.0 - min(1.0, variance_diff * 10)

        # Weighted combination
        score = (
            0.25 * resolution_score
            + 0.20 * strength_score
            + 0.35 * arc_score
            + 0.20 * variance_score
        )

        return float(np.clip(score, 0.0, 1.0))
