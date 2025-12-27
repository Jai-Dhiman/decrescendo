"""Expression analysis for Musicality dimension.

Analyzes dynamic and expressive qualities of music including
loudness variation, dynamic range, and articulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from .config import ExpressionConfig
from .exceptions import ExpressionAnalysisError


@dataclass
class ExpressionReport:
    """Report of expression analysis.

    Attributes:
        dynamic_range_db: Dynamic range in decibels.
        dynamic_variation: Normalized variation in loudness (0-1).
        loudness_curve: Time-series of short-term loudness (dB).
        loudness_mean_db: Mean loudness in dB.
        loudness_std_db: Standard deviation of loudness in dB.
        crescendo_count: Number of crescendo (increasing loudness) events.
        decrescendo_count: Number of decrescendo (decreasing loudness) events.
        frame_rate: Frame rate of the loudness curve in Hz.
    """

    dynamic_range_db: float = 0.0
    dynamic_variation: float = 0.0
    loudness_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    loudness_mean_db: float = 0.0
    loudness_std_db: float = 0.0
    crescendo_count: int = 0
    decrescendo_count: int = 0
    frame_rate: float = 0.0


class ExpressionAnalyzer:
    """Analyzes expressive qualities of music.

    Measures dynamic variation, loudness changes, and other
    expressive features that contribute to musicality.

    Example:
        >>> analyzer = ExpressionAnalyzer()
        >>> report = analyzer.analyze(audio, sample_rate=22050)
        >>> print(f"Dynamic range: {report.dynamic_range_db:.1f} dB")
        >>> print(f"Dynamic variation: {report.dynamic_variation:.2f}")
    """

    def __init__(self, config: ExpressionConfig | None = None) -> None:
        """Initialize the expression analyzer.

        Args:
            config: Expression analysis configuration. Uses defaults if None.
        """
        self.config = config or ExpressionConfig()

    def compute_loudness_curve(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, float]:
        """Compute short-term loudness curve.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (loudness_curve_db, frame_rate).

        Raises:
            ExpressionAnalysisError: If computation fails.
        """
        try:
            # Compute RMS energy
            rms = librosa.feature.rms(
                y=audio,
                frame_length=self.config.rms_frame_length,
                hop_length=self.config.rms_hop_length,
            )[0]

            # Convert to dB, with floor to avoid -inf
            rms_db = 20 * np.log10(np.maximum(rms, 1e-10))

            # Compute frame rate
            frame_rate = sample_rate / self.config.rms_hop_length

            return rms_db, frame_rate

        except Exception as e:
            raise ExpressionAnalysisError(
                f"Loudness curve computation failed: {e}"
            ) from e

    def compute_dynamic_range(
        self,
        loudness_db: np.ndarray,
    ) -> float:
        """Compute dynamic range from loudness curve.

        Uses robust estimation (10th to 90th percentile) to avoid
        influence from outliers.

        Args:
            loudness_db: Loudness values in dB.

        Returns:
            Dynamic range in dB.
        """
        if len(loudness_db) < 2:
            return 0.0

        # Use percentiles for robust estimation
        low = np.percentile(loudness_db, 10)
        high = np.percentile(loudness_db, 90)

        dynamic_range = high - low

        return float(max(0.0, dynamic_range))

    def compute_dynamic_variation(
        self,
        loudness_db: np.ndarray,
    ) -> float:
        """Compute normalized dynamic variation.

        Maps the standard deviation of loudness to a 0-1 scale
        based on typical musical ranges.

        Args:
            loudness_db: Loudness values in dB.

        Returns:
            Normalized variation score (0-1).
        """
        if len(loudness_db) < 2:
            return 0.0

        std_db = np.std(loudness_db)

        # Normalize based on typical range
        # Std of 3-10 dB is typical for expressive music
        normalized = std_db / 10.0

        return float(np.clip(normalized, 0.0, 1.0))

    def detect_dynamics_events(
        self,
        loudness_db: np.ndarray,
        frame_rate: float,
    ) -> tuple[int, int]:
        """Detect crescendo and decrescendo events.

        Crescendo: sustained increase in loudness
        Decrescendo: sustained decrease in loudness

        Args:
            loudness_db: Loudness values in dB.
            frame_rate: Frame rate in Hz.

        Returns:
            Tuple of (crescendo_count, decrescendo_count).
        """
        if len(loudness_db) < 10:
            return 0, 0

        # Smooth the curve to avoid counting noise
        from scipy import ndimage
        smoothed = ndimage.uniform_filter1d(loudness_db, size=5)

        # Compute gradient
        gradient = np.diff(smoothed)

        # Minimum event duration in frames (about 0.5 seconds)
        min_duration = max(3, int(0.5 * frame_rate))

        crescendo_count = 0
        decrescendo_count = 0

        # Track sustained direction changes
        current_direction = 0  # 0=neutral, 1=up, -1=down
        duration = 0
        min_change_db = 3.0  # Minimum 3dB change to count

        for i in range(len(gradient)):
            if gradient[i] > 0.1:  # Increasing
                if current_direction == 1:
                    duration += 1
                else:
                    if current_direction == -1 and duration >= min_duration:
                        total_change = abs(smoothed[i] - smoothed[i - duration])
                        if total_change >= min_change_db:
                            decrescendo_count += 1
                    current_direction = 1
                    duration = 1
            elif gradient[i] < -0.1:  # Decreasing
                if current_direction == -1:
                    duration += 1
                else:
                    if current_direction == 1 and duration >= min_duration:
                        total_change = abs(smoothed[i] - smoothed[i - duration])
                        if total_change >= min_change_db:
                            crescendo_count += 1
                    current_direction = -1
                    duration = 1
            else:
                # Check if we just ended a sustained event
                if duration >= min_duration:
                    if current_direction == 1:
                        total_change = abs(smoothed[i] - smoothed[max(0, i - duration)])
                        if total_change >= min_change_db:
                            crescendo_count += 1
                    elif current_direction == -1:
                        total_change = abs(smoothed[i] - smoothed[max(0, i - duration)])
                        if total_change >= min_change_db:
                            decrescendo_count += 1
                current_direction = 0
                duration = 0

        return crescendo_count, decrescendo_count

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> ExpressionReport:
        """Run complete expression analysis.

        Args:
            audio: Audio samples (mono, float32, normalized).
            sample_rate: Sample rate in Hz.

        Returns:
            ExpressionReport with all expression analysis results.

        Raises:
            ExpressionAnalysisError: If analysis fails.
        """
        try:
            # Compute loudness curve
            loudness_db, frame_rate = self.compute_loudness_curve(audio, sample_rate)

            if len(loudness_db) == 0:
                return ExpressionReport()

            # Compute metrics
            dynamic_range = self.compute_dynamic_range(loudness_db)
            dynamic_variation = self.compute_dynamic_variation(loudness_db)

            # Statistics
            loudness_mean = float(np.mean(loudness_db))
            loudness_std = float(np.std(loudness_db))

            # Detect dynamics events
            crescendo_count, decrescendo_count = self.detect_dynamics_events(
                loudness_db, frame_rate
            )

            return ExpressionReport(
                dynamic_range_db=dynamic_range,
                dynamic_variation=dynamic_variation,
                loudness_curve=loudness_db,
                loudness_mean_db=loudness_mean,
                loudness_std_db=loudness_std,
                crescendo_count=crescendo_count,
                decrescendo_count=decrescendo_count,
                frame_rate=frame_rate,
            )

        except ExpressionAnalysisError:
            raise
        except Exception as e:
            raise ExpressionAnalysisError(f"Expression analysis failed: {e}") from e

    def compute_score(self, report: ExpressionReport) -> float:
        """Convert expression report to musicality score.

        Higher scores for:
        - Appropriate dynamic range (not flat, not excessive)
        - Dynamic variation following musical structure
        - Presence of crescendo/decrescendo (intentional dynamics)

        Args:
            report: ExpressionReport from analyze().

        Returns:
            Musicality score from 0.0 to 1.0 (higher = better).
        """
        # 1. Dynamic range score
        # Optimal range is between min and max config values
        dr = report.dynamic_range_db
        min_dr = self.config.min_dynamic_range_db
        max_dr = self.config.max_dynamic_range_db

        if dr < min_dr:
            # Too compressed/flat
            range_score = dr / min_dr
        elif dr > max_dr:
            # Excessive range (might indicate issues)
            range_score = max(0.3, 1.0 - (dr - max_dr) / max_dr)
        else:
            # In optimal range
            range_score = 1.0

        # 2. Dynamic variation score
        # Already normalized to 0-1, but we want moderate values
        var = report.dynamic_variation
        # Optimal variation around 0.3-0.6
        if var < 0.1:
            variation_score = var * 5  # Too flat
        elif var > 0.8:
            variation_score = max(0.4, 1.0 - (var - 0.8) * 2)  # Too chaotic
        else:
            variation_score = 0.6 + 0.4 * min(1.0, var / 0.6)

        # 3. Dynamics events score
        # Presence of intentional crescendo/decrescendo is good
        total_events = report.crescendo_count + report.decrescendo_count
        if total_events == 0:
            events_score = 0.3  # No dynamics events
        elif total_events <= 3:
            events_score = 0.5 + 0.15 * total_events  # Some events
        else:
            events_score = min(1.0, 0.8 + 0.05 * total_events)  # Many events

        # 4. Balance score: crescendos and decrescendos should be roughly balanced
        if report.crescendo_count + report.decrescendo_count > 0:
            balance = min(report.crescendo_count, report.decrescendo_count) / max(
                report.crescendo_count, report.decrescendo_count
            )
            balance_score = 0.5 + 0.5 * balance
        else:
            balance_score = 0.5

        # Weighted combination
        score = (
            0.30 * range_score
            + 0.30 * variation_score
            + 0.25 * events_score
            + 0.15 * balance_score
        )

        return float(np.clip(score, 0.0, 1.0))
