"""Musicality evaluator for MusiCritic."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import (
    BaseDimensionEvaluator,
    DimensionCategory,
    DimensionResult,
    QualityDimension,
)
from .config import MusicalityConfig
from .exceptions import AudioTooShortError
from .expression import ExpressionAnalyzer, ExpressionReport
from .tension import TensionAnalyzer, TensionReport
from .tis import TISAnalyzer, TISReport


class MusicalityEvaluator(BaseDimensionEvaluator):
    """Evaluator for Musicality (Dimension 4).

    Measures expressive and aesthetic qualities by analyzing:
    - TIS (Tonal Interval Space): Harmonic complexity, momentum, tension
    - Tension-Resolution: Musical narrative and phrase structure
    - Expression: Dynamic variation, crescendo/decrescendo patterns

    The evaluator combines sub-scores from each component with
    configurable weights to produce a final musicality score.

    Example:
        >>> evaluator = MusicalityEvaluator()
        >>> result = evaluator.evaluate(audio, sample_rate=22050)
        >>> print(f"Score: {result.scaled_score:.1f}/100")
        >>> print(f"Musicality level: {result.metadata['musicality_level']}")

    Attributes:
        dimension: QualityDimension.MUSICALITY
        category: DimensionCategory.QUALITY
    """

    dimension = QualityDimension.MUSICALITY
    category = DimensionCategory.QUALITY

    def __init__(
        self,
        config: MusicalityConfig | None = None,
        tis_analyzer: TISAnalyzer | None = None,
        tension_analyzer: TensionAnalyzer | None = None,
        expression_analyzer: ExpressionAnalyzer | None = None,
        cache_embeddings: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration.
            tis_analyzer: Pre-initialized TIS analyzer.
            tension_analyzer: Pre-initialized tension analyzer.
            expression_analyzer: Pre-initialized expression analyzer.
            cache_embeddings: Whether to cache analysis results.
        """
        super().__init__(cache_embeddings=cache_embeddings)
        self.config = config or MusicalityConfig()
        self._tis_analyzer = tis_analyzer
        self._tension_analyzer = tension_analyzer
        self._expression_analyzer = expression_analyzer

    @property
    def tis_analyzer(self) -> TISAnalyzer:
        """Lazily initialize and return TIS analyzer."""
        if self._tis_analyzer is None:
            self._tis_analyzer = TISAnalyzer(self.config.tis_config)
        return self._tis_analyzer

    @property
    def tension_analyzer(self) -> TensionAnalyzer:
        """Lazily initialize and return tension analyzer."""
        if self._tension_analyzer is None:
            self._tension_analyzer = TensionAnalyzer(self.config.tension_config)
        return self._tension_analyzer

    @property
    def expression_analyzer(self) -> ExpressionAnalyzer:
        """Lazily initialize and return expression analyzer."""
        if self._expression_analyzer is None:
            self._expression_analyzer = ExpressionAnalyzer(self.config.expression_config)
        return self._expression_analyzer

    def _evaluate_impl(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> DimensionResult:
        """Evaluate musicality.

        Args:
            audio: Preprocessed audio (mono, float32, normalized).
            sample_rate: Sample rate of audio.
            prompt: Text prompt (unused for musicality).
            **kwargs: Additional arguments (unused).

        Returns:
            DimensionResult with musicality score and detailed breakdown.

        Raises:
            AudioTooShortError: If audio is too short for analysis.
        """
        # Check minimum duration
        duration = len(audio) / sample_rate
        if duration < self.config.min_audio_duration:
            raise AudioTooShortError(duration, self.config.min_audio_duration)

        # Run TIS analysis (foundational)
        tis_report = self.tis_analyzer.analyze(audio, sample_rate)

        # Run tension analysis (depends on TIS)
        tension_report = self.tension_analyzer.analyze(audio, sample_rate, tis_report)

        # Run expression analysis (independent)
        expression_report = self.expression_analyzer.analyze(audio, sample_rate)

        # Compute sub-scores
        tis_score = self.tis_analyzer.compute_score(tis_report)
        tension_score = self.tension_analyzer.compute_score(tension_report, duration)
        expression_score = self.expression_analyzer.compute_score(expression_report)

        # Weighted combination
        final_score = (
            self.config.tis_weight * tis_score
            + self.config.tension_weight * tension_score
            + self.config.expression_weight * expression_score
        )

        # Classify musicality level
        musicality_level = self._classify_musicality(final_score)

        # Compute confidence
        confidence = self._compute_confidence(
            duration, tis_score, tension_score, expression_score
        )

        # Build metadata
        metadata = self._build_metadata(
            tis_report, tension_report, expression_report, musicality_level
        )

        # Generate explanation
        explanation = self._generate_explanation(final_score, metadata)

        return DimensionResult(
            dimension=self.dimension,
            score=final_score,
            confidence=confidence,
            sub_scores={
                "tis": tis_score,
                "tension": tension_score,
                "expression": expression_score,
            },
            metadata=metadata,
            explanation=explanation,
        )

    def _classify_musicality(self, score: float) -> str:
        """Classify musicality level based on score.

        Args:
            score: Normalized score (0-1).

        Returns:
            Musicality level: "excellent", "good", "moderate", or "poor".
        """
        if score >= self.config.excellent_threshold:
            return "excellent"
        elif score >= self.config.good_threshold:
            return "good"
        elif score >= self.config.moderate_threshold:
            return "moderate"
        else:
            return "poor"

    def _compute_confidence(
        self,
        duration: float,
        tis_score: float,
        tension_score: float,
        expression_score: float,
    ) -> float:
        """Compute confidence in the evaluation.

        Higher confidence for:
        - Longer audio (more data)
        - Agreement between sub-scores

        Args:
            duration: Audio duration in seconds.
            tis_score: TIS sub-score.
            tension_score: Tension sub-score.
            expression_score: Expression sub-score.

        Returns:
            Confidence value (0-1).
        """
        # Duration confidence: reaches 1.0 at 15 seconds
        duration_conf = min(1.0, duration / 15.0)

        # Score agreement: lower std = higher confidence
        scores = [tis_score, tension_score, expression_score]
        score_std = np.std(scores)
        agreement_conf = 1.0 - min(1.0, score_std * 2)

        # Combine
        confidence = 0.4 * duration_conf + 0.6 * agreement_conf

        return float(np.clip(confidence, 0.5, 1.0))

    def _build_metadata(
        self,
        tis_report: TISReport,
        tension_report: TensionReport,
        expression_report: ExpressionReport,
        musicality_level: str,
    ) -> dict[str, Any]:
        """Build metadata dictionary from analysis reports.

        Args:
            tis_report: TIS analysis results.
            tension_report: Tension analysis results.
            expression_report: Expression analysis results.
            musicality_level: Classified musicality level.

        Returns:
            Metadata dictionary.
        """
        # Map tonal center to pitch class name
        pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        tonal_center_name = pitch_classes[tis_report.tonal_center]

        return {
            # Classification
            "musicality_level": musicality_level,
            # TIS metrics
            "cloud_diameter": tis_report.cloud_diameter,
            "cloud_momentum": tis_report.cloud_momentum,
            "tensile_strain": tis_report.tensile_strain,
            "tonal_center": tonal_center_name,
            # Tension metrics
            "resolution_count": tension_report.resolution_count,
            "resolution_strength": tension_report.resolution_strength,
            "arc_quality": tension_report.arc_quality,
            "average_tension": tension_report.average_tension,
            # Expression metrics
            "dynamic_range_db": expression_report.dynamic_range_db,
            "dynamic_variation": expression_report.dynamic_variation,
            "crescendo_count": expression_report.crescendo_count,
            "decrescendo_count": expression_report.decrescendo_count,
        }

    def _generate_explanation(
        self,
        score: float,
        metadata: dict[str, Any],
    ) -> str:
        """Generate human-readable explanation of the score.

        Args:
            score: Final musicality score (0-1).
            metadata: Analysis metadata.

        Returns:
            Explanation string.
        """
        score_100 = score * 100
        level = metadata["musicality_level"]

        parts = []

        # Overall assessment
        if level == "excellent":
            parts.append(f"Excellent musicality (score: {score_100:.1f}/100).")
        elif level == "good":
            parts.append(f"Good musicality (score: {score_100:.1f}/100).")
        elif level == "moderate":
            parts.append(f"Moderate musicality (score: {score_100:.1f}/100).")
        else:
            parts.append(f"Poor musicality (score: {score_100:.1f}/100).")

        # Harmonic complexity
        diameter = metadata["cloud_diameter"]
        if diameter < 0.3:
            parts.append("Simple harmonic content.")
        elif diameter > 0.6:
            parts.append("Complex harmonic content.")
        else:
            parts.append("Moderate harmonic complexity.")

        # Tension-resolution
        resolution_count = metadata["resolution_count"]
        arc_quality = metadata["arc_quality"]
        if resolution_count > 0:
            if arc_quality > 0.7:
                parts.append(
                    f"Strong musical narrative with {resolution_count} resolution(s)."
                )
            else:
                parts.append(f"Detected {resolution_count} tension resolution(s).")
        else:
            parts.append("Limited tension-resolution structure detected.")

        # Dynamics
        dr = metadata["dynamic_range_db"]
        dynamics_events = metadata["crescendo_count"] + metadata["decrescendo_count"]
        if dr < 6:
            parts.append("Limited dynamic range.")
        elif dynamics_events > 0:
            parts.append(f"Expressive dynamics with {dynamics_events} dynamic event(s).")
        else:
            parts.append(f"Dynamic range of {dr:.1f} dB.")

        return " ".join(parts)
