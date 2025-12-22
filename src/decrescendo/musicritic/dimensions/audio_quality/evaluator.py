"""Audio Quality evaluator for MusiCritic."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import (
    BaseDimensionEvaluator,
    DimensionCategory,
    DimensionResult,
    QualityDimension,
)
from .artifacts import ArtifactDetector, ArtifactReport
from .config import AudioQualityConfig
from .exceptions import AudioTooShortError
from .loudness import LoudnessAnalyzer, LoudnessReport
from .perceptual import PerceptualAnalyzer, PerceptualReport


class AudioQualityEvaluator(BaseDimensionEvaluator):
    """Evaluator for Audio Quality (Dimension 3).

    Measures audio production quality by analyzing:
    - Artifacts: clicks, clipping, AI generation fingerprints
    - Loudness: LUFS, LRA, True Peak (streaming compliance)
    - Perceptual: spectral quality, frequency balance

    The evaluator combines sub-scores from each component with
    configurable weights to produce a final quality score.

    Example:
        >>> evaluator = AudioQualityEvaluator()
        >>> result = evaluator.evaluate(audio, sample_rate=44100)
        >>> print(f"Score: {result.scaled_score:.1f}/100")
        >>> print(f"Streaming compliant: {result.metadata['streaming_compliant']}")

    Attributes:
        dimension: QualityDimension.AUDIO_QUALITY
        category: DimensionCategory.QUALITY
    """

    dimension = QualityDimension.AUDIO_QUALITY
    category = DimensionCategory.QUALITY

    def __init__(
        self,
        config: AudioQualityConfig | None = None,
        artifact_detector: ArtifactDetector | None = None,
        loudness_analyzer: LoudnessAnalyzer | None = None,
        perceptual_analyzer: PerceptualAnalyzer | None = None,
        cache_embeddings: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration.
            artifact_detector: Pre-initialized artifact detector.
            loudness_analyzer: Pre-initialized loudness analyzer.
            perceptual_analyzer: Pre-initialized perceptual analyzer.
            cache_embeddings: Whether to cache audio analysis results.
        """
        super().__init__(cache_embeddings=cache_embeddings)
        self.config = config or AudioQualityConfig()
        self._artifact_detector = artifact_detector
        self._loudness_analyzer = loudness_analyzer
        self._perceptual_analyzer = perceptual_analyzer

    @property
    def artifact_detector(self) -> ArtifactDetector:
        """Lazily initialize and return artifact detector."""
        if self._artifact_detector is None:
            self._artifact_detector = ArtifactDetector(self.config.artifact_config)
        return self._artifact_detector

    @property
    def loudness_analyzer(self) -> LoudnessAnalyzer:
        """Lazily initialize and return loudness analyzer."""
        if self._loudness_analyzer is None:
            self._loudness_analyzer = LoudnessAnalyzer(self.config.loudness_config)
        return self._loudness_analyzer

    @property
    def perceptual_analyzer(self) -> PerceptualAnalyzer:
        """Lazily initialize and return perceptual analyzer."""
        if self._perceptual_analyzer is None:
            self._perceptual_analyzer = PerceptualAnalyzer(
                self.config.perceptual_config
            )
        return self._perceptual_analyzer

    def _evaluate_impl(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> DimensionResult:
        """Evaluate audio quality.

        Args:
            audio: Preprocessed audio (mono, float32, normalized).
            sample_rate: Sample rate of audio.
            prompt: Text prompt (unused for audio quality).
            **kwargs: Additional arguments (unused).

        Returns:
            DimensionResult with quality score and detailed breakdown.

        Raises:
            AudioTooShortError: If audio is too short for analysis.
        """
        # Check minimum duration
        duration = len(audio) / sample_rate
        if duration < self.config.min_audio_duration:
            raise AudioTooShortError(duration, self.config.min_audio_duration)

        # Run all analyzers
        artifact_report = self.artifact_detector.analyze(audio, sample_rate)
        loudness_report = self.loudness_analyzer.analyze(audio, sample_rate)
        perceptual_report = self.perceptual_analyzer.analyze(audio, sample_rate)

        # Compute sub-scores
        artifact_score = self.artifact_detector.compute_score(
            artifact_report, audio_duration=duration
        )
        loudness_score = self.loudness_analyzer.compute_score(loudness_report)
        perceptual_score = self.perceptual_analyzer.compute_score(perceptual_report)

        # Weighted combination
        final_score = (
            self.config.artifact_weight * artifact_score
            + self.config.loudness_weight * loudness_score
            + self.config.perceptual_weight * perceptual_score
        )

        # Classify quality level
        quality_level = self._classify_quality(final_score)

        # Compute confidence
        confidence = self._compute_confidence(
            duration, artifact_score, loudness_score, perceptual_score
        )

        # Build metadata
        metadata = self._build_metadata(
            artifact_report, loudness_report, perceptual_report, quality_level
        )

        # Generate explanation
        explanation = self._generate_explanation(final_score, metadata)

        return DimensionResult(
            dimension=self.dimension,
            score=final_score,
            confidence=confidence,
            sub_scores={
                "artifacts": artifact_score,
                "loudness": loudness_score,
                "perceptual": perceptual_score,
            },
            metadata=metadata,
            explanation=explanation,
        )

    def _classify_quality(self, score: float) -> str:
        """Classify quality level based on score.

        Args:
            score: Normalized score (0-1).

        Returns:
            Quality level: "excellent", "good", "acceptable", or "poor".
        """
        if score >= self.config.excellent_threshold:
            return "excellent"
        elif score >= self.config.good_threshold:
            return "good"
        elif score >= self.config.acceptable_threshold:
            return "acceptable"
        else:
            return "poor"

    def _compute_confidence(
        self,
        duration: float,
        artifact_score: float,
        loudness_score: float,
        perceptual_score: float,
    ) -> float:
        """Compute confidence in the evaluation.

        Higher confidence for:
        - Longer audio (more data)
        - Agreement between sub-scores

        Args:
            duration: Audio duration in seconds.
            artifact_score: Artifact sub-score.
            loudness_score: Loudness sub-score.
            perceptual_score: Perceptual sub-score.

        Returns:
            Confidence value (0-1).
        """
        # Duration confidence: reaches 1.0 at 10 seconds
        duration_conf = min(1.0, duration / 10.0)

        # Score agreement: lower std = higher confidence
        scores = [artifact_score, loudness_score, perceptual_score]
        score_std = np.std(scores)
        agreement_conf = 1.0 - min(1.0, score_std * 2)

        # Combine
        confidence = 0.4 * duration_conf + 0.6 * agreement_conf

        return float(np.clip(confidence, 0.5, 1.0))

    def _build_metadata(
        self,
        artifact_report: ArtifactReport,
        loudness_report: LoudnessReport,
        perceptual_report: PerceptualReport,
        quality_level: str,
    ) -> dict[str, Any]:
        """Build metadata dictionary from analysis reports.

        Args:
            artifact_report: Artifact detection results.
            loudness_report: Loudness analysis results.
            perceptual_report: Perceptual analysis results.
            quality_level: Classified quality level.

        Returns:
            Metadata dictionary.
        """
        return {
            # Quality classification
            "quality_level": quality_level,
            # Streaming compliance
            "streaming_compliant": loudness_report.streaming_compliant,
            "true_peak_compliant": loudness_report.true_peak_compliant,
            # Loudness measurements
            "integrated_lufs": loudness_report.integrated_lufs,
            "loudness_range_lu": loudness_report.loudness_range_lu,
            "true_peak_dbtp": loudness_report.true_peak_dbtp,
            # Artifact counts
            "click_count": artifact_report.click_count,
            "clipping_count": artifact_report.clipping_count,
            "clipping_severity": artifact_report.clipping_severity,
            "ai_artifact_score": artifact_report.ai_artifact_score,
            # Perceptual metrics
            "spectral_centroid_hz": perceptual_report.spectral_centroid_mean,
            "spectral_flatness": perceptual_report.spectral_flatness_mean,
            "frequency_balance": perceptual_report.frequency_balance,
            "balance_deviation": perceptual_report.balance_deviation,
        }

    def _generate_explanation(
        self,
        score: float,
        metadata: dict[str, Any],
    ) -> str:
        """Generate human-readable explanation of the score.

        Args:
            score: Final quality score (0-1).
            metadata: Analysis metadata.

        Returns:
            Explanation string.
        """
        score_100 = score * 100
        quality_level = metadata["quality_level"]
        streaming_compliant = metadata["streaming_compliant"]

        # Build explanation parts
        parts = []

        # Overall assessment
        if quality_level == "excellent":
            parts.append(
                f"Excellent audio quality (score: {score_100:.1f}/100)."
            )
        elif quality_level == "good":
            parts.append(
                f"Good audio quality (score: {score_100:.1f}/100)."
            )
        elif quality_level == "acceptable":
            parts.append(
                f"Acceptable audio quality (score: {score_100:.1f}/100)."
            )
        else:
            parts.append(
                f"Poor audio quality (score: {score_100:.1f}/100)."
            )

        # Streaming compliance
        if streaming_compliant:
            parts.append("Meets streaming platform requirements.")
        else:
            lufs = metadata["integrated_lufs"]
            peak = metadata["true_peak_dbtp"]
            issues = []
            if abs(lufs - (-14.0)) > 2.0:
                issues.append(f"LUFS ({lufs:.1f}) outside target range")
            if peak > -1.0:
                issues.append(f"True Peak ({peak:.1f} dBTP) too high")
            if issues:
                parts.append(f"Streaming issues: {'; '.join(issues)}.")

        # Artifacts
        click_count = metadata["click_count"]
        clipping_count = metadata["clipping_count"]
        if click_count > 0 or clipping_count > 0:
            artifact_issues = []
            if click_count > 0:
                artifact_issues.append(f"{click_count} click(s)")
            if clipping_count > 0:
                artifact_issues.append(f"{clipping_count} clipping event(s)")
            parts.append(f"Detected artifacts: {', '.join(artifact_issues)}.")

        # AI artifacts warning
        ai_score = metadata["ai_artifact_score"]
        if ai_score > 0.6:
            parts.append("Audio shows characteristics of AI generation.")

        return " ".join(parts)
