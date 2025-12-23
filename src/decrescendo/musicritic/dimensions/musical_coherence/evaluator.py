"""Musical Coherence evaluator for MusiCritic."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import (
    BaseDimensionEvaluator,
    DimensionCategory,
    DimensionResult,
    QualityDimension,
)
from .config import MusicalCoherenceConfig
from .exceptions import AudioTooShortError
from .harmony import HarmonyAnalyzer, HarmonyReport
from .melody import MelodyAnalyzer, MelodyReport
from .rhythm import RhythmAnalyzer, RhythmReport
from .structure import StructureAnalyzer, StructureReport


class MusicalCoherenceEvaluator(BaseDimensionEvaluator):
    """Evaluator for Musical Coherence (Dimension 2).

    Measures structural and compositional quality by analyzing:
    - Structure: section detection, repetition patterns
    - Harmony: chord progressions, key consistency
    - Rhythm: beat tracking, tempo stability
    - Melody: pitch coherence, phrase structure

    The evaluator combines sub-scores from each component with
    configurable weights to produce a final coherence score.

    Example:
        >>> evaluator = MusicalCoherenceEvaluator()
        >>> result = evaluator.evaluate(audio, sample_rate=22050)
        >>> print(f"Score: {result.scaled_score:.1f}/100")
        >>> print(f"Coherence level: {result.metadata['coherence_level']}")

    Attributes:
        dimension: QualityDimension.MUSICAL_COHERENCE
        category: DimensionCategory.QUALITY
    """

    dimension = QualityDimension.MUSICAL_COHERENCE
    category = DimensionCategory.QUALITY

    def __init__(
        self,
        config: MusicalCoherenceConfig | None = None,
        structure_analyzer: StructureAnalyzer | None = None,
        harmony_analyzer: HarmonyAnalyzer | None = None,
        rhythm_analyzer: RhythmAnalyzer | None = None,
        melody_analyzer: MelodyAnalyzer | None = None,
        cache_embeddings: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration.
            structure_analyzer: Pre-initialized structure analyzer.
            harmony_analyzer: Pre-initialized harmony analyzer.
            rhythm_analyzer: Pre-initialized rhythm analyzer.
            melody_analyzer: Pre-initialized melody analyzer.
            cache_embeddings: Whether to cache analysis results.
        """
        super().__init__(cache_embeddings=cache_embeddings)
        self.config = config or MusicalCoherenceConfig()
        self._structure_analyzer = structure_analyzer
        self._harmony_analyzer = harmony_analyzer
        self._rhythm_analyzer = rhythm_analyzer
        self._melody_analyzer = melody_analyzer

    @property
    def structure_analyzer(self) -> StructureAnalyzer:
        """Lazily initialize and return structure analyzer."""
        if self._structure_analyzer is None:
            self._structure_analyzer = StructureAnalyzer(self.config.structure_config)
        return self._structure_analyzer

    @property
    def harmony_analyzer(self) -> HarmonyAnalyzer:
        """Lazily initialize and return harmony analyzer."""
        if self._harmony_analyzer is None:
            self._harmony_analyzer = HarmonyAnalyzer(self.config.harmony_config)
        return self._harmony_analyzer

    @property
    def rhythm_analyzer(self) -> RhythmAnalyzer:
        """Lazily initialize and return rhythm analyzer."""
        if self._rhythm_analyzer is None:
            self._rhythm_analyzer = RhythmAnalyzer(self.config.rhythm_config)
        return self._rhythm_analyzer

    @property
    def melody_analyzer(self) -> MelodyAnalyzer:
        """Lazily initialize and return melody analyzer."""
        if self._melody_analyzer is None:
            self._melody_analyzer = MelodyAnalyzer(self.config.melody_config)
        return self._melody_analyzer

    def _evaluate_impl(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> DimensionResult:
        """Evaluate musical coherence.

        Args:
            audio: Preprocessed audio (mono, float32, normalized).
            sample_rate: Sample rate of audio.
            prompt: Text prompt (unused for coherence).
            **kwargs: Additional arguments (unused).

        Returns:
            DimensionResult with coherence score and detailed breakdown.

        Raises:
            AudioTooShortError: If audio is too short for analysis.
        """
        # Check minimum duration
        duration = len(audio) / sample_rate
        if duration < self.config.min_audio_duration:
            raise AudioTooShortError(duration, self.config.min_audio_duration)

        # Run all analyzers
        structure_report = self.structure_analyzer.analyze(audio, sample_rate)
        harmony_report = self.harmony_analyzer.analyze(audio, sample_rate)
        rhythm_report = self.rhythm_analyzer.analyze(audio, sample_rate)
        melody_report = self.melody_analyzer.analyze(audio, sample_rate)

        # Compute sub-scores
        structure_score = self.structure_analyzer.compute_score(
            structure_report, audio_duration=duration
        )
        harmony_score = self.harmony_analyzer.compute_score(harmony_report)
        rhythm_score = self.rhythm_analyzer.compute_score(rhythm_report)
        melody_score = self.melody_analyzer.compute_score(melody_report, duration)

        # Weighted combination
        final_score = (
            self.config.structure_weight * structure_score
            + self.config.harmony_weight * harmony_score
            + self.config.rhythm_weight * rhythm_score
            + self.config.melody_weight * melody_score
        )

        # Classify coherence level
        coherence_level = self._classify_coherence(final_score)

        # Compute confidence
        confidence = self._compute_confidence(
            duration, structure_score, harmony_score, rhythm_score, melody_score
        )

        # Build metadata
        metadata = self._build_metadata(
            structure_report,
            harmony_report,
            rhythm_report,
            melody_report,
            coherence_level,
        )

        # Generate explanation
        explanation = self._generate_explanation(final_score, metadata)

        return DimensionResult(
            dimension=self.dimension,
            score=final_score,
            confidence=confidence,
            sub_scores={
                "structure": structure_score,
                "harmony": harmony_score,
                "rhythm": rhythm_score,
                "melody": melody_score,
            },
            metadata=metadata,
            explanation=explanation,
        )

    def _classify_coherence(self, score: float) -> str:
        """Classify coherence level based on score.

        Args:
            score: Normalized score (0-1).

        Returns:
            Coherence level: "excellent", "good", "moderate", or "poor".
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
        structure_score: float,
        harmony_score: float,
        rhythm_score: float,
        melody_score: float,
    ) -> float:
        """Compute confidence in the evaluation.

        Higher confidence for:
        - Longer audio (more data)
        - Agreement between sub-scores

        Args:
            duration: Audio duration in seconds.
            structure_score: Structure sub-score.
            harmony_score: Harmony sub-score.
            rhythm_score: Rhythm sub-score.
            melody_score: Melody sub-score.

        Returns:
            Confidence value (0-1).
        """
        # Duration confidence: reaches 1.0 at 30 seconds
        duration_conf = min(1.0, duration / 30.0)

        # Score agreement: lower std = higher confidence
        scores = [structure_score, harmony_score, rhythm_score, melody_score]
        score_std = np.std(scores)
        agreement_conf = 1.0 - min(1.0, score_std * 2)

        # Combine
        confidence = 0.4 * duration_conf + 0.6 * agreement_conf

        return float(np.clip(confidence, 0.5, 1.0))

    def _build_metadata(
        self,
        structure_report: StructureReport,
        harmony_report: HarmonyReport,
        rhythm_report: RhythmReport,
        melody_report: MelodyReport,
        coherence_level: str,
    ) -> dict[str, Any]:
        """Build metadata dictionary from analysis reports.

        Args:
            structure_report: Structure analysis results.
            harmony_report: Harmony analysis results.
            rhythm_report: Rhythm analysis results.
            melody_report: Melody analysis results.
            coherence_level: Classified coherence level.

        Returns:
            Metadata dictionary.
        """
        return {
            # Coherence classification
            "coherence_level": coherence_level,
            # Structure metrics
            "section_count": structure_report.section_count,
            "sections": structure_report.sections,
            "repetition_ratio": structure_report.repetition_ratio,
            "structure_clarity": structure_report.structure_clarity,
            # Harmony metrics
            "detected_key": harmony_report.detected_key,
            "key_confidence": harmony_report.key_confidence,
            "key_consistency": harmony_report.key_consistency,
            "progression_quality": harmony_report.progression_quality,
            "chord_count": harmony_report.chord_count,
            "unique_chord_count": harmony_report.unique_chord_count,
            # Rhythm metrics
            "tempo_bpm": rhythm_report.tempo_bpm,
            "tempo_confidence": rhythm_report.tempo_confidence,
            "tempo_stability": rhythm_report.tempo_stability,
            "beat_strength": rhythm_report.beat_strength,
            "beat_count": rhythm_report.beat_count,
            # Melody metrics
            "voiced_ratio": melody_report.voiced_ratio,
            "pitch_range_hz": melody_report.pitch_range_hz,
            "phrase_count": melody_report.phrase_count,
            "contour_complexity": melody_report.contour_complexity,
            "pitch_stability": melody_report.pitch_stability,
        }

    def _generate_explanation(
        self,
        score: float,
        metadata: dict[str, Any],
    ) -> str:
        """Generate human-readable explanation of the score.

        Args:
            score: Final coherence score (0-1).
            metadata: Analysis metadata.

        Returns:
            Explanation string.
        """
        score_100 = score * 100
        coherence_level = metadata["coherence_level"]

        # Build explanation parts
        parts = []

        # Overall assessment
        if coherence_level == "excellent":
            parts.append(f"Excellent musical coherence (score: {score_100:.1f}/100).")
        elif coherence_level == "good":
            parts.append(f"Good musical coherence (score: {score_100:.1f}/100).")
        elif coherence_level == "moderate":
            parts.append(f"Moderate musical coherence (score: {score_100:.1f}/100).")
        else:
            parts.append(f"Poor musical coherence (score: {score_100:.1f}/100).")

        # Key information
        detected_key = metadata["detected_key"]
        if detected_key:
            key_conf = metadata["key_confidence"]
            if key_conf > 0.7:
                parts.append(f"Clear tonal center in {detected_key}.")
            elif key_conf > 0.4:
                parts.append(f"Detected key: {detected_key} (moderate confidence).")

        # Tempo information
        tempo = metadata["tempo_bpm"]
        tempo_stability = metadata["tempo_stability"]
        if tempo > 0:
            if tempo_stability > 0.8:
                parts.append(f"Steady tempo at {tempo:.0f} BPM.")
            elif tempo_stability > 0.5:
                parts.append(f"Tempo around {tempo:.0f} BPM with some variation.")
            else:
                parts.append(f"Tempo approximately {tempo:.0f} BPM (unstable).")

        # Structure information
        section_count = metadata["section_count"]
        repetition_ratio = metadata["repetition_ratio"]
        if section_count > 1:
            if repetition_ratio > 0.3:
                parts.append(f"Clear structure with {section_count} sections and repetition.")
            else:
                parts.append(f"Detected {section_count} sections (limited repetition).")

        # Melodic content
        voiced_ratio = metadata["voiced_ratio"]
        if voiced_ratio > 0.3:
            parts.append("Strong melodic content detected.")
        elif voiced_ratio < 0.1:
            parts.append("Limited melodic content.")

        return " ".join(parts)
