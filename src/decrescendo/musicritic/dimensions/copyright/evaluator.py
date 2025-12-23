"""Copyright & Originality evaluator for MusiCritic."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import (
    BaseDimensionEvaluator,
    DimensionCategory,
    DimensionResult,
    SafetyDecision,
    SafetyDimension,
)
from .config import CopyrightConfig
from .exceptions import AudioTooShortError
from .fingerprint import (
    ChromaprintEncoder,
    FingerprintDatabase,
    FingerprintMatch,
    is_chromaprint_available,
)
from .similarity import (
    MelodyExtractor,
    MelodyReport,
    RhythmExtractor,
    RhythmReport,
    SimilarityMatcher,
)


class CopyrightEvaluator(BaseDimensionEvaluator):
    """Evaluator for Copyright & Originality (Dimension 5).

    Detects potential plagiarism or excessive similarity to existing music.
    This is a safety dimension - higher scores indicate more concerning
    similarity (potential copyright violation).

    The evaluator combines:
    - Fingerprint matching: Exact or near-exact audio matching via Chromaprint
    - Melody similarity: Pitch contour comparison
    - Rhythm similarity: Onset pattern and tempo comparison
    - Harmony similarity: Chroma/chord progression comparison

    Decisions:
    - ALLOW: Original content (score < flag_threshold)
    - FLAG: Potential similarity, requires human review (flag_threshold <= score < block_threshold)
    - BLOCK: High-confidence plagiarism (score >= block_threshold)

    Example:
        >>> evaluator = CopyrightEvaluator()
        >>> result = evaluator.evaluate(audio, sample_rate=44100)
        >>> print(f"Originality score: {(1 - result.score) * 100:.1f}%")
        >>> print(f"Decision: {result.metadata['decision']}")

    Attributes:
        dimension: SafetyDimension.COPYRIGHT
        category: DimensionCategory.SAFETY
    """

    dimension = SafetyDimension.COPYRIGHT
    category = DimensionCategory.SAFETY

    def __init__(
        self,
        config: CopyrightConfig | None = None,
        fingerprint_db: FingerprintDatabase | None = None,
        reference_audios: list[tuple[np.ndarray, int, str]] | None = None,
        cache_embeddings: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration.
            fingerprint_db: Pre-initialized fingerprint database for matching.
            reference_audios: List of (audio, sample_rate, name) tuples for
                comparison. Used when no fingerprint database is provided.
            cache_embeddings: Whether to cache audio analysis results.
        """
        super().__init__(cache_embeddings=cache_embeddings)
        self.config = config or CopyrightConfig()
        self._fingerprint_db = fingerprint_db
        self._reference_audios = reference_audios or []

        # Lazy-initialized components
        self._fingerprint_encoder: ChromaprintEncoder | None = None
        self._melody_extractor: MelodyExtractor | None = None
        self._rhythm_extractor: RhythmExtractor | None = None
        self._similarity_matcher: SimilarityMatcher | None = None

    @property
    def fingerprint_encoder(self) -> ChromaprintEncoder | None:
        """Lazily initialize and return fingerprint encoder.

        Returns None if Chromaprint is not available.
        """
        if self._fingerprint_encoder is None and is_chromaprint_available():
            self._fingerprint_encoder = ChromaprintEncoder(self.config.fingerprint_config)
        return self._fingerprint_encoder

    @property
    def melody_extractor(self) -> MelodyExtractor:
        """Lazily initialize and return melody extractor."""
        if self._melody_extractor is None:
            self._melody_extractor = MelodyExtractor(self.config.melody_config)
        return self._melody_extractor

    @property
    def rhythm_extractor(self) -> RhythmExtractor:
        """Lazily initialize and return rhythm extractor."""
        if self._rhythm_extractor is None:
            self._rhythm_extractor = RhythmExtractor(self.config.rhythm_config)
        return self._rhythm_extractor

    @property
    def similarity_matcher(self) -> SimilarityMatcher:
        """Lazily initialize and return similarity matcher."""
        if self._similarity_matcher is None:
            self._similarity_matcher = SimilarityMatcher(
                melody_config=self.config.melody_config,
                rhythm_config=self.config.rhythm_config,
                melody_weight=self.config.melody_weight,
                rhythm_weight=self.config.rhythm_weight,
                harmony_weight=self.config.harmony_weight,
            )
        return self._similarity_matcher

    @property
    def fingerprint_db(self) -> FingerprintDatabase | None:
        """Get the fingerprint database."""
        return self._fingerprint_db

    def set_fingerprint_db(self, db: FingerprintDatabase) -> None:
        """Set the fingerprint database.

        Args:
            db: Fingerprint database to use for matching.
        """
        self._fingerprint_db = db

    def add_reference(
        self,
        audio: np.ndarray,
        sample_rate: int,
        name: str,
    ) -> None:
        """Add a reference audio for comparison.

        Args:
            audio: Reference audio waveform.
            sample_rate: Sample rate of the audio.
            name: Identifier for the reference.
        """
        self._reference_audios.append((audio, sample_rate, name))

    def clear_references(self) -> None:
        """Clear all reference audios."""
        self._reference_audios.clear()

    def _evaluate_impl(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> DimensionResult:
        """Evaluate audio for copyright concerns.

        Args:
            audio: Preprocessed audio (mono, float32, normalized).
            sample_rate: Sample rate of audio.
            prompt: Text prompt (unused for copyright evaluation).
            **kwargs: Additional arguments (unused).

        Returns:
            DimensionResult with plagiarism score and evidence.

        Raises:
            AudioTooShortError: If audio is too short for analysis.
        """
        # Check minimum duration
        duration = len(audio) / sample_rate
        if duration < self.config.min_audio_duration:
            raise AudioTooShortError(duration, self.config.min_audio_duration)

        # Initialize sub-scores
        fingerprint_score = 0.0
        melody_score = 0.0
        rhythm_score = 0.0
        harmony_score = 0.0

        fingerprint_matches: list[dict[str, Any]] = []
        similarity_matches: list[dict[str, Any]] = []

        # 1. Fingerprint matching (if available)
        if self.fingerprint_encoder is not None and self._fingerprint_db is not None:
            try:
                fp, fp_duration = self.fingerprint_encoder.encode(audio, sample_rate)
                matches = self._fingerprint_db.search(fp, top_k=5, threshold=0.7)

                if matches:
                    fingerprint_score = matches[0].similarity
                    fingerprint_matches = [
                        {
                            "name": m.name,
                            "similarity": m.similarity,
                            "metadata": m.metadata,
                        }
                        for m in matches
                    ]
            except Exception:
                # Fingerprinting failed, continue with other methods
                pass

        # 2. Similarity matching against reference audios
        if self._reference_audios:
            max_melody_sim = 0.0
            max_rhythm_sim = 0.0
            max_harmony_sim = 0.0

            for ref_audio, ref_sr, ref_name in self._reference_audios:
                # Resample reference if needed
                if ref_sr != sample_rate:
                    import librosa

                    ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=sample_rate)

                report = self.similarity_matcher.compare(audio, ref_audio, sample_rate)

                if report.overall_similarity > 0.5:
                    similarity_matches.append(
                        {
                            "name": ref_name,
                            "melody_similarity": report.melody_similarity,
                            "rhythm_similarity": report.rhythm_similarity,
                            "harmony_similarity": report.harmony_similarity,
                            "overall_similarity": report.overall_similarity,
                        }
                    )

                max_melody_sim = max(max_melody_sim, report.melody_similarity)
                max_rhythm_sim = max(max_rhythm_sim, report.rhythm_similarity)
                max_harmony_sim = max(max_harmony_sim, report.harmony_similarity)

            melody_score = max_melody_sim
            rhythm_score = max_rhythm_sim
            harmony_score = max_harmony_sim

        # 3. Compute final score (weighted combination)
        # Note: fingerprint_weight is only used if we have fingerprint matches
        if fingerprint_matches:
            final_score = (
                self.config.fingerprint_weight * fingerprint_score
                + self.config.melody_weight * melody_score
                + self.config.rhythm_weight * rhythm_score
                + self.config.harmony_weight * harmony_score
            )
        else:
            # Redistribute fingerprint weight to other components
            remaining_weight = 1.0 - self.config.fingerprint_weight
            if remaining_weight > 0:
                final_score = (
                    (self.config.melody_weight / remaining_weight) * melody_score
                    + (self.config.rhythm_weight / remaining_weight) * rhythm_score
                    + (self.config.harmony_weight / remaining_weight) * harmony_score
                ) * remaining_weight
            else:
                final_score = 0.0

        # If no references and no fingerprint DB, score is 0 (cannot detect plagiarism)
        if not self._reference_audios and self._fingerprint_db is None:
            final_score = 0.0

        # Clamp score to [0, 1]
        final_score = max(0.0, min(1.0, final_score))

        # Determine safety decision
        decision = self._make_decision(final_score)

        # Compute confidence
        confidence = self._compute_confidence(
            duration,
            fingerprint_matches,
            similarity_matches,
        )

        # Build metadata
        metadata = self._build_metadata(
            decision,
            fingerprint_matches,
            similarity_matches,
            duration,
        )

        # Generate explanation
        explanation = self._generate_explanation(final_score, decision, metadata)

        return DimensionResult(
            dimension=self.dimension,
            score=final_score,
            confidence=confidence,
            sub_scores={
                "fingerprint": fingerprint_score,
                "melody": melody_score,
                "rhythm": rhythm_score,
                "harmony": harmony_score,
            },
            metadata=metadata,
            explanation=explanation,
        )

    def _make_decision(self, score: float) -> SafetyDecision:
        """Make safety decision based on score.

        Args:
            score: Plagiarism score (0-1).

        Returns:
            SafetyDecision (ALLOW, FLAG, or BLOCK).
        """
        if score >= self.config.block_threshold:
            return SafetyDecision.BLOCK
        elif score >= self.config.flag_threshold:
            return SafetyDecision.FLAG
        else:
            return SafetyDecision.ALLOW

    def _compute_confidence(
        self,
        duration: float,
        fingerprint_matches: list[dict[str, Any]],
        similarity_matches: list[dict[str, Any]],
    ) -> float:
        """Compute confidence in the evaluation.

        Higher confidence when:
        - Longer audio (more data)
        - Have reference database to compare against
        - Strong matches or clear non-matches

        Args:
            duration: Audio duration in seconds.
            fingerprint_matches: List of fingerprint matches.
            similarity_matches: List of similarity matches.

        Returns:
            Confidence value (0-1).
        """
        # Duration confidence: reaches 1.0 at 30 seconds
        duration_conf = min(1.0, duration / 30.0)

        # Reference confidence: higher if we have references to compare
        has_fingerprint_db = self._fingerprint_db is not None and len(self._fingerprint_db) > 0
        has_references = len(self._reference_audios) > 0

        if has_fingerprint_db or has_references:
            reference_conf = 0.8
        else:
            # No references - low confidence (can't detect plagiarism)
            reference_conf = 0.3

        # Match confidence: higher for clear matches or non-matches
        if fingerprint_matches or similarity_matches:
            match_conf = 0.9  # Clear evidence
        else:
            match_conf = 0.7  # No matches found

        # Combine confidences
        confidence = 0.3 * duration_conf + 0.4 * reference_conf + 0.3 * match_conf

        return float(np.clip(confidence, 0.3, 1.0))

    def _build_metadata(
        self,
        decision: SafetyDecision,
        fingerprint_matches: list[dict[str, Any]],
        similarity_matches: list[dict[str, Any]],
        duration: float,
    ) -> dict[str, Any]:
        """Build metadata dictionary.

        Args:
            decision: Safety decision.
            fingerprint_matches: List of fingerprint matches.
            similarity_matches: List of similarity matches.
            duration: Audio duration in seconds.

        Returns:
            Metadata dictionary.
        """
        return {
            "decision": decision.value,
            "duration_seconds": duration,
            "fingerprint_available": is_chromaprint_available(),
            "fingerprint_db_size": len(self._fingerprint_db) if self._fingerprint_db else 0,
            "reference_count": len(self._reference_audios),
            "fingerprint_matches": fingerprint_matches,
            "similarity_matches": similarity_matches,
            "has_concerning_matches": len(fingerprint_matches) > 0 or len(similarity_matches) > 0,
        }

    def _generate_explanation(
        self,
        score: float,
        decision: SafetyDecision,
        metadata: dict[str, Any],
    ) -> str:
        """Generate human-readable explanation.

        Args:
            score: Final plagiarism score.
            decision: Safety decision.
            metadata: Analysis metadata.

        Returns:
            Explanation string.
        """
        parts = []

        # Overall assessment
        originality = (1.0 - score) * 100
        if decision == SafetyDecision.BLOCK:
            parts.append(
                f"High plagiarism risk detected (originality: {originality:.0f}%). {decision.value}."
            )
        elif decision == SafetyDecision.FLAG:
            parts.append(
                f"Potential similarity detected (originality: {originality:.0f}%). {decision.value} for human review."
            )
        else:
            parts.append(f"Original content (originality: {originality:.0f}%). {decision.value}.")

        # Fingerprint matches
        fp_matches = metadata.get("fingerprint_matches", [])
        if fp_matches:
            top_match = fp_matches[0]
            parts.append(
                f"Fingerprint match: '{top_match['name']}' ({top_match['similarity']:.0%} similarity)."
            )

        # Similarity matches
        sim_matches = metadata.get("similarity_matches", [])
        if sim_matches:
            top_match = sim_matches[0]
            parts.append(
                f"Similar to: '{top_match['name']}' "
                f"(melody: {top_match['melody_similarity']:.0%}, "
                f"rhythm: {top_match['rhythm_similarity']:.0%})."
            )

        # No references warning
        if not metadata.get("fingerprint_matches") and not metadata.get("similarity_matches"):
            if (
                metadata.get("reference_count", 0) == 0
                and metadata.get("fingerprint_db_size", 0) == 0
            ):
                parts.append("Note: No reference database available for comparison.")

        return " ".join(parts)
