"""Prompt Adherence evaluator for MusiCritic."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import (
    BaseDimensionEvaluator,
    DimensionCategory,
    DimensionResult,
    QualityDimension,
)
from .clap_encoder import CLAPEncoder
from .config import PromptAdherenceConfig
from .exceptions import PromptRequiredError


class PromptAdherenceEvaluator(BaseDimensionEvaluator):
    """Evaluator for Prompt Adherence (Dimension 1).

    Measures how well AI-generated audio matches a text prompt using
    CLAP (Contrastive Language-Audio Pretraining) embeddings.

    The evaluator computes cosine similarity between the text prompt
    embedding and the audio embedding. Scores are normalized to 0-1 range
    and classified as:
    - >0.7: Strong adherence
    - 0.5-0.7: Moderate adherence
    - <0.5: Poor adherence

    Example:
        >>> evaluator = PromptAdherenceEvaluator()
        >>> result = evaluator.evaluate(
        ...     audio=audio_array,
        ...     sample_rate=44100,
        ...     prompt="upbeat electronic dance music",
        ... )
        >>> print(f"Score: {result.scaled_score:.1f}/100")

    Attributes:
        dimension: QualityDimension.PROMPT_ADHERENCE
        category: DimensionCategory.QUALITY
    """

    dimension = QualityDimension.PROMPT_ADHERENCE
    category = DimensionCategory.QUALITY

    def __init__(
        self,
        config: PromptAdherenceConfig | None = None,
        encoder: CLAPEncoder | None = None,
        cache_embeddings: bool = True,
    ) -> None:
        """Initialize evaluator.

        Args:
            config: Evaluation configuration.
            encoder: Pre-initialized CLAP encoder. If None, creates one.
            cache_embeddings: Whether to cache audio embeddings.
        """
        super().__init__(cache_embeddings=cache_embeddings)
        self.config = config or PromptAdherenceConfig()
        self._encoder = encoder

    @property
    def encoder(self) -> CLAPEncoder:
        """Lazily initialize and return CLAP encoder."""
        if self._encoder is None:
            self._encoder = CLAPEncoder(self.config.encoder_config)
        return self._encoder

    def _evaluate_impl(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> DimensionResult:
        """Evaluate prompt adherence.

        Args:
            audio: Preprocessed audio (mono, float32, normalized).
            sample_rate: Sample rate of audio.
            prompt: Text prompt to evaluate against.
            **kwargs: Additional arguments (unused).

        Returns:
            DimensionResult with adherence score.

        Raises:
            PromptRequiredError: If prompt is None or empty.
        """
        # Validate prompt
        if prompt is None or prompt.strip() == "":
            raise PromptRequiredError(
                "Prompt is required for Prompt Adherence evaluation. "
                "Provide a text prompt describing the expected audio."
            )

        prompt = prompt.strip()

        # Get embeddings
        text_embedding = self.encoder.encode_text(
            prompt,
            use_cache=self.config.cache_text_embeddings,
        )
        audio_embedding = self.encoder.encode_audio(audio, sample_rate)

        # Compute similarity
        raw_similarity = self.encoder.compute_similarity(
            text_embedding,
            audio_embedding,
        )

        # Normalize to 0-1 range
        # CLAP similarity can range from -1 to 1, but for music prompts
        # it typically stays positive. We clip to 0-1 for scoring.
        score = float(np.clip(raw_similarity, 0.0, 1.0))

        # Generate explanation
        explanation = self._generate_explanation(score, prompt)

        # Classify adherence level
        adherence_level = self._classify_adherence(score)

        # Compute confidence
        confidence = self._compute_confidence(score)

        return DimensionResult(
            dimension=self.dimension,
            score=score,
            confidence=confidence,
            sub_scores={
                "clap_similarity": raw_similarity,
            },
            metadata={
                "prompt": prompt,
                "adherence_level": adherence_level,
                "embedding_dim": self.config.encoder_config.embedding_dim,
            },
            explanation=explanation,
        )

    def _classify_adherence(self, score: float) -> str:
        """Classify adherence level based on score.

        Args:
            score: Normalized score (0-1).

        Returns:
            Adherence level: "strong", "moderate", or "poor".
        """
        if score >= self.config.strong_adherence_threshold:
            return "strong"
        elif score >= self.config.moderate_adherence_threshold:
            return "moderate"
        else:
            return "poor"

    def _compute_confidence(self, score: float) -> float:
        """Compute confidence in the score.

        Higher confidence for scores further from the decision boundaries.

        Args:
            score: Normalized score (0-1).

        Returns:
            Confidence value (0-1).
        """
        # Distance from nearest threshold
        thresholds = [
            self.config.moderate_adherence_threshold,
            self.config.strong_adherence_threshold,
        ]

        min_distance = min(abs(score - t) for t in thresholds)

        # Higher distance = higher confidence
        # Base confidence of 0.5, plus up to 0.5 based on distance
        # Max distance from any threshold is about 0.5
        confidence = 0.5 + min_distance

        return float(np.clip(confidence, 0.5, 1.0))

    def _generate_explanation(self, score: float, prompt: str) -> str:
        """Generate human-readable explanation of the score.

        Args:
            score: Normalized score (0-1).
            prompt: The text prompt used.

        Returns:
            Explanation string.
        """
        level = self._classify_adherence(score)
        score_100 = score * 100

        # Truncate long prompts for display
        display_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt

        if level == "strong":
            return (
                f"Strong prompt adherence (score: {score_100:.1f}/100). "
                f"The audio closely matches the prompt: '{display_prompt}'"
            )
        elif level == "moderate":
            return (
                f"Moderate prompt adherence (score: {score_100:.1f}/100). "
                f"The audio partially matches the prompt but may be missing some elements."
            )
        else:
            return (
                f"Poor prompt adherence (score: {score_100:.1f}/100). "
                f"The audio does not adequately match the prompt. "
                f"Consider revising the generation or prompt."
            )
