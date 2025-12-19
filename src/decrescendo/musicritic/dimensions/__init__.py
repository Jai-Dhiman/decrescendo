"""MusiCritic evaluation dimensions.

This module provides the 8-dimension evaluation framework:

Quality Dimensions (1-4):
- prompt_adherence: How well audio matches the text prompt
- musical_coherence: Structure, harmony, rhythm, melody quality
- audio_quality: Artifact detection, loudness, production quality
- musicality: Tension-resolution, expressiveness, genre authenticity

Safety Dimensions (5-8):
- copyright: Originality and plagiarism detection
- voice_cloning: Protected voice detection
- cultural_sensitivity: Cultural appropriation flagging
- content_safety: Harmful content detection
"""

from .base import (
    DimensionCategory,
    DimensionEvaluator,
    DimensionResult,
    DimensionRegistry,
    EvaluationConfig,
    QualityDimension,
    SafetyDimension,
)

__all__ = [
    "DimensionCategory",
    "DimensionEvaluator",
    "DimensionResult",
    "DimensionRegistry",
    "EvaluationConfig",
    "QualityDimension",
    "SafetyDimension",
]
