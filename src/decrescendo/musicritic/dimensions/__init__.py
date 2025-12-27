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
    BaseDimensionEvaluator,
    DimensionCategory,
    DimensionEvaluator,
    DimensionResult,
    DimensionRegistry,
    EvaluationConfig,
    QualityDimension,
    SafetyDimension,
)

# Quality Dimensions
from .prompt_adherence import (
    CLAPEncoder,
    CLAPEncoderConfig,
    PromptAdherenceConfig,
    PromptAdherenceEvaluator,
)
from .musical_coherence import (
    MusicalCoherenceConfig,
    MusicalCoherenceEvaluator,
)
from .audio_quality import (
    AudioQualityConfig,
    AudioQualityEvaluator,
)
from .musicality import (
    MusicalityConfig,
    MusicalityEvaluator,
)

# Safety Dimensions
from .copyright import (
    CopyrightConfig,
    CopyrightEvaluator,
)

__all__ = [
    # Base classes
    "BaseDimensionEvaluator",
    "DimensionCategory",
    "DimensionEvaluator",
    "DimensionResult",
    "DimensionRegistry",
    "EvaluationConfig",
    "QualityDimension",
    "SafetyDimension",
    # Prompt Adherence (Dimension 1)
    "PromptAdherenceEvaluator",
    "PromptAdherenceConfig",
    "CLAPEncoder",
    "CLAPEncoderConfig",
    # Musical Coherence (Dimension 2)
    "MusicalCoherenceEvaluator",
    "MusicalCoherenceConfig",
    # Audio Quality (Dimension 3)
    "AudioQualityEvaluator",
    "AudioQualityConfig",
    # Musicality (Dimension 4)
    "MusicalityEvaluator",
    "MusicalityConfig",
    # Copyright (Dimension 5)
    "CopyrightEvaluator",
    "CopyrightConfig",
]
