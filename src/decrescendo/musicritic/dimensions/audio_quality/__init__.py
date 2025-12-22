"""Audio Quality dimension evaluator.

This module provides the AudioQualityEvaluator for measuring audio
production quality including artifacts, loudness, and perceptual metrics.
"""

from .artifacts import ArtifactDetector, ArtifactReport
from .config import (
    ArtifactDetectionConfig,
    AudioQualityConfig,
    LoudnessConfig,
    PerceptualConfig,
)
from .evaluator import AudioQualityEvaluator
from .exceptions import (
    ArtifactDetectionError,
    AudioQualityError,
    AudioTooShortError,
    LoudnessAnalysisError,
    PerceptualAnalysisError,
)
from .loudness import LoudnessAnalyzer, LoudnessReport
from .perceptual import PerceptualAnalyzer, PerceptualReport

__all__ = [
    # Evaluator
    "AudioQualityEvaluator",
    # Analyzers
    "ArtifactDetector",
    "LoudnessAnalyzer",
    "PerceptualAnalyzer",
    # Reports
    "ArtifactReport",
    "LoudnessReport",
    "PerceptualReport",
    # Config
    "AudioQualityConfig",
    "ArtifactDetectionConfig",
    "LoudnessConfig",
    "PerceptualConfig",
    # Exceptions
    "AudioQualityError",
    "AudioTooShortError",
    "LoudnessAnalysisError",
    "ArtifactDetectionError",
    "PerceptualAnalysisError",
]
