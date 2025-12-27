"""Musicality dimension for MusiCritic.

This module evaluates expressive and aesthetic qualities of AI-generated music:
- TIS (Tonal Interval Space): Harmonic complexity, momentum, tension
- Tension-Resolution: Musical narrative and phrase structure
- Expression: Dynamic variation, crescendo/decrescendo patterns

Example:
    >>> from decrescendo.musicritic.dimensions.musicality import MusicalityEvaluator
    >>> evaluator = MusicalityEvaluator()
    >>> result = evaluator.evaluate(audio, sample_rate=22050)
    >>> print(f"Musicality: {result.scaled_score:.1f}/100")
"""

from .config import (
    ExpressionConfig,
    MusicalityConfig,
    TensionConfig,
    TISConfig,
)
from .evaluator import MusicalityEvaluator
from .exceptions import (
    AudioTooShortError,
    ExpressionAnalysisError,
    MusicalityError,
    TensionAnalysisError,
    TISAnalysisError,
)
from .expression import ExpressionAnalyzer, ExpressionReport
from .tension import TensionAnalyzer, TensionReport
from .tis import TISAnalyzer, TISReport

__all__ = [
    # Evaluator
    "MusicalityEvaluator",
    # Config
    "MusicalityConfig",
    "TISConfig",
    "TensionConfig",
    "ExpressionConfig",
    # Analyzers
    "TISAnalyzer",
    "TensionAnalyzer",
    "ExpressionAnalyzer",
    # Reports
    "TISReport",
    "TensionReport",
    "ExpressionReport",
    # Exceptions
    "MusicalityError",
    "AudioTooShortError",
    "TISAnalysisError",
    "TensionAnalysisError",
    "ExpressionAnalysisError",
]
