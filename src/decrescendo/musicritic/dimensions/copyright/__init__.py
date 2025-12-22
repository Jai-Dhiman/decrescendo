"""Copyright & Originality dimension for MusiCritic.

This module evaluates AI-generated audio for potential copyright violations
by detecting similarity to existing music through:
- Audio fingerprinting (Chromaprint)
- Melody similarity (pitch contour analysis)
- Rhythm similarity (onset patterns, tempo)
- Harmony similarity (chroma features)

This is a safety dimension - higher scores indicate more concerning
similarity (potential plagiarism).

Example:
    >>> from decrescendo.musicritic.dimensions.copyright import (
    ...     CopyrightEvaluator,
    ...     CopyrightConfig,
    ... )
    >>> evaluator = CopyrightEvaluator()
    >>> result = evaluator.evaluate(audio, sample_rate=44100)
    >>> print(f"Decision: {result.metadata['decision']}")
"""

from .config import (
    CopyrightConfig,
    FingerprintConfig,
    MelodySimilarityConfig,
    RhythmSimilarityConfig,
)
from .evaluator import CopyrightEvaluator
from .exceptions import (
    AudioTooShortError,
    CopyrightError,
    DatabaseError,
    DatabaseNotFoundError,
    FingerprintError,
    FingerprintNotAvailableError,
    MelodySimilarityError,
    RhythmSimilarityError,
)
from .fingerprint import (
    ChromaprintEncoder,
    FingerprintDatabase,
    FingerprintEntry,
    FingerprintMatch,
    is_chromaprint_available,
)
from .similarity import (
    MelodyExtractor,
    MelodyReport,
    RhythmExtractor,
    RhythmReport,
    SimilarityMatcher,
    SimilarityReport,
)

__all__ = [
    # Config
    "CopyrightConfig",
    "FingerprintConfig",
    "MelodySimilarityConfig",
    "RhythmSimilarityConfig",
    # Evaluator
    "CopyrightEvaluator",
    # Exceptions
    "CopyrightError",
    "AudioTooShortError",
    "FingerprintError",
    "FingerprintNotAvailableError",
    "MelodySimilarityError",
    "RhythmSimilarityError",
    "DatabaseError",
    "DatabaseNotFoundError",
    # Fingerprint
    "ChromaprintEncoder",
    "FingerprintDatabase",
    "FingerprintEntry",
    "FingerprintMatch",
    "is_chromaprint_available",
    # Similarity
    "MelodyExtractor",
    "MelodyReport",
    "RhythmExtractor",
    "RhythmReport",
    "SimilarityMatcher",
    "SimilarityReport",
]
