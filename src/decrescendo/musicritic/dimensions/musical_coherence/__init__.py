"""Musical Coherence dimension for MusiCritic.

This module provides evaluation of structural and compositional quality
of AI-generated music, measuring:
- Structure: Section detection, repetition patterns
- Harmony: Chord progressions, key consistency
- Rhythm: Beat tracking, tempo stability
- Melody: Pitch coherence, phrase structure
"""

from decrescendo.musicritic.dimensions.musical_coherence.config import (
    HarmonyConfig,
    MelodyConfig,
    MusicalCoherenceConfig,
    RhythmConfig,
    StructureConfig,
)
from decrescendo.musicritic.dimensions.musical_coherence.evaluator import (
    MusicalCoherenceEvaluator,
)
from decrescendo.musicritic.dimensions.musical_coherence.exceptions import (
    AudioTooShortError,
    DependencyNotAvailableError,
    HarmonyAnalysisError,
    MelodyAnalysisError,
    MusicalCoherenceError,
    RhythmAnalysisError,
    StructureAnalysisError,
)
from decrescendo.musicritic.dimensions.musical_coherence.harmony import (
    HarmonyAnalyzer,
    HarmonyReport,
)
from decrescendo.musicritic.dimensions.musical_coherence.melody import (
    MelodyAnalyzer,
    MelodyReport,
)
from decrescendo.musicritic.dimensions.musical_coherence.rhythm import (
    RhythmAnalyzer,
    RhythmReport,
)
from decrescendo.musicritic.dimensions.musical_coherence.structure import (
    StructureAnalyzer,
    StructureReport,
)

__all__ = [
    # Main evaluator
    "MusicalCoherenceEvaluator",
    # Config
    "MusicalCoherenceConfig",
    "StructureConfig",
    "HarmonyConfig",
    "RhythmConfig",
    "MelodyConfig",
    # Analyzers
    "StructureAnalyzer",
    "HarmonyAnalyzer",
    "RhythmAnalyzer",
    "MelodyAnalyzer",
    # Reports
    "StructureReport",
    "HarmonyReport",
    "RhythmReport",
    "MelodyReport",
    # Exceptions
    "MusicalCoherenceError",
    "AudioTooShortError",
    "StructureAnalysisError",
    "HarmonyAnalysisError",
    "RhythmAnalysisError",
    "MelodyAnalysisError",
    "DependencyNotAvailableError",
]
