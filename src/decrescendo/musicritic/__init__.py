"""MusiCritic: Unified evaluation framework for AI-generated music."""

from .pipeline import (
    ClassifierNotEnabledError,
    ConstitutionalAudio,
    GenerationClassificationResult,
    PipelineAudioResult,
    PipelineConfig,
    PipelineConfigError,
    PipelineDecision,
    PipelineError,
    PromptClassificationResult,
    load_constitutional_audio,
)

# Aliases for new naming (will gradually transition)
MusiCritic = ConstitutionalAudio
load_musicritic = load_constitutional_audio

__all__ = [
    # Main class (new name)
    "MusiCritic",
    # Legacy name (for backwards compatibility during transition)
    "ConstitutionalAudio",
    # Config and results
    "PipelineConfig",
    "PipelineDecision",
    "PromptClassificationResult",
    "PipelineAudioResult",
    "GenerationClassificationResult",
    # Factory functions
    "load_musicritic",
    "load_constitutional_audio",
    # Exceptions
    "PipelineError",
    "PipelineConfigError",
    "ClassifierNotEnabledError",
]
