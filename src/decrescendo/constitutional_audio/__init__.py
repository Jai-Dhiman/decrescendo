"""Constitutional Audio: Safety framework for audio generation systems."""

from .pipeline import (
    ConstitutionalAudio,
    PipelineConfig,
    PipelineDecision,
    PromptClassificationResult,
    PipelineAudioResult,
    GenerationClassificationResult,
    PipelineError,
    PipelineConfigError,
    ClassifierNotEnabledError,
    load_constitutional_audio,
)

__all__ = [
    # Main class
    "ConstitutionalAudio",
    # Config and results
    "PipelineConfig",
    "PipelineDecision",
    "PromptClassificationResult",
    "PipelineAudioResult",
    "GenerationClassificationResult",
    # Factory functions
    "load_constitutional_audio",
    # Exceptions
    "PipelineError",
    "PipelineConfigError",
    "ClassifierNotEnabledError",
]
