"""Configuration dataclasses for Prompt Adherence dimension."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CLAPEncoderConfig:
    """Configuration for CLAP encoder.

    Attributes:
        model_name: HuggingFace model name or path to load.
        sample_rate: Expected audio sample rate (48kHz for CLAP).
        embedding_dim: Output embedding dimension.
    """

    model_name: str = "laion/larger_clap_music"
    sample_rate: int = 48000
    embedding_dim: int = 512


@dataclass(frozen=True)
class PromptAdherenceConfig:
    """Configuration for Prompt Adherence evaluation.

    Attributes:
        encoder_config: Configuration for the CLAP encoder.
        strong_adherence_threshold: Score above which adherence is "strong".
        moderate_adherence_threshold: Score above which adherence is "moderate".
        cache_text_embeddings: Whether to cache text embeddings.
    """

    encoder_config: CLAPEncoderConfig = field(default_factory=CLAPEncoderConfig)
    strong_adherence_threshold: float = 0.7
    moderate_adherence_threshold: float = 0.5
    cache_text_embeddings: bool = True
