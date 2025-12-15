"""Configuration for the Output Classifier."""

from dataclasses import dataclass, field
from enum import Enum


class HarmCategory(Enum):
    """Categories of potential harm in audio content."""

    COPYRIGHT_IP = 0  # Copyright and intellectual property violations
    VOICE_CLONING = 1  # Unauthorized voice cloning
    CULTURAL = 2  # Cultural appropriation or sacred content
    MISINFORMATION = 3  # Synthetic speech for misinformation
    EMOTIONAL_MANIPULATION = 4  # Subliminal or manipulative patterns
    CONTENT_SAFETY = 5  # Hate speech, harmful instructions
    PHYSICAL_SAFETY = 6  # Harmful frequencies, volume spikes


HARM_CATEGORY_NAMES = [
    "copyright_ip",
    "voice_cloning",
    "cultural",
    "misinformation",
    "emotional_manipulation",
    "content_safety",
    "physical_safety",
]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for audio preprocessing.

    Matches the design doc specifications for standardized preprocessing.
    """

    # Target sample rate for processing
    sample_rate: int = 24000  # 24kHz for MERT-style processing

    # Chunk settings
    chunk_duration_sec: float = 1.0  # 1-second chunks for streaming
    hop_duration_sec: float = 0.5  # 50% overlap

    # Normalization
    normalize_audio: bool = True
    target_db: float = -20.0  # Target RMS level in dB

    # Channel handling
    mono: bool = True  # Convert to mono

    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk."""
        return int(self.chunk_duration_sec * self.sample_rate)

    @property
    def hop_samples(self) -> int:
        """Number of samples to hop between chunks."""
        return int(self.hop_duration_sec * self.sample_rate)


@dataclass(frozen=True)
class AudioEncoderConfig:
    """Configuration for the audio encoder.

    A CNN-based encoder that extracts embeddings from audio chunks.
    Can be replaced with pretrained encoders (MERT, WavLM) when available.
    """

    # Input settings
    input_samples: int = 24000  # 1 second at 24kHz

    # CNN architecture
    num_conv_layers: int = 6
    base_channels: int = 64
    channel_multiplier: int = 2  # Channels double every 2 layers
    kernel_size: int = 7
    stride: int = 2

    # Output embedding
    embedding_dim: int = 512

    # Regularization
    dropout_rate: float = 0.1


@dataclass(frozen=True)
class SpeakerConfig:
    """Configuration for speaker embedding and comparison."""

    # Speaker embedding dimension (matches ECAPA-TDNN style)
    embedding_dim: int = 192

    # Similarity thresholds
    match_threshold: float = 0.85  # Cosine similarity for positive match
    flag_threshold: float = 0.70  # Similarity for flagging review

    # Encoder settings (simplified speaker encoder)
    num_conv_layers: int = 4
    base_channels: int = 32


@dataclass(frozen=True)
class OutputClassifierConfig:
    """Full configuration for the Output Classifier."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    audio_encoder: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)

    # Number of harm categories
    num_harm_categories: int = 7

    # Classification head settings
    classifier_hidden_dim: int = 256
    classifier_dropout: float = 0.1

    # Aggregation settings (for streaming)
    aggregation_window: int = 10  # Number of chunks to aggregate
    exponential_decay: float = 0.9  # Decay factor for older chunks

    # Decision thresholds
    block_threshold: float = 0.8  # Block if harm score exceeds this
    flag_threshold: float = 0.5  # Flag for review if exceeds this
