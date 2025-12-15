"""Configuration dataclasses for the Input Classifier."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the transformer backbone.

    Defaults match RoBERTa-base architecture.
    """

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 1

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        return self.hidden_size // self.num_attention_heads


@dataclass(frozen=True)
class ClassificationConfig:
    """Configuration for classification heads.

    The classifier handles multiple prediction tasks:
    - Intent: Is the prompt benign, suspicious, or malicious?
    - Artist request: Does the prompt request a specific artist's voice/style?
    - Voice request: Does the prompt request a celebrity/politician voice?
    - Policy violations: Multi-label detection of 7 harm categories
    """

    # Intent classification (multi-class)
    num_intent_classes: int = 3  # benign, suspicious, malicious

    # Artist request detection (multi-class)
    num_artist_classes: int = 3  # none, named_artist, style_reference

    # Voice request detection (multi-class)
    num_voice_classes: int = 3  # none, celebrity, politician

    # Policy violation detection (multi-label, 7 harm categories)
    # Categories: copyright, voice_cloning, cultural, misinformation,
    #             emotional_manipulation, content_safety, physical_safety
    num_policy_labels: int = 7

    # Classification head settings
    classifier_dropout: float = 0.1


@dataclass(frozen=True)
class InputClassifierConfig:
    """Full configuration for the Input Classifier model."""

    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)

    # Pretrained model settings
    pretrained_model_name: str = "roberta-base"
    use_pretrained: bool = True


# Intent labels
INTENT_LABELS = ["benign", "suspicious", "malicious"]

# Artist request labels
ARTIST_LABELS = ["none", "named_artist", "style_reference"]

# Voice request labels
VOICE_LABELS = ["none", "celebrity", "politician"]

# Policy violation categories (multi-label)
POLICY_LABELS = [
    "copyright_ip",
    "voice_cloning",
    "cultural",
    "misinformation",
    "emotional_manipulation",
    "content_safety",
    "physical_safety",
]
