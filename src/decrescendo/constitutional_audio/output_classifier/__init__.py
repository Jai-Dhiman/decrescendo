"""Output Classifier: Real-time audio safety classification."""

from .checkpointing import (
    CheckpointConfigError,
    CheckpointCorruptedError,
    CheckpointError,
    CheckpointMetadata,
    CheckpointNotFoundError,
    CheckpointVersionError,
    OutputClassifierCheckpointer,
    VoiceDatabaseError,
    VoiceDatabaseNotFoundError,
    VoiceEntry,
    load_output_classifier,
    load_output_classifier_inference,
    load_voice_database,
    save_output_classifier,
    save_voice_database,
)
from .config import (
    AudioEncoderConfig,
    HarmCategory,
    OutputClassifierConfig,
    PreprocessingConfig,
    SpeakerConfig,
)
from .inference import (
    AggregatedResult,
    AudioClassificationResult,
    Decision,
    OutputClassifierInference,
    SpeakerMatch,
)
from .model import AudioEncoder, HarmClassifier, OutputClassifierModel
from .pretrained_audio import (
    HybridAudioClassifier,
    PretrainedAudioConfig,
    PretrainedAudioEncoderWrapper,
    PretrainedAudioLoadError,
    PretrainedAudioProjection,
    create_pretrained_training_setup,
    initialize_hybrid_classifier,
    load_mert_encoder,
    load_wavlm_encoder,
    precompute_embeddings,
)
from .voice_database import (
    SimilarityResult,
    VoiceDatabase,
    VoiceDuplicateError,
    VoiceNotFoundError,
)
from .voice_enrollment import (
    AudioQualityError,
    EnrollmentError,
    EnrollmentResult,
    QualityCheckResult,
    VoiceEnroller,
    create_voice_enroller,
    create_voice_enroller_from_inference,
)

__all__ = [
    # Config
    "OutputClassifierConfig",
    "PreprocessingConfig",
    "AudioEncoderConfig",
    "SpeakerConfig",
    "HarmCategory",
    # Model
    "AudioEncoder",
    "HarmClassifier",
    "OutputClassifierModel",
    # Inference
    "OutputClassifierInference",
    "AudioClassificationResult",
    "SpeakerMatch",
    "AggregatedResult",
    "Decision",
    # Checkpointing
    "OutputClassifierCheckpointer",
    "save_output_classifier",
    "load_output_classifier",
    "load_output_classifier_inference",
    "save_voice_database",
    "load_voice_database",
    "VoiceEntry",
    "CheckpointMetadata",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointVersionError",
    "CheckpointConfigError",
    "CheckpointCorruptedError",
    "VoiceDatabaseError",
    "VoiceDatabaseNotFoundError",
    # Pretrained audio
    "PretrainedAudioConfig",
    "PretrainedAudioLoadError",
    "PretrainedAudioEncoderWrapper",
    "PretrainedAudioProjection",
    "HybridAudioClassifier",
    "load_mert_encoder",
    "load_wavlm_encoder",
    "precompute_embeddings",
    "initialize_hybrid_classifier",
    "create_pretrained_training_setup",
    # Voice database
    "VoiceDatabase",
    "SimilarityResult",
    "VoiceNotFoundError",
    "VoiceDuplicateError",
    # Voice enrollment
    "VoiceEnroller",
    "EnrollmentResult",
    "QualityCheckResult",
    "EnrollmentError",
    "AudioQualityError",
    "create_voice_enroller",
    "create_voice_enroller_from_inference",
]
