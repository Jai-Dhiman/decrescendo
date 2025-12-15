"""Unified pipeline for Constitutional Audio safety classification.

Combines the Input Classifier (text prompts) and Output Classifier (audio)
into a single cohesive API with unified decision making.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .input_classifier.inference import (
    ClassificationResult,
    Decision as InputDecision,
    InferenceConfig,
    InputClassifierInference,
)
from .input_classifier.checkpointing import load_input_classifier_inference
from .output_classifier.inference import (
    AggregatedResult,
    Decision as OutputDecision,
    OutputClassifierInference,
)
from .output_classifier.config import OutputClassifierConfig
from .output_classifier.checkpointing import load_output_classifier_inference
from .output_classifier.voice_database import VoiceDatabase


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class PipelineError(Exception):
    """Base exception for pipeline operations."""

    pass


class PipelineConfigError(PipelineError):
    """Raised when pipeline configuration is invalid."""

    pass


class ClassifierNotEnabledError(PipelineError):
    """Raised when attempting to use a disabled classifier."""

    pass


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class PipelineDecision(Enum):
    """Unified decision for the Constitutional Audio pipeline.

    Maps decisions from both classifiers:
    - Input classifier: ALLOW, FLAG_FOR_REVIEW, BLOCK
    - Output classifier: CONTINUE, FLAG_FOR_REVIEW, BLOCK

    ALLOW corresponds to both Input ALLOW and Output CONTINUE.
    """

    ALLOW = "ALLOW"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    BLOCK = "BLOCK"


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the Constitutional Audio pipeline.

    Controls which classifiers are enabled and their configurations.

    Attributes:
        enable_input_classifier: Whether to enable the input (text) classifier.
        enable_output_classifier: Whether to enable the output (audio) classifier.
        enable_voice_matching: Whether to enable protected voice matching.
        input_classifier_config: Configuration for the input classifier.
        output_classifier_config: Configuration for the output classifier.
        voice_database_path: Path to voice database for protected voices.
    """

    enable_input_classifier: bool = True
    enable_output_classifier: bool = True
    enable_voice_matching: bool = True
    input_classifier_config: InferenceConfig | None = None
    output_classifier_config: OutputClassifierConfig | None = None
    voice_database_path: Path | str | None = None

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            PipelineConfigError: If configuration is invalid.
        """
        if not self.enable_input_classifier and not self.enable_output_classifier:
            raise PipelineConfigError(
                "At least one classifier must be enabled"
            )


# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------


@dataclass
class PromptClassificationResult:
    """Result of prompt classification.

    Wraps the input classifier result with a unified decision.

    Attributes:
        input_result: Original result from the input classifier.
        decision: Unified pipeline decision.
    """

    input_result: ClassificationResult
    decision: PipelineDecision

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": "prompt",
            "input_classifier": self.input_result.to_dict(),
            "decision": self.decision.value,
        }


@dataclass
class PipelineAudioResult:
    """Result of audio classification.

    Wraps the output classifier result with a unified decision.

    Attributes:
        output_result: Original aggregated result from the output classifier.
        decision: Unified pipeline decision.
    """

    output_result: AggregatedResult
    decision: PipelineDecision

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": "audio",
            "output_classifier": self.output_result.to_dict(),
            "decision": self.decision.value,
        }


@dataclass
class GenerationClassificationResult:
    """Result of full generation pipeline (prompt + audio).

    Contains results from both classifiers with an aggregated decision.

    Attributes:
        prompt_result: Prompt classification result (None if input classifier disabled).
        audio_result: Audio classification result (None if output classifier disabled).
        decision: Aggregated pipeline decision.
        decision_reasons: List of reasons for the decision.
        prompt_processed: Whether a prompt was processed.
        audio_processed: Whether audio was processed.
    """

    prompt_result: PromptClassificationResult | None
    audio_result: PipelineAudioResult | None
    decision: PipelineDecision
    decision_reasons: list[str]
    prompt_processed: bool
    audio_processed: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": "generation",
            "prompt": self.prompt_result.to_dict() if self.prompt_result else None,
            "audio": self.audio_result.to_dict() if self.audio_result else None,
            "decision": self.decision.value,
            "decision_reasons": self.decision_reasons,
            "prompt_processed": self.prompt_processed,
            "audio_processed": self.audio_processed,
        }


# -----------------------------------------------------------------------------
# Main Pipeline Class
# -----------------------------------------------------------------------------


class ConstitutionalAudio:
    """Unified pipeline for Constitutional Audio safety classification.

    Combines input classifier (text prompts) and output classifier (audio)
    into a single cohesive pipeline with unified decision making.

    Example:
        >>> # Create from checkpoints
        >>> pipeline = load_constitutional_audio(
        ...     input_checkpoint="checkpoints/input",
        ...     output_checkpoint="checkpoints/output",
        ...     voice_database_path="voices/",
        ... )
        >>>
        >>> # Classify a prompt
        >>> result = pipeline.classify_prompt("Generate music like Drake")
        >>> print(result.decision)
        >>>
        >>> # Classify audio
        >>> result = pipeline.classify_audio("output.wav")
        >>> print(result.decision)
        >>>
        >>> # Full generation pipeline
        >>> result = pipeline.classify_generation(
        ...     prompt="Create a pop song",
        ...     audio=audio_array,
        ...     sample_rate=24000,
        ... )
        >>> print(result.decision)

    Attributes:
        config: Pipeline configuration.
        input_classifier: Input classifier inference (or None if disabled).
        output_classifier: Output classifier inference (or None if disabled).
        voice_database: Voice database for protected voices (or None).
    """

    def __init__(
        self,
        input_classifier: InputClassifierInference | None = None,
        output_classifier: OutputClassifierInference | None = None,
        voice_database: VoiceDatabase | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialize Constitutional Audio pipeline.

        Args:
            input_classifier: Pre-instantiated input classifier inference.
            output_classifier: Pre-instantiated output classifier inference.
            voice_database: Voice database for protected voice matching.
            config: Pipeline configuration.

        Raises:
            PipelineConfigError: If configuration is invalid.
        """
        self.config = config or PipelineConfig()
        self._input_classifier = input_classifier
        self._output_classifier = output_classifier
        self._voice_database = voice_database

        # Validate configuration
        self.config.validate()
        self._validate_setup()

    def _validate_setup(self) -> None:
        """Validate that the pipeline is correctly configured.

        Raises:
            PipelineConfigError: If required classifiers are missing.
        """
        if self.config.enable_input_classifier and self._input_classifier is None:
            raise PipelineConfigError(
                "Input classifier is enabled but no classifier was provided"
            )
        if self.config.enable_output_classifier and self._output_classifier is None:
            raise PipelineConfigError(
                "Output classifier is enabled but no classifier was provided"
            )

    @property
    def input_classifier(self) -> InputClassifierInference | None:
        """Get input classifier (None if disabled)."""
        if self.config.enable_input_classifier:
            return self._input_classifier
        return None

    @property
    def output_classifier(self) -> OutputClassifierInference | None:
        """Get output classifier (None if disabled)."""
        if self.config.enable_output_classifier:
            return self._output_classifier
        return None

    @property
    def voice_database(self) -> VoiceDatabase | None:
        """Get voice database (None if voice matching disabled)."""
        if self.config.enable_voice_matching:
            return self._voice_database
        return None

    # -------------------------------------------------------------------------
    # Classification Methods
    # -------------------------------------------------------------------------

    def classify_prompt(self, prompt: str) -> PromptClassificationResult:
        """Classify a text prompt for safety concerns.

        Args:
            prompt: Text prompt to classify.

        Returns:
            PromptClassificationResult with unified decision.

        Raises:
            ClassifierNotEnabledError: If input classifier is disabled.
        """
        if not self.config.enable_input_classifier:
            raise ClassifierNotEnabledError("Input classifier is disabled")

        # Run input classifier
        input_result = self._input_classifier.classify(prompt)

        # Map to unified decision
        decision = self._map_input_decision(input_result.decision)

        return PromptClassificationResult(
            input_result=input_result,
            decision=decision,
        )

    def classify_prompt_batch(
        self,
        prompts: list[str],
    ) -> list[PromptClassificationResult]:
        """Classify multiple prompts in a batch.

        Args:
            prompts: List of text prompts.

        Returns:
            List of PromptClassificationResult.

        Raises:
            ClassifierNotEnabledError: If input classifier is disabled.
        """
        if not self.config.enable_input_classifier:
            raise ClassifierNotEnabledError("Input classifier is disabled")

        input_results = self._input_classifier.classify_batch(prompts)

        return [
            PromptClassificationResult(
                input_result=result,
                decision=self._map_input_decision(result.decision),
            )
            for result in input_results
        ]

    def classify_audio(
        self,
        audio: np.ndarray | str | Path,
        sample_rate: int | None = None,
    ) -> PipelineAudioResult:
        """Classify audio for safety concerns.

        Args:
            audio: Audio array or path to audio file.
            sample_rate: Sample rate (required if audio is array).

        Returns:
            PipelineAudioResult with unified decision.

        Raises:
            ClassifierNotEnabledError: If output classifier is disabled.
            ValueError: If audio is array but sample_rate not provided.
        """
        if not self.config.enable_output_classifier:
            raise ClassifierNotEnabledError("Output classifier is disabled")

        # Handle different input types
        if isinstance(audio, (str, Path)):
            output_result = self._output_classifier.classify_file(audio)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate is required when audio is an array")
            output_result = self._output_classifier.classify_array(audio, sample_rate)

        # Map to unified decision
        decision = self._map_output_decision(output_result.decision)

        return PipelineAudioResult(
            output_result=output_result,
            decision=decision,
        )

    def classify_generation(
        self,
        prompt: str | None = None,
        audio: np.ndarray | str | Path | None = None,
        sample_rate: int | None = None,
    ) -> GenerationClassificationResult:
        """Run full classification pipeline on generated audio.

        Classifies both the prompt (if provided and input classifier enabled)
        and the audio output (if provided and output classifier enabled),
        then aggregates the decisions.

        Args:
            prompt: Text prompt used for generation (optional).
            audio: Generated audio array or path (optional).
            sample_rate: Sample rate (required if audio is array).

        Returns:
            GenerationClassificationResult with aggregated decision.

        Raises:
            PipelineError: If both prompt and audio are None.
            ValueError: If audio is array but sample_rate not provided.
        """
        if prompt is None and audio is None:
            raise PipelineError("At least one of prompt or audio must be provided")

        prompt_result: PromptClassificationResult | None = None
        audio_result: PipelineAudioResult | None = None
        reasons: list[str] = []

        # Classify prompt
        if prompt is not None and self.config.enable_input_classifier:
            prompt_result = self.classify_prompt(prompt)
            if prompt_result.decision == PipelineDecision.BLOCK:
                reasons.extend([
                    f"Prompt blocked: {r}"
                    for r in prompt_result.input_result.decision_reasons
                ])
            elif prompt_result.decision == PipelineDecision.FLAG_FOR_REVIEW:
                reasons.extend([
                    f"Prompt flagged: {r}"
                    for r in prompt_result.input_result.decision_reasons
                ])

        # Classify audio
        if audio is not None and self.config.enable_output_classifier:
            audio_result = self.classify_audio(audio, sample_rate)
            if audio_result.decision == PipelineDecision.BLOCK:
                reasons.extend([
                    f"Audio blocked: {r}"
                    for r in audio_result.output_result.decision_reasons
                ])
            elif audio_result.decision == PipelineDecision.FLAG_FOR_REVIEW:
                reasons.extend([
                    f"Audio flagged: {r}"
                    for r in audio_result.output_result.decision_reasons
                ])

        # Aggregate decisions
        decision = self._aggregate_decisions(prompt_result, audio_result)

        if not reasons:
            reasons = ["No safety concerns detected"]

        return GenerationClassificationResult(
            prompt_result=prompt_result,
            audio_result=audio_result,
            decision=decision,
            decision_reasons=reasons,
            prompt_processed=prompt_result is not None,
            audio_processed=audio_result is not None,
        )

    # -------------------------------------------------------------------------
    # Decision Mapping and Aggregation
    # -------------------------------------------------------------------------

    def _map_input_decision(
        self,
        decision: InputDecision,
    ) -> PipelineDecision:
        """Map input classifier decision to unified decision.

        Args:
            decision: Decision from input classifier.

        Returns:
            Corresponding PipelineDecision.
        """
        mapping = {
            InputDecision.ALLOW: PipelineDecision.ALLOW,
            InputDecision.FLAG_FOR_REVIEW: PipelineDecision.FLAG_FOR_REVIEW,
            InputDecision.BLOCK: PipelineDecision.BLOCK,
        }
        return mapping[decision]

    def _map_output_decision(
        self,
        decision: OutputDecision,
    ) -> PipelineDecision:
        """Map output classifier decision to unified decision.

        Args:
            decision: Decision from output classifier.

        Returns:
            Corresponding PipelineDecision.
        """
        mapping = {
            OutputDecision.CONTINUE: PipelineDecision.ALLOW,
            OutputDecision.FLAG_FOR_REVIEW: PipelineDecision.FLAG_FOR_REVIEW,
            OutputDecision.BLOCK: PipelineDecision.BLOCK,
        }
        return mapping[decision]

    def _aggregate_decisions(
        self,
        prompt_result: PromptClassificationResult | None,
        audio_result: PipelineAudioResult | None,
    ) -> PipelineDecision:
        """Aggregate decisions from prompt and audio classification.

        Uses conservative aggregation: BLOCK > FLAG_FOR_REVIEW > ALLOW.

        Args:
            prompt_result: Result from prompt classification (or None).
            audio_result: Result from audio classification (or None).

        Returns:
            Aggregated PipelineDecision.
        """
        decisions: list[PipelineDecision] = []

        if prompt_result is not None:
            decisions.append(prompt_result.decision)
        if audio_result is not None:
            decisions.append(audio_result.decision)

        if not decisions:
            return PipelineDecision.ALLOW

        # Check for any blocks
        if any(d == PipelineDecision.BLOCK for d in decisions):
            return PipelineDecision.BLOCK

        # Check for any flags
        if any(d == PipelineDecision.FLAG_FOR_REVIEW for d in decisions):
            return PipelineDecision.FLAG_FOR_REVIEW

        return PipelineDecision.ALLOW


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def load_constitutional_audio(
    input_checkpoint: Path | str | None = None,
    output_checkpoint: Path | str | None = None,
    voice_database_path: Path | str | None = None,
    config: PipelineConfig | None = None,
    input_step: int | None = None,
    output_step: int | None = None,
) -> ConstitutionalAudio:
    """Load Constitutional Audio pipeline from checkpoints.

    Convenience function for loading a complete pipeline from disk.

    Args:
        input_checkpoint: Path to input classifier checkpoint directory.
        output_checkpoint: Path to output classifier checkpoint directory.
        voice_database_path: Path to voice database directory.
        config: Pipeline configuration (uses defaults if None).
        input_step: Specific step for input checkpoint (None for latest).
        output_step: Specific step for output checkpoint (None for latest).

    Returns:
        Configured ConstitutionalAudio instance.

    Raises:
        CheckpointNotFoundError: If checkpoint not found.
        CheckpointVersionError: If checkpoint version incompatible.
        PipelineConfigError: If configuration is invalid.

    Example:
        >>> pipeline = load_constitutional_audio(
        ...     input_checkpoint="checkpoints/input",
        ...     output_checkpoint="checkpoints/output",
        ...     voice_database_path="voices/",
        ... )
    """
    config = config or PipelineConfig()

    input_classifier: InputClassifierInference | None = None
    output_classifier: OutputClassifierInference | None = None
    voice_database: VoiceDatabase | None = None

    # Load input classifier
    if input_checkpoint is not None and config.enable_input_classifier:
        input_classifier = load_input_classifier_inference(
            path=input_checkpoint,
            step=input_step,
            inference_config=config.input_classifier_config,
        )

    # Load voice database
    if voice_database_path is not None and config.enable_voice_matching:
        voice_database = VoiceDatabase.load(voice_database_path)

    # Load output classifier with voice database
    if output_checkpoint is not None and config.enable_output_classifier:
        protected_voices = None
        voice_names: list[str] | None = None

        if voice_database is not None:
            protected_voices, voice_names = voice_database.get_all_embeddings()

        output_classifier = load_output_classifier_inference(
            path=output_checkpoint,
            step=output_step,
            protected_voices=protected_voices,
            protected_voice_names=voice_names,
        )

    return ConstitutionalAudio(
        input_classifier=input_classifier,
        output_classifier=output_classifier,
        voice_database=voice_database,
        config=config,
    )
