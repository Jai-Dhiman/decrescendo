"""Inference pipeline for Input Classifier."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp

from .config import ARTIST_LABELS, INTENT_LABELS, POLICY_LABELS, VOICE_LABELS


class Intent(Enum):
    """Intent classification result."""

    BENIGN = 0
    SUSPICIOUS = 1
    MALICIOUS = 2


class ArtistRequest(Enum):
    """Artist request classification result."""

    NONE = 0
    NAMED_ARTIST = 1
    STYLE_REFERENCE = 2


class VoiceRequest(Enum):
    """Voice request classification result."""

    NONE = 0
    CELEBRITY = 1
    POLITICIAN = 2


class PolicyCategory(Enum):
    """Policy violation categories."""

    COPYRIGHT_IP = 0
    VOICE_CLONING = 1
    CULTURAL = 2
    MISINFORMATION = 3
    EMOTIONAL_MANIPULATION = 4
    CONTENT_SAFETY = 5
    PHYSICAL_SAFETY = 6


class Decision(Enum):
    """Final classification decision."""

    ALLOW = "ALLOW"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    BLOCK = "BLOCK"


@dataclass
class ClassificationResult:
    """Result of prompt classification.

    Contains predictions from all classification heads and
    a final decision recommendation.
    """

    # Intent classification
    intent: Intent
    intent_confidence: float
    intent_probabilities: dict[str, float]

    # Artist request detection
    artist_request: ArtistRequest
    artist_confidence: float
    artist_probabilities: dict[str, float]

    # Voice request detection
    voice_request: VoiceRequest
    voice_confidence: float
    voice_probabilities: dict[str, float]

    # Policy violations (multi-label)
    policy_violations: dict[str, float]
    policy_flags: list[str]  # Categories above threshold

    # Final decision
    decision: Decision
    decision_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "intent": {
                "label": self.intent.name,
                "confidence": self.intent_confidence,
                "probabilities": self.intent_probabilities,
            },
            "artist_request": {
                "label": self.artist_request.name,
                "confidence": self.artist_confidence,
                "probabilities": self.artist_probabilities,
            },
            "voice_request": {
                "label": self.voice_request.name,
                "confidence": self.voice_confidence,
                "probabilities": self.voice_probabilities,
            },
            "policy_violations": {
                "scores": self.policy_violations,
                "flagged": self.policy_flags,
            },
            "decision": {
                "action": self.decision.value,
                "reasons": self.decision_reasons,
            },
        }


@dataclass
class InferenceConfig:
    """Configuration for inference decision thresholds."""

    # Intent thresholds
    malicious_threshold: float = 0.7
    suspicious_threshold: float = 0.5

    # Policy violation threshold
    policy_violation_threshold: float = 0.5

    # Maximum sequence length
    max_length: int = 512


class InputClassifierInference:
    """Inference pipeline for Constitutional Audio Input Classifier.

    Provides classification of text prompts with:
    - Intent detection (benign/suspicious/malicious)
    - Artist/voice request detection
    - Policy violation prediction
    - Decision recommendation (ALLOW/FLAG_FOR_REVIEW/BLOCK)

    Example:
        >>> from decrescendo.constitutional_audio.input_classifier import (
        ...     InputClassifier, InputClassifierConfig
        ... )
        >>> from decrescendo.constitutional_audio.input_classifier.pretrained import (
        ...     initialize_from_pretrained
        ... )
        >>> from decrescendo.constitutional_audio.input_classifier.inference import (
        ...     InputClassifierInference
        ... )
        >>>
        >>> config = InputClassifierConfig()
        >>> model, params, tokenizer = initialize_from_pretrained(config, rng)
        >>> classifier = InputClassifierInference(model, params, tokenizer)
        >>> result = classifier.classify("Generate music in Drake's style")
        >>> print(result.decision)
    """

    def __init__(
        self,
        model: Any,
        params: dict[str, Any],
        tokenizer: Any,
        config: InferenceConfig | None = None,
    ) -> None:
        """Initialize inference pipeline.

        Args:
            model: Flax InputClassifier model
            params: Model parameters
            tokenizer: HuggingFace tokenizer
            config: Inference configuration (optional)
        """
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()

        # JIT compile the forward pass
        self._forward = jax.jit(self._forward_fn)

    def _forward_fn(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """JIT-compiled forward pass."""
        return self.model.apply(
            {"params": self.params},
            input_ids=input_ids,
            attention_mask=attention_mask,
            deterministic=True,
        )

    def classify(self, prompt: str) -> ClassificationResult:
        """Classify a single prompt.

        Args:
            prompt: Text prompt to classify

        Returns:
            ClassificationResult with all predictions and decision
        """
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="np",
        )

        input_ids = jnp.array(encoding["input_ids"])
        attention_mask = jnp.array(encoding["attention_mask"])

        # Forward pass
        outputs = self._forward(input_ids, attention_mask)

        # Process outputs
        return self._process_outputs(outputs)

    def classify_batch(self, prompts: list[str]) -> list[ClassificationResult]:
        """Classify multiple prompts in a batch.

        Args:
            prompts: List of text prompts to classify

        Returns:
            List of ClassificationResult for each prompt
        """
        # Tokenize all prompts
        encodings = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="np",
        )

        input_ids = jnp.array(encodings["input_ids"])
        attention_mask = jnp.array(encodings["attention_mask"])

        # Forward pass
        outputs = self._forward(input_ids, attention_mask)

        # Process each sample
        results = []
        for i in range(len(prompts)):
            sample_outputs = {k: v[i : i + 1] for k, v in outputs.items()}
            results.append(self._process_outputs(sample_outputs))

        return results

    def _process_outputs(
        self,
        outputs: dict[str, jnp.ndarray],
    ) -> ClassificationResult:
        """Process model outputs into ClassificationResult."""
        # Intent
        intent_probs = jax.nn.softmax(outputs["intent_logits"][0])
        intent_idx = int(jnp.argmax(intent_probs))
        intent = Intent(intent_idx)
        intent_confidence = float(intent_probs[intent_idx])
        intent_probabilities = {
            label: float(intent_probs[i]) for i, label in enumerate(INTENT_LABELS)
        }

        # Artist request
        artist_probs = jax.nn.softmax(outputs["artist_logits"][0])
        artist_idx = int(jnp.argmax(artist_probs))
        artist_request = ArtistRequest(artist_idx)
        artist_confidence = float(artist_probs[artist_idx])
        artist_probabilities = {
            label: float(artist_probs[i]) for i, label in enumerate(ARTIST_LABELS)
        }

        # Voice request
        voice_probs = jax.nn.softmax(outputs["voice_logits"][0])
        voice_idx = int(jnp.argmax(voice_probs))
        voice_request = VoiceRequest(voice_idx)
        voice_confidence = float(voice_probs[voice_idx])
        voice_probabilities = {
            label: float(voice_probs[i]) for i, label in enumerate(VOICE_LABELS)
        }

        # Policy violations (sigmoid for multi-label)
        policy_probs = jax.nn.sigmoid(outputs["policy_logits"][0])
        policy_violations = {
            label: float(policy_probs[i]) for i, label in enumerate(POLICY_LABELS)
        }
        policy_flags = [
            label
            for label, score in policy_violations.items()
            if score > self.config.policy_violation_threshold
        ]

        # Make decision
        decision, reasons = self._make_decision(
            intent,
            intent_confidence,
            artist_request,
            voice_request,
            policy_violations,
            policy_flags,
        )

        return ClassificationResult(
            intent=intent,
            intent_confidence=intent_confidence,
            intent_probabilities=intent_probabilities,
            artist_request=artist_request,
            artist_confidence=artist_confidence,
            artist_probabilities=artist_probabilities,
            voice_request=voice_request,
            voice_confidence=voice_confidence,
            voice_probabilities=voice_probabilities,
            policy_violations=policy_violations,
            policy_flags=policy_flags,
            decision=decision,
            decision_reasons=reasons,
        )

    def _make_decision(
        self,
        intent: Intent,
        intent_confidence: float,
        artist_request: ArtistRequest,
        voice_request: VoiceRequest,
        policy_violations: dict[str, float],
        policy_flags: list[str],
    ) -> tuple[Decision, list[str]]:
        """Determine action based on classification results.

        Returns:
            Tuple of (Decision, list of reasons)
        """
        reasons: list[str] = []

        # Check for blocks first
        if intent == Intent.MALICIOUS and intent_confidence > self.config.malicious_threshold:
            reasons.append(f"Malicious intent detected (confidence: {intent_confidence:.2f})")
            return Decision.BLOCK, reasons

        # Check policy violations
        if policy_flags:
            max_policy_score = max(policy_violations[flag] for flag in policy_flags)
            if max_policy_score > self.config.policy_violation_threshold:
                reasons.append(f"Policy violations detected: {', '.join(policy_flags)}")
                return Decision.BLOCK, reasons

        # Check for review conditions
        if intent == Intent.SUSPICIOUS and intent_confidence > self.config.suspicious_threshold:
            reasons.append(f"Suspicious intent detected (confidence: {intent_confidence:.2f})")

        if artist_request == ArtistRequest.NAMED_ARTIST:
            reasons.append("Named artist reference detected")

        if voice_request == VoiceRequest.CELEBRITY:
            reasons.append("Celebrity voice request detected")
        elif voice_request == VoiceRequest.POLITICIAN:
            reasons.append("Politician voice request detected")

        # Return decision
        if reasons:
            return Decision.FLAG_FOR_REVIEW, reasons

        return Decision.ALLOW, ["No safety concerns detected"]
