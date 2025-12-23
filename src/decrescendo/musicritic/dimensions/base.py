"""Base classes for MusiCritic evaluation dimensions.

This module defines the core abstractions for the 8-dimension evaluation framework:
- DimensionEvaluator: Protocol for dimension evaluators
- DimensionResult: Structured result from evaluation
- DimensionRegistry: Registry for managing evaluators
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class DimensionCategory(Enum):
    """Category of evaluation dimension."""

    QUALITY = "quality"
    SAFETY = "safety"


class QualityDimension(Enum):
    """Quality evaluation dimensions (1-4)."""

    PROMPT_ADHERENCE = "prompt_adherence"
    MUSICAL_COHERENCE = "musical_coherence"
    AUDIO_QUALITY = "audio_quality"
    MUSICALITY = "musicality"


class SafetyDimension(Enum):
    """Safety evaluation dimensions (5-8)."""

    COPYRIGHT = "copyright"
    VOICE_CLONING = "voice_cloning"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    CONTENT_SAFETY = "content_safety"


class SafetyDecision(Enum):
    """Safety classification decision."""

    ALLOW = "ALLOW"
    FLAG = "FLAG"
    BLOCK = "BLOCK"


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for dimension evaluation.

    Attributes:
        enabled_quality_dimensions: Which quality dimensions to evaluate.
        enabled_safety_dimensions: Which safety dimensions to evaluate.
        safety_thresholds: Thresholds for safety decisions (dimension -> (flag, block)).
        quality_weights: Weights for quality score aggregation.
        cache_embeddings: Whether to cache computed embeddings.
        batch_size: Batch size for processing multiple samples.
    """

    enabled_quality_dimensions: frozenset[QualityDimension] = field(
        default_factory=lambda: frozenset(QualityDimension)
    )
    enabled_safety_dimensions: frozenset[SafetyDimension] = field(
        default_factory=lambda: frozenset(SafetyDimension)
    )
    safety_thresholds: dict[SafetyDimension, tuple[float, float]] = field(
        default_factory=lambda: {
            SafetyDimension.COPYRIGHT: (0.7, 0.95),
            SafetyDimension.VOICE_CLONING: (0.7, 0.95),
            SafetyDimension.CULTURAL_SENSITIVITY: (0.7, 0.95),
            SafetyDimension.CONTENT_SAFETY: (0.7, 0.95),
        }
    )
    quality_weights: dict[QualityDimension, float] = field(
        default_factory=lambda: {
            QualityDimension.PROMPT_ADHERENCE: 0.25,
            QualityDimension.MUSICAL_COHERENCE: 0.30,
            QualityDimension.AUDIO_QUALITY: 0.25,
            QualityDimension.MUSICALITY: 0.20,
        }
    )
    cache_embeddings: bool = True
    batch_size: int = 1


# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------


@dataclass
class DimensionResult:
    """Result from a dimension evaluation.

    All dimensions return a score between 0.0 and 1.0, where:
    - For quality dimensions: higher is better
    - For safety dimensions: higher means more concerning (more likely to flag/block)

    Attributes:
        dimension: The dimension that was evaluated.
        score: Primary score (0.0-1.0).
        confidence: Confidence in the score (0.0-1.0).
        sub_scores: Breakdown of component scores.
        metadata: Additional dimension-specific data.
        explanation: Human-readable explanation of the result.
        timestamps: Time-aligned scores for temporal analysis.
    """

    dimension: QualityDimension | SafetyDimension
    score: float
    confidence: float = 1.0
    sub_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    timestamps: list[tuple[float, float]] | None = None  # (time_sec, score)

    def __post_init__(self) -> None:
        """Validate score ranges."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "confidence": self.confidence,
            "sub_scores": self.sub_scores,
            "metadata": self.metadata,
            "explanation": self.explanation,
            "timestamps": self.timestamps,
        }

    @property
    def scaled_score(self) -> float:
        """Score scaled to 0-100 range."""
        return self.score * 100


@dataclass
class QualityResult:
    """Aggregated result for quality dimensions.

    Attributes:
        overall_score: Combined quality score (0-100).
        dimension_results: Results for each quality dimension.
        confidence: Overall confidence in the score.
        explanation: Summary explanation.
    """

    overall_score: float
    dimension_results: dict[QualityDimension, DimensionResult]
    confidence: float = 1.0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "dimensions": {
                dim.value: result.to_dict() for dim, result in self.dimension_results.items()
            },
        }


@dataclass
class SafetyResult:
    """Aggregated result for safety dimensions.

    Attributes:
        decision: Overall safety decision (ALLOW/FLAG/BLOCK).
        dimension_results: Results for each safety dimension.
        flags: List of safety flags raised.
        evidence: Evidence for each flag.
    """

    decision: SafetyDecision
    dimension_results: dict[SafetyDimension, DimensionResult]
    flags: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "flags": self.flags,
            "evidence": self.evidence,
            "dimensions": {
                dim.value: result.to_dict() for dim, result in self.dimension_results.items()
            },
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result combining quality and safety.

    Attributes:
        quality: Quality evaluation result.
        safety: Safety evaluation result.
        processing_time_ms: Total processing time in milliseconds.
        audio_duration_sec: Duration of evaluated audio in seconds.
    """

    quality: QualityResult | None
    safety: SafetyResult | None
    processing_time_ms: float = 0.0
    audio_duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "quality": self.quality.to_dict() if self.quality else None,
            "safety": self.safety.to_dict() if self.safety else None,
            "processing_time_ms": self.processing_time_ms,
            "audio_duration_sec": self.audio_duration_sec,
        }


# -----------------------------------------------------------------------------
# Evaluator Protocol
# -----------------------------------------------------------------------------


@runtime_checkable
class DimensionEvaluator(Protocol):
    """Protocol for dimension evaluators.

    Each dimension evaluator implements this protocol to provide:
    - evaluate(): Main evaluation method
    - dimension: The dimension this evaluator handles
    - category: Whether this is a quality or safety dimension

    Example:
        >>> class PromptAdherenceEvaluator:
        ...     dimension = QualityDimension.PROMPT_ADHERENCE
        ...     category = DimensionCategory.QUALITY
        ...
        ...     def evaluate(
        ...         self,
        ...         audio: np.ndarray,
        ...         sample_rate: int,
        ...         prompt: str | None = None,
        ...     ) -> DimensionResult:
        ...         # Compute CLAP similarity
        ...         score = self._compute_clap_similarity(audio, prompt)
        ...         return DimensionResult(
        ...             dimension=self.dimension,
        ...             score=score,
        ...             explanation=f"CLAP similarity: {score:.2f}",
        ...         )
    """

    dimension: QualityDimension | SafetyDimension
    category: DimensionCategory

    def evaluate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> DimensionResult:
        """Evaluate audio on this dimension.

        Args:
            audio: Audio waveform as numpy array (mono, float32, normalized).
            sample_rate: Sample rate of the audio.
            prompt: Optional text prompt for prompt-dependent evaluations.
            **kwargs: Additional dimension-specific arguments.

        Returns:
            DimensionResult with score and metadata.
        """
        ...


# -----------------------------------------------------------------------------
# Abstract Base Class
# -----------------------------------------------------------------------------


class BaseDimensionEvaluator(ABC):
    """Abstract base class for dimension evaluators.

    Provides common functionality for dimension evaluators:
    - Audio preprocessing
    - Caching
    - Error handling

    Subclasses must implement:
    - _evaluate_impl(): Core evaluation logic
    """

    dimension: QualityDimension | SafetyDimension
    category: DimensionCategory

    def __init__(self, cache_embeddings: bool = True) -> None:
        """Initialize evaluator.

        Args:
            cache_embeddings: Whether to cache computed embeddings.
        """
        self._cache_embeddings = cache_embeddings
        self._embedding_cache: dict[str, np.ndarray] = {}

    def evaluate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> DimensionResult:
        """Evaluate audio on this dimension.

        Wraps _evaluate_impl with preprocessing and error handling.

        Args:
            audio: Audio waveform as numpy array.
            sample_rate: Sample rate of the audio.
            prompt: Optional text prompt.
            **kwargs: Additional arguments.

        Returns:
            DimensionResult with score and metadata.

        Raises:
            ValueError: If audio is invalid.
        """
        # Validate input
        if audio.ndim > 2:
            raise ValueError(f"Audio must be 1D or 2D, got {audio.ndim}D")

        # Convert stereo to mono if needed
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        return self._evaluate_impl(audio, sample_rate, prompt, **kwargs)

    @abstractmethod
    def _evaluate_impl(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> DimensionResult:
        """Core evaluation implementation.

        Args:
            audio: Preprocessed audio (mono, float32, normalized).
            sample_rate: Sample rate.
            prompt: Optional text prompt.
            **kwargs: Additional arguments.

        Returns:
            DimensionResult.
        """
        ...

    def _cache_key(self, audio: np.ndarray) -> str:
        """Generate cache key for audio."""
        # Use hash of audio bytes
        return str(hash(audio.tobytes()))


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


class DimensionRegistry:
    """Registry for managing dimension evaluators.

    Allows registration and retrieval of evaluators by dimension.

    Example:
        >>> registry = DimensionRegistry()
        >>> registry.register(PromptAdherenceEvaluator())
        >>> evaluator = registry.get(QualityDimension.PROMPT_ADHERENCE)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._quality_evaluators: dict[QualityDimension, DimensionEvaluator] = {}
        self._safety_evaluators: dict[SafetyDimension, DimensionEvaluator] = {}

    def register(self, evaluator: DimensionEvaluator) -> None:
        """Register an evaluator.

        Args:
            evaluator: Evaluator to register.

        Raises:
            ValueError: If evaluator for dimension already registered.
        """
        if evaluator.category == DimensionCategory.QUALITY:
            dim = evaluator.dimension
            if not isinstance(dim, QualityDimension):
                raise ValueError(f"Quality evaluator must have QualityDimension, got {type(dim)}")
            if dim in self._quality_evaluators:
                raise ValueError(f"Evaluator for {dim.value} already registered")
            self._quality_evaluators[dim] = evaluator
        else:
            dim = evaluator.dimension
            if not isinstance(dim, SafetyDimension):
                raise ValueError(f"Safety evaluator must have SafetyDimension, got {type(dim)}")
            if dim in self._safety_evaluators:
                raise ValueError(f"Evaluator for {dim.value} already registered")
            self._safety_evaluators[dim] = evaluator

    def get(self, dimension: QualityDimension | SafetyDimension) -> DimensionEvaluator | None:
        """Get evaluator for a dimension.

        Args:
            dimension: Dimension to get evaluator for.

        Returns:
            Evaluator or None if not registered.
        """
        if isinstance(dimension, QualityDimension):
            return self._quality_evaluators.get(dimension)
        return self._safety_evaluators.get(dimension)

    def get_quality_evaluators(self) -> dict[QualityDimension, DimensionEvaluator]:
        """Get all registered quality evaluators."""
        return self._quality_evaluators.copy()

    def get_safety_evaluators(self) -> dict[SafetyDimension, DimensionEvaluator]:
        """Get all registered safety evaluators."""
        return self._safety_evaluators.copy()

    def list_registered(self) -> list[QualityDimension | SafetyDimension]:
        """List all registered dimensions."""
        return list(self._quality_evaluators.keys()) + list(self._safety_evaluators.keys())

    def __len__(self) -> int:
        """Number of registered evaluators."""
        return len(self._quality_evaluators) + len(self._safety_evaluators)
