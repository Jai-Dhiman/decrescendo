"""Inference pipeline for the Output Classifier."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from .audio_preprocessing import AudioPreprocessor
from .config import HARM_CATEGORY_NAMES, HarmCategory, OutputClassifierConfig
from .model import (
    OutputClassifierModel,
    compare_against_protected_voices,
    compute_speaker_similarity,
)


class Decision(Enum):
    """Classification decision for audio content."""

    CONTINUE = "CONTINUE"  # Safe to continue generation
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"  # Needs human review
    BLOCK = "BLOCK"  # Block/stop generation


@dataclass
class SpeakerMatch:
    """Result of speaker matching against protected voices."""

    matched: bool
    similarity: float
    matched_voice_id: int | None = None
    matched_voice_name: str | None = None


@dataclass
class AudioClassificationResult:
    """Classification result for a single audio chunk."""

    # Harm scores (0-1 probability for each category)
    harm_scores: dict[str, float]

    # Flagged categories (above threshold)
    flagged_categories: list[str]

    # Speaker matching
    speaker_match: SpeakerMatch

    # Decision for this chunk
    chunk_decision: Decision

    # Raw outputs
    audio_embedding: jnp.ndarray | None = None
    speaker_embedding: jnp.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "harm_scores": self.harm_scores,
            "flagged_categories": self.flagged_categories,
            "speaker_match": {
                "matched": self.speaker_match.matched,
                "similarity": self.speaker_match.similarity,
                "matched_voice_id": self.speaker_match.matched_voice_id,
            },
            "chunk_decision": self.chunk_decision.value,
        }


@dataclass
class AggregatedResult:
    """Aggregated classification result across multiple chunks."""

    # Aggregated harm scores (max across chunks)
    harm_scores: dict[str, float]

    # All flagged categories
    flagged_categories: list[str]

    # Best speaker match across chunks
    best_speaker_match: SpeakerMatch

    # Final decision
    decision: Decision
    decision_reasons: list[str]

    # Number of chunks processed
    num_chunks: int

    # Per-chunk results
    chunk_results: list[AudioClassificationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "harm_scores": self.harm_scores,
            "flagged_categories": self.flagged_categories,
            "speaker_match": {
                "matched": self.best_speaker_match.matched,
                "similarity": self.best_speaker_match.similarity,
            },
            "decision": self.decision.value,
            "decision_reasons": self.decision_reasons,
            "num_chunks": self.num_chunks,
        }


class ScoreAggregator:
    """Aggregates scores across audio chunks with exponential decay."""

    def __init__(
        self,
        num_categories: int = 7,
        decay_factor: float = 0.9,
        window_size: int = 10,
    ) -> None:
        """Initialize aggregator.

        Args:
            num_categories: Number of harm categories
            decay_factor: Exponential decay for older scores
            window_size: Maximum number of chunks to track
        """
        self.num_categories = num_categories
        self.decay_factor = decay_factor
        self.window_size = window_size
        self.score_history: list[np.ndarray] = []

    def add_scores(self, scores: np.ndarray) -> None:
        """Add new chunk scores.

        Args:
            scores: Harm scores for current chunk (num_categories,)
        """
        self.score_history.append(scores)
        if len(self.score_history) > self.window_size:
            self.score_history.pop(0)

    def get_aggregated_scores(self) -> np.ndarray:
        """Get aggregated scores with exponential decay.

        More recent chunks have higher weight.

        Returns:
            Aggregated scores (num_categories,)
        """
        if not self.score_history:
            return np.zeros(self.num_categories)

        n = len(self.score_history)
        weights = np.array([self.decay_factor ** (n - 1 - i) for i in range(n)])
        weights = weights / weights.sum()

        weighted_scores = np.zeros(self.num_categories)
        for i, scores in enumerate(self.score_history):
            weighted_scores += weights[i] * scores

        return weighted_scores

    def get_max_scores(self) -> np.ndarray:
        """Get maximum scores across all chunks.

        Returns:
            Max scores (num_categories,)
        """
        if not self.score_history:
            return np.zeros(self.num_categories)

        return np.max(np.stack(self.score_history), axis=0)

    def reset(self) -> None:
        """Reset the aggregator."""
        self.score_history = []


class OutputClassifierInference:
    """Inference pipeline for Constitutional Audio Output Classifier.

    Analyzes audio content for:
    - Harm category classification (7 categories)
    - Speaker matching against protected voices
    - Streaming aggregation for real-time decisions

    Example:
        >>> config = OutputClassifierConfig()
        >>> classifier = OutputClassifierInference(model, params, config)
        >>>
        >>> # Classify a file
        >>> result = classifier.classify_file("audio.wav")
        >>> print(result.decision)
        >>>
        >>> # Streaming classification
        >>> for chunk_result in classifier.classify_stream(audio_chunks):
        ...     if chunk_result.chunk_decision == Decision.BLOCK:
        ...         break
    """

    def __init__(
        self,
        model: OutputClassifierModel,
        variables: dict[str, Any],
        config: OutputClassifierConfig | None = None,
        protected_voices: jnp.ndarray | None = None,
        protected_voice_names: list[str] | None = None,
    ) -> None:
        """Initialize inference pipeline.

        Args:
            model: OutputClassifierModel instance
            variables: Model variables containing 'params' and 'batch_stats'
            config: Output classifier configuration
            protected_voices: Protected voice embeddings (num_voices, speaker_dim)
            protected_voice_names: Names of protected voices
        """
        self.model = model
        self.variables = variables
        self.config = config or OutputClassifierConfig()

        self.protected_voices = protected_voices
        self.protected_voice_names = protected_voice_names or []

        # Preprocessor
        self.preprocessor = AudioPreprocessor(self.config.preprocessing)

        # JIT compile forward pass
        self._forward = jax.jit(self._forward_fn)

        # Score aggregator for streaming
        self.aggregator = ScoreAggregator(
            num_categories=self.config.num_harm_categories,
            decay_factor=self.config.exponential_decay,
            window_size=self.config.aggregation_window,
        )

    def _forward_fn(self, audio: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """JIT-compiled forward pass."""
        return self.model.apply(
            self.variables,
            audio,
            train=False,
        )

    def classify_chunk(
        self,
        audio_chunk: jnp.ndarray,
        include_embeddings: bool = False,
    ) -> AudioClassificationResult:
        """Classify a single audio chunk.

        Args:
            audio_chunk: Audio chunk of shape (samples,)
            include_embeddings: Whether to include raw embeddings in result

        Returns:
            AudioClassificationResult for this chunk
        """
        # Add batch dimension
        audio_batch = audio_chunk[None, :]

        # Forward pass
        outputs = self._forward(audio_batch)

        # Extract outputs (remove batch dim)
        harm_logits = outputs["harm_logits"][0]
        speaker_embedding = outputs["speaker_embeddings"][0]
        audio_embedding = outputs["audio_embeddings"][0]

        # Convert to probabilities
        harm_probs = jax.nn.sigmoid(harm_logits)
        harm_scores = {name: float(harm_probs[i]) for i, name in enumerate(HARM_CATEGORY_NAMES)}

        # Check for flagged categories
        flagged = [
            name for name, score in harm_scores.items() if score > self.config.flag_threshold
        ]

        # Speaker matching
        speaker_match = self._check_speaker_match(speaker_embedding)

        # Determine chunk decision
        chunk_decision = self._make_chunk_decision(harm_scores, speaker_match)

        return AudioClassificationResult(
            harm_scores=harm_scores,
            flagged_categories=flagged,
            speaker_match=speaker_match,
            chunk_decision=chunk_decision,
            audio_embedding=audio_embedding if include_embeddings else None,
            speaker_embedding=speaker_embedding if include_embeddings else None,
        )

    def _check_speaker_match(
        self,
        speaker_embedding: jnp.ndarray,
    ) -> SpeakerMatch:
        """Check if speaker matches any protected voice.

        Args:
            speaker_embedding: Speaker embedding (speaker_dim,)

        Returns:
            SpeakerMatch result
        """
        if self.protected_voices is None or len(self.protected_voices) == 0:
            return SpeakerMatch(matched=False, similarity=0.0)

        max_sim, best_idx = compare_against_protected_voices(
            speaker_embedding,
            self.protected_voices,
        )

        similarity = float(max_sim[0])
        idx = int(best_idx[0])

        matched = similarity >= self.config.speaker.match_threshold
        voice_name = (
            self.protected_voice_names[idx]
            if matched and idx < len(self.protected_voice_names)
            else None
        )

        return SpeakerMatch(
            matched=matched,
            similarity=similarity,
            matched_voice_id=idx if matched else None,
            matched_voice_name=voice_name,
        )

    def _make_chunk_decision(
        self,
        harm_scores: dict[str, float],
        speaker_match: SpeakerMatch,
    ) -> Decision:
        """Make decision for a single chunk.

        Args:
            harm_scores: Harm scores for this chunk
            speaker_match: Speaker matching result

        Returns:
            Decision for this chunk
        """
        # Check for blocking conditions
        max_harm = max(harm_scores.values())
        if max_harm >= self.config.block_threshold:
            return Decision.BLOCK

        if speaker_match.matched:
            return Decision.BLOCK

        # Check for flagging conditions
        if max_harm >= self.config.flag_threshold:
            return Decision.FLAG_FOR_REVIEW

        if speaker_match.similarity >= self.config.speaker.flag_threshold:
            return Decision.FLAG_FOR_REVIEW

        return Decision.CONTINUE

    def classify_stream(
        self,
        audio_chunks: Iterator[jnp.ndarray],
    ) -> Iterator[AudioClassificationResult]:
        """Classify audio stream chunk by chunk.

        Args:
            audio_chunks: Iterator of audio chunks

        Yields:
            AudioClassificationResult for each chunk
        """
        self.aggregator.reset()

        for chunk in audio_chunks:
            result = self.classify_chunk(chunk)

            # Add to aggregator
            scores_array = np.array([result.harm_scores[name] for name in HARM_CATEGORY_NAMES])
            self.aggregator.add_scores(scores_array)

            yield result

    def classify_file(
        self,
        path: Path | str,
    ) -> AggregatedResult:
        """Classify an audio file.

        Args:
            path: Path to audio file

        Returns:
            AggregatedResult with final decision
        """
        self.aggregator.reset()
        chunk_results: list[AudioClassificationResult] = []
        best_speaker_match = SpeakerMatch(matched=False, similarity=0.0)

        for chunk in self.preprocessor.process_file(path):
            result = self.classify_chunk(chunk)
            chunk_results.append(result)

            # Track best speaker match
            if result.speaker_match.similarity > best_speaker_match.similarity:
                best_speaker_match = result.speaker_match

            # Add to aggregator
            scores_array = np.array([result.harm_scores[name] for name in HARM_CATEGORY_NAMES])
            self.aggregator.add_scores(scores_array)

        return self._aggregate_results(chunk_results, best_speaker_match)

    def classify_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> AggregatedResult:
        """Classify an audio array.

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            AggregatedResult with final decision
        """
        self.aggregator.reset()
        chunk_results: list[AudioClassificationResult] = []
        best_speaker_match = SpeakerMatch(matched=False, similarity=0.0)

        for chunk in self.preprocessor.process_array(audio, sample_rate):
            result = self.classify_chunk(chunk)
            chunk_results.append(result)

            if result.speaker_match.similarity > best_speaker_match.similarity:
                best_speaker_match = result.speaker_match

            scores_array = np.array([result.harm_scores[name] for name in HARM_CATEGORY_NAMES])
            self.aggregator.add_scores(scores_array)

        return self._aggregate_results(chunk_results, best_speaker_match)

    def _aggregate_results(
        self,
        chunk_results: list[AudioClassificationResult],
        best_speaker_match: SpeakerMatch,
    ) -> AggregatedResult:
        """Aggregate results from all chunks.

        Args:
            chunk_results: Results from each chunk
            best_speaker_match: Best speaker match across chunks

        Returns:
            AggregatedResult
        """
        if not chunk_results:
            return AggregatedResult(
                harm_scores={name: 0.0 for name in HARM_CATEGORY_NAMES},
                flagged_categories=[],
                best_speaker_match=SpeakerMatch(matched=False, similarity=0.0),
                decision=Decision.CONTINUE,
                decision_reasons=["No audio to classify"],
                num_chunks=0,
                chunk_results=[],
            )

        # Get max scores across all chunks
        max_scores = self.aggregator.get_max_scores()
        harm_scores = {name: float(max_scores[i]) for i, name in enumerate(HARM_CATEGORY_NAMES)}

        # Collect all flagged categories
        flagged = list(set(cat for result in chunk_results for cat in result.flagged_categories))

        # Make final decision
        decision, reasons = self._make_final_decision(
            harm_scores, best_speaker_match, chunk_results
        )

        return AggregatedResult(
            harm_scores=harm_scores,
            flagged_categories=flagged,
            best_speaker_match=best_speaker_match,
            decision=decision,
            decision_reasons=reasons,
            num_chunks=len(chunk_results),
            chunk_results=chunk_results,
        )

    def _make_final_decision(
        self,
        harm_scores: dict[str, float],
        speaker_match: SpeakerMatch,
        chunk_results: list[AudioClassificationResult],
    ) -> tuple[Decision, list[str]]:
        """Make final decision based on aggregated results.

        Args:
            harm_scores: Aggregated harm scores
            speaker_match: Best speaker match
            chunk_results: All chunk results

        Returns:
            Tuple of (Decision, list of reasons)
        """
        reasons: list[str] = []

        # Check for any blocking chunks
        blocking_chunks = [
            i for i, r in enumerate(chunk_results) if r.chunk_decision == Decision.BLOCK
        ]
        if blocking_chunks:
            reasons.append(f"Blocking content detected in chunks: {blocking_chunks}")
            return Decision.BLOCK, reasons

        # Check harm scores
        max_harm = max(harm_scores.values())
        if max_harm >= self.config.block_threshold:
            high_categories = [
                name for name, score in harm_scores.items() if score >= self.config.block_threshold
            ]
            reasons.append(f"High harm scores in categories: {high_categories}")
            return Decision.BLOCK, reasons

        # Check speaker match
        if speaker_match.matched:
            reasons.append(
                f"Protected voice detected: {speaker_match.matched_voice_name or 'Unknown'} "
                f"(similarity: {speaker_match.similarity:.2f})"
            )
            return Decision.BLOCK, reasons

        # Check for flagging
        if max_harm >= self.config.flag_threshold:
            flagged_categories = [
                name for name, score in harm_scores.items() if score >= self.config.flag_threshold
            ]
            reasons.append(f"Elevated harm scores in: {flagged_categories}")
            return Decision.FLAG_FOR_REVIEW, reasons

        if speaker_match.similarity >= self.config.speaker.flag_threshold:
            reasons.append(
                f"Possible voice similarity detected (similarity: {speaker_match.similarity:.2f})"
            )
            return Decision.FLAG_FOR_REVIEW, reasons

        reasons.append("No safety concerns detected")
        return Decision.CONTINUE, reasons


def initialize_output_classifier(
    config: OutputClassifierConfig | None = None,
    rng: jax.Array | None = None,
) -> tuple[OutputClassifierModel, dict[str, Any]]:
    """Initialize output classifier with random weights.

    Args:
        config: Model configuration
        rng: JAX random key (default: random seed 42)

    Returns:
        Tuple of (model, variables) where variables contains 'params' and 'batch_stats'
    """
    if config is None:
        config = OutputClassifierConfig()

    if rng is None:
        rng = jax.random.PRNGKey(42)

    model = OutputClassifierModel(config=config)

    # Initialize with dummy input (use train=True to initialize batch_stats)
    dummy_audio = jnp.zeros((1, config.preprocessing.chunk_samples))
    variables = model.init(rng, dummy_audio, train=True)

    return model, variables
