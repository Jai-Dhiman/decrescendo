"""Configuration dataclasses for Copyright dimension."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FingerprintConfig:
    """Configuration for audio fingerprinting.

    Attributes:
        min_duration: Minimum audio duration in seconds for fingerprinting.
        target_sample_rate: Sample rate for fingerprint generation (Chromaprint uses 11025).
    """

    min_duration: float = 1.0
    target_sample_rate: int = 11025


@dataclass(frozen=True)
class MelodySimilarityConfig:
    """Configuration for melody similarity analysis.

    Attributes:
        hop_length: Hop length in samples for pitch extraction.
        fmin: Minimum frequency for pitch detection (Hz).
        fmax: Maximum frequency for pitch detection (Hz).
        frame_length: Frame length for pitch extraction.
        target_sample_rate: Sample rate for analysis.
    """

    hop_length: int = 512
    fmin: float = 65.0  # C2
    fmax: float = 2093.0  # C7
    frame_length: int = 2048
    target_sample_rate: int = 22050


@dataclass(frozen=True)
class RhythmSimilarityConfig:
    """Configuration for rhythm similarity analysis.

    Attributes:
        hop_length: Hop length in samples for onset detection.
        target_sample_rate: Sample rate for analysis.
    """

    hop_length: int = 512
    target_sample_rate: int = 22050


@dataclass(frozen=True)
class CopyrightConfig:
    """Configuration for Copyright & Originality evaluation.

    This is a safety dimension - higher scores indicate more concerning
    similarity to existing content (potential plagiarism).

    Attributes:
        fingerprint_config: Configuration for audio fingerprinting.
        melody_config: Configuration for melody similarity.
        rhythm_config: Configuration for rhythm similarity.
        fingerprint_weight: Weight for fingerprint match score.
        melody_weight: Weight for melody similarity (primary signal).
        rhythm_weight: Weight for rhythm similarity.
        harmony_weight: Weight for harmony similarity (low, many songs share progressions).
        flag_threshold: Score above this triggers FLAG decision (human review).
        block_threshold: Score above this triggers BLOCK decision (automatic rejection).
        min_audio_duration: Minimum audio duration for analysis in seconds.
    """

    fingerprint_config: FingerprintConfig = field(default_factory=FingerprintConfig)
    melody_config: MelodySimilarityConfig = field(default_factory=MelodySimilarityConfig)
    rhythm_config: RhythmSimilarityConfig = field(default_factory=RhythmSimilarityConfig)

    # Sub-score weights (should sum to 1.0)
    # Melody + rhythm prioritized per Architecture.md
    fingerprint_weight: float = 0.15
    melody_weight: float = 0.50  # Primary signal
    rhythm_weight: float = 0.25
    harmony_weight: float = 0.10  # Low weight - many songs share progressions

    # Safety decision thresholds
    flag_threshold: float = 0.7
    block_threshold: float = 0.95

    # Minimum audio duration for reliable analysis
    min_audio_duration: float = 1.0
