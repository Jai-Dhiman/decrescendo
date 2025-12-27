"""Configuration dataclasses for Musicality dimension."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TISConfig:
    """Configuration for Tonal Interval Space computation.

    TIS is a geometric representation where chords are mapped to points
    in a 12-dimensional space (one dimension per pitch class), allowing
    computation of harmonic tension and complexity.

    Attributes:
        hop_length: Hop length for chroma computation.
        smoothing_window: Window size for smoothing TIS features.
        min_chroma_energy: Minimum chroma energy to consider frame valid.
    """

    hop_length: int = 2048
    smoothing_window: int = 5
    min_chroma_energy: float = 0.01


@dataclass(frozen=True)
class TensionConfig:
    """Configuration for tension-resolution analysis.

    Attributes:
        resolution_threshold: Minimum tension drop to count as resolution.
        cadence_window_sec: Window size for cadence detection in seconds.
        min_tension_drop: Minimum drop ratio to detect tension resolution.
        smoothing_window: Window for smoothing tension curve.
    """

    resolution_threshold: float = 0.15
    cadence_window_sec: float = 2.0
    min_tension_drop: float = 0.2
    smoothing_window: int = 5


@dataclass(frozen=True)
class ExpressionConfig:
    """Configuration for expression analysis.

    Attributes:
        rms_frame_length: Frame length for RMS computation.
        rms_hop_length: Hop length for RMS computation.
        min_dynamic_range_db: Minimum expected dynamic range in dB.
        max_dynamic_range_db: Maximum expected dynamic range in dB.
    """

    rms_frame_length: int = 2048
    rms_hop_length: int = 512
    min_dynamic_range_db: float = 6.0
    max_dynamic_range_db: float = 40.0


@dataclass(frozen=True)
class MusicalityConfig:
    """Configuration for Musicality evaluation.

    Attributes:
        tis_config: TIS computation configuration.
        tension_config: Tension-resolution configuration.
        expression_config: Expression analysis configuration.
        tis_weight: Weight for TIS score (0-1).
        tension_weight: Weight for tension score (0-1).
        expression_weight: Weight for expression score (0-1).
        excellent_threshold: Score threshold for "excellent".
        good_threshold: Score threshold for "good".
        moderate_threshold: Score threshold for "moderate".
        min_audio_duration: Minimum audio duration in seconds.
    """

    tis_config: TISConfig = field(default_factory=TISConfig)
    tension_config: TensionConfig = field(default_factory=TensionConfig)
    expression_config: ExpressionConfig = field(default_factory=ExpressionConfig)

    # Sub-score weights (should sum to 1.0)
    tis_weight: float = 0.40
    tension_weight: float = 0.35
    expression_weight: float = 0.25

    # Level thresholds
    excellent_threshold: float = 0.80
    good_threshold: float = 0.65
    moderate_threshold: float = 0.45

    # Minimum audio duration for reliable analysis
    min_audio_duration: float = 3.0
