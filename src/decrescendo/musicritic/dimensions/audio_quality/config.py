"""Configuration dataclasses for Audio Quality dimension."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArtifactDetectionConfig:
    """Configuration for artifact detection.

    Attributes:
        click_threshold: Sensitivity for click/pop detection (0-1).
            Higher values detect fewer, more prominent clicks.
        clipping_threshold: Sample value threshold for clipping detection.
            Samples above this absolute value are considered clipped.
        min_clipping_samples: Minimum consecutive samples to count as a clip.
        spectral_flux_threshold: Threshold for spectral flux anomaly detection.
    """

    click_threshold: float = 0.1
    clipping_threshold: float = 0.99
    min_clipping_samples: int = 3
    spectral_flux_threshold: float = 2.0


@dataclass(frozen=True)
class LoudnessConfig:
    """Configuration for loudness analysis.

    All measurements follow ITU-R BS.1770-4 standard.

    Attributes:
        target_lufs: Target integrated loudness in LUFS.
            -14 LUFS is the streaming platform standard.
        max_true_peak_dbtp: Maximum True Peak in dBTP.
            -1 dBTP is required for streaming compliance.
        min_lra: Minimum acceptable Loudness Range in LU.
        max_lra: Maximum acceptable Loudness Range in LU.
        block_size: Block size for LUFS measurement in seconds.
    """

    target_lufs: float = -14.0
    max_true_peak_dbtp: float = -1.0
    min_lra: float = 4.0
    max_lra: float = 20.0
    block_size: float = 0.4


@dataclass(frozen=True)
class PerceptualConfig:
    """Configuration for perceptual quality analysis.

    Attributes:
        target_sample_rate: Sample rate for analysis.
        frequency_bands: Tuple of (low, high) frequency ranges in Hz.
        ideal_balance: Ideal energy distribution across bands (must sum to 1.0).
        min_centroid_hz: Minimum expected spectral centroid for music.
        max_centroid_hz: Maximum expected spectral centroid for music.
    """

    target_sample_rate: int = 44100
    frequency_bands: tuple[tuple[int, int], ...] = (
        (20, 250),  # Sub-bass + Bass
        (250, 2000),  # Mids
        (2000, 8000),  # Upper mids / Presence
        (8000, 20000),  # Highs / Brilliance
    )
    ideal_balance: tuple[float, ...] = (0.25, 0.35, 0.25, 0.15)
    min_centroid_hz: float = 500.0
    max_centroid_hz: float = 4000.0


@dataclass(frozen=True)
class AudioQualityConfig:
    """Configuration for Audio Quality evaluation.

    Attributes:
        artifact_config: Configuration for artifact detection.
        loudness_config: Configuration for loudness analysis.
        perceptual_config: Configuration for perceptual quality.
        artifact_weight: Weight for artifact score (0-1).
        loudness_weight: Weight for loudness score (0-1).
        perceptual_weight: Weight for perceptual score (0-1).
        excellent_threshold: Score threshold for "excellent" quality.
        good_threshold: Score threshold for "good" quality.
        acceptable_threshold: Score threshold for "acceptable" quality.
        min_audio_duration: Minimum audio duration in seconds.
    """

    artifact_config: ArtifactDetectionConfig = field(default_factory=ArtifactDetectionConfig)
    loudness_config: LoudnessConfig = field(default_factory=LoudnessConfig)
    perceptual_config: PerceptualConfig = field(default_factory=PerceptualConfig)

    # Sub-score weights (should sum to 1.0)
    artifact_weight: float = 0.35
    loudness_weight: float = 0.35
    perceptual_weight: float = 0.30

    # Quality level thresholds
    excellent_threshold: float = 0.85
    good_threshold: float = 0.70
    acceptable_threshold: float = 0.50

    # Minimum audio duration for reliable analysis
    min_audio_duration: float = 0.5
