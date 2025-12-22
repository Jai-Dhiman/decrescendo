"""Configuration dataclasses for Musical Coherence dimension."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StructureConfig:
    """Configuration for structure analysis.

    Attributes:
        min_section_duration: Minimum section length in seconds.
        novelty_threshold: Threshold for detecting section boundaries (0-1).
            Lower values detect more boundaries.
        hop_length: Hop length for feature extraction in samples.
        n_fft: FFT size for spectral analysis.
    """

    min_section_duration: float = 4.0
    novelty_threshold: float = 0.3
    hop_length: int = 512
    n_fft: int = 2048


@dataclass(frozen=True)
class HarmonyConfig:
    """Configuration for harmony analysis.

    Attributes:
        use_essentia: Whether to use Essentia for chord detection.
            If True and Essentia unavailable, raises exception.
        key_profile: Which key profile to use ("krumhansl" or "temperley").
        min_chord_duration: Minimum chord duration in seconds.
        hop_length: Hop length for chroma computation in samples.
    """

    use_essentia: bool = False  # Default to librosa (essentia is optional)
    key_profile: str = "krumhansl"
    min_chord_duration: float = 0.25
    hop_length: int = 2048


@dataclass(frozen=True)
class RhythmConfig:
    """Configuration for rhythm analysis.

    Attributes:
        use_madmom: Whether to use madmom for beat tracking.
            If True and madmom unavailable, raises exception.
        min_tempo: Minimum expected tempo in BPM.
        max_tempo: Maximum expected tempo in BPM.
        tempo_stability_window: Number of beats for stability calculation.
        hop_length: Hop length for onset detection in samples.
    """

    use_madmom: bool = False  # Default to librosa (madmom is optional)
    min_tempo: float = 40.0
    max_tempo: float = 240.0
    tempo_stability_window: int = 8
    hop_length: int = 512


@dataclass(frozen=True)
class MelodyConfig:
    """Configuration for melody analysis.

    Attributes:
        fmin: Minimum frequency for pitch tracking in Hz.
        fmax: Maximum frequency for pitch tracking in Hz.
        frame_length: Frame length for pYIN in samples.
        hop_length: Hop length for pYIN in samples.
        min_voiced_ratio: Minimum ratio of voiced frames for valid melody.
    """

    fmin: float = 65.0  # C2
    fmax: float = 2093.0  # C7
    frame_length: int = 2048
    hop_length: int = 512
    min_voiced_ratio: float = 0.1


@dataclass(frozen=True)
class MusicalCoherenceConfig:
    """Configuration for Musical Coherence evaluation.

    Attributes:
        structure_config: Configuration for structure analysis.
        harmony_config: Configuration for harmony analysis.
        rhythm_config: Configuration for rhythm analysis.
        melody_config: Configuration for melody analysis.
        structure_weight: Weight for structure score (0-1).
        harmony_weight: Weight for harmony score (0-1).
        rhythm_weight: Weight for rhythm score (0-1).
        melody_weight: Weight for melody score (0-1).
        excellent_threshold: Score threshold for "excellent" coherence.
        good_threshold: Score threshold for "good" coherence.
        moderate_threshold: Score threshold for "moderate" coherence.
        min_audio_duration: Minimum audio duration in seconds.
    """

    structure_config: StructureConfig = field(default_factory=StructureConfig)
    harmony_config: HarmonyConfig = field(default_factory=HarmonyConfig)
    rhythm_config: RhythmConfig = field(default_factory=RhythmConfig)
    melody_config: MelodyConfig = field(default_factory=MelodyConfig)

    # Sub-score weights (should sum to 1.0)
    structure_weight: float = 0.25
    harmony_weight: float = 0.30
    rhythm_weight: float = 0.25
    melody_weight: float = 0.20

    # Coherence level thresholds
    excellent_threshold: float = 0.80
    good_threshold: float = 0.65
    moderate_threshold: float = 0.45

    # Minimum audio duration for reliable analysis
    min_audio_duration: float = 2.0
