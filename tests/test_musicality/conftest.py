"""Shared fixtures for Musicality tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate for tests."""
    return 22050


@pytest.fixture
def tonal_audio(sample_rate: int) -> tuple[np.ndarray, int]:
    """Generate audio with clear tonal content (C major arpeggio).

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    n_samples = int(sample_rate * duration)
    audio = np.zeros(n_samples, dtype=np.float32)

    # C major arpeggio frequencies
    frequencies = [261.63, 329.63, 392.00, 523.25]  # C4, E4, G4, C5
    note_duration = duration / len(frequencies)

    for i, freq in enumerate(frequencies):
        start = int(i * note_duration * sample_rate)
        end = int((i + 1) * note_duration * sample_rate)
        t = np.linspace(0, note_duration, end - start, dtype=np.float32)
        envelope = np.exp(-t * 2) * 0.5
        audio[start:end] = (envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def tension_resolution_audio(sample_rate: int) -> tuple[np.ndarray, int]:
    """Generate audio with clear tension-resolution pattern.

    Pattern: Tonic -> Dominant -> Tonic (I-V-I cadence)

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 6.0
    n_samples = int(sample_rate * duration)
    audio = np.zeros(n_samples, dtype=np.float32)

    # I chord (C major) - low tension
    # V chord (G major) - high tension
    # I chord (C major) - resolution
    chords = [
        ([261.63, 329.63, 392.00], 0.3),  # C major - tonic
        ([392.00, 493.88, 587.33], 0.5),  # G major - dominant (tension)
        ([261.63, 329.63, 392.00], 0.3),  # C major - resolution
    ]

    chord_duration = duration / len(chords)

    for i, (freqs, amp) in enumerate(chords):
        start = int(i * chord_duration * sample_rate)
        end = int((i + 1) * chord_duration * sample_rate)
        t = np.linspace(0, chord_duration, end - start, dtype=np.float32)

        chord_audio = np.zeros(end - start, dtype=np.float32)
        for freq in freqs:
            chord_audio += amp * np.sin(2 * np.pi * freq * t) / len(freqs)

        audio[start:end] = chord_audio.astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def dynamic_audio(sample_rate: int) -> tuple[np.ndarray, int]:
    """Generate audio with dynamic variation (crescendo-decrescendo).

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    # Crescendo-decrescendo envelope
    envelope = np.sin(np.pi * t / duration)

    # Base tone
    audio = (envelope * 0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def flat_audio(sample_rate: int) -> tuple[np.ndarray, int]:
    """Generate audio with no dynamic variation.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def atonal_audio(sample_rate: int) -> tuple[np.ndarray, int]:
    """Generate atonal audio (random pitch classes).

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    n_samples = int(sample_rate * duration)
    audio = np.zeros(n_samples, dtype=np.float32)

    rng = np.random.default_rng(42)
    n_notes = 16
    note_duration = duration / n_notes

    for i in range(n_notes):
        # Random pitch class (atonal)
        freq = 220.0 * 2 ** (rng.integers(0, 12) / 12)
        start = int(i * note_duration * sample_rate)
        end = int((i + 1) * note_duration * sample_rate)
        t = np.linspace(0, note_duration, end - start, dtype=np.float32)
        envelope = np.exp(-t * 3) * 0.3
        audio[start:end] = (envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def short_audio(sample_rate: int) -> tuple[np.ndarray, int]:
    """Generate audio shorter than minimum duration.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 1.0  # Less than 3.0 second minimum
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def silence_audio(sample_rate: int) -> tuple[np.ndarray, int]:
    """Generate silent audio.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    n_samples = int(sample_rate * duration)
    audio = np.zeros(n_samples, dtype=np.float32)
    return audio, sample_rate


@pytest.fixture
def complex_harmonic_audio(sample_rate: int) -> tuple[np.ndarray, int]:
    """Generate audio with complex harmonic content (diminished chord).

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    # Diminished 7th chord (high dissonance/complexity)
    freqs = [261.63, 311.13, 369.99, 440.0]  # C, Eb, Gb, A (dim7)
    audio = np.zeros(n_samples, dtype=np.float32)

    for freq in freqs:
        audio += 0.2 * np.sin(2 * np.pi * freq * t)

    return audio.astype(np.float32), sample_rate
