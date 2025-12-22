"""Shared fixtures for Musical Coherence tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 22050


@pytest.fixture
def sample_audio(sample_rate):
    """Generate basic sample audio at 22.05kHz.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # 440 Hz sine wave (A4 note)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def short_audio(sample_rate):
    """Generate short audio sample (1 second).

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def rhythmic_audio(sample_rate):
    """Generate audio with clear rhythmic pulse (120 BPM).

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    bpm = 120
    beat_interval = 60.0 / bpm  # seconds per beat
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    audio = np.zeros(n_samples, dtype=np.float32)

    # Add clicks at each beat
    for beat_time in np.arange(0, duration, beat_interval):
        beat_sample = int(beat_time * sample_rate)
        if beat_sample < n_samples - 100:
            # Short transient (click/drum hit)
            attack = np.linspace(0, 0.8, 10)
            decay = np.exp(-np.linspace(0, 5, 90)) * 0.8
            envelope = np.concatenate([attack, decay])
            audio[beat_sample : beat_sample + 100] += envelope.astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def melodic_audio(sample_rate):
    """Generate audio with clear melodic content (C major scale).

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    n_samples = int(sample_rate * duration)
    audio = np.zeros(n_samples, dtype=np.float32)

    # C major scale frequencies
    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    note_duration = duration / len(frequencies)

    for i, freq in enumerate(frequencies):
        start = int(i * note_duration * sample_rate)
        end = int((i + 1) * note_duration * sample_rate)
        t = np.linspace(0, note_duration, end - start, dtype=np.float32)
        # Apply envelope
        envelope = np.exp(-t * 2) * 0.5
        audio[start:end] = (envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def harmonic_audio(sample_rate):
    """Generate audio with harmonic content (C major chord).

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    # C major chord (C4, E4, G4)
    c4 = 261.63
    e4 = 329.63
    g4 = 392.00

    audio = (
        0.3 * np.sin(2 * np.pi * c4 * t)
        + 0.3 * np.sin(2 * np.pi * e4 * t)
        + 0.3 * np.sin(2 * np.pi * g4 * t)
    ).astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def chord_progression_audio(sample_rate):
    """Generate audio with chord progression (I-IV-V-I).

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 8.0
    n_samples = int(sample_rate * duration)
    audio = np.zeros(n_samples, dtype=np.float32)

    # Chord frequencies (C major context)
    chords = [
        [261.63, 329.63, 392.00],  # C major (I)
        [349.23, 440.00, 523.25],  # F major (IV)
        [392.00, 493.88, 587.33],  # G major (V)
        [261.63, 329.63, 392.00],  # C major (I)
    ]

    chord_duration = duration / len(chords)

    for i, chord_freqs in enumerate(chords):
        start = int(i * chord_duration * sample_rate)
        end = int((i + 1) * chord_duration * sample_rate)
        t = np.linspace(0, chord_duration, end - start, dtype=np.float32)

        chord_audio = np.zeros(end - start, dtype=np.float32)
        for freq in chord_freqs:
            chord_audio += 0.2 * np.sin(2 * np.pi * freq * t)

        audio[start:end] = chord_audio.astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def structured_audio(sample_rate):
    """Generate audio with clear verse-chorus structure.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 16.0
    n_samples = int(sample_rate * duration)
    audio = np.zeros(n_samples, dtype=np.float32)

    section_duration = 4.0

    # Section A (verse) - lower, quieter
    a_freq = 220.0
    a_amp = 0.3

    # Section B (chorus) - higher, louder
    b_freq = 440.0
    b_amp = 0.5

    # ABAB structure
    sections = [
        (a_freq, a_amp),
        (b_freq, b_amp),
        (a_freq, a_amp),
        (b_freq, b_amp),
    ]

    for i, (freq, amp) in enumerate(sections):
        start = int(i * section_duration * sample_rate)
        end = int((i + 1) * section_duration * sample_rate)
        t = np.linspace(0, section_duration, end - start, dtype=np.float32)
        audio[start:end] = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def silence(sample_rate):
    """Generate silence.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    return audio, sample_rate


@pytest.fixture
def white_noise(sample_rate):
    """Generate white noise.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    rng = np.random.default_rng(42)
    audio = 0.3 * rng.standard_normal(int(sample_rate * duration)).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def varying_tempo_audio(sample_rate):
    """Generate audio with varying tempo.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 4.0
    n_samples = int(sample_rate * duration)
    audio = np.zeros(n_samples, dtype=np.float32)

    # Start at 100 BPM, accelerate to 140 BPM
    current_time = 0.0
    bpm_start = 100.0
    bpm_end = 140.0

    while current_time < duration:
        # Linear tempo interpolation
        progress = current_time / duration
        current_bpm = bpm_start + (bpm_end - bpm_start) * progress
        beat_interval = 60.0 / current_bpm

        beat_sample = int(current_time * sample_rate)
        if beat_sample < n_samples - 100:
            attack = np.linspace(0, 0.8, 10)
            decay = np.exp(-np.linspace(0, 5, 90)) * 0.8
            envelope = np.concatenate([attack, decay])
            audio[beat_sample : beat_sample + 100] += envelope.astype(np.float32)

        current_time += beat_interval

    return audio, sample_rate
