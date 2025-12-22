"""Fixtures for Copyright dimension tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate for tests."""
    return 44100


@pytest.fixture
def short_audio(sample_rate: int) -> np.ndarray:
    """Very short audio (0.5 seconds) - below minimum duration."""
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def sine_440hz(sample_rate: int) -> np.ndarray:
    """2-second 440 Hz sine wave."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def sine_880hz(sample_rate: int) -> np.ndarray:
    """2-second 880 Hz sine wave (octave above 440 Hz)."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return np.sin(2 * np.pi * 880 * t).astype(np.float32)


@pytest.fixture
def sine_550hz(sample_rate: int) -> np.ndarray:
    """2-second 550 Hz sine wave (different pitch)."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return np.sin(2 * np.pi * 550 * t).astype(np.float32)


@pytest.fixture
def complex_audio(sample_rate: int) -> np.ndarray:
    """3-second complex audio with multiple frequencies and amplitude envelope."""
    duration = 3.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, dtype=np.float32)

    # Multiple harmonics
    signal = (
        0.5 * np.sin(2 * np.pi * 220 * t)  # A3
        + 0.3 * np.sin(2 * np.pi * 330 * t)  # E4
        + 0.2 * np.sin(2 * np.pi * 440 * t)  # A4
    )

    # Apply amplitude envelope
    envelope = np.exp(-t * 0.5)
    signal = signal * envelope

    return signal.astype(np.float32)


@pytest.fixture
def rhythmic_audio(sample_rate: int) -> np.ndarray:
    """3-second audio with clear rhythmic pattern (120 BPM)."""
    duration = 3.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, dtype=np.float32)

    # Base tone
    signal = 0.3 * np.sin(2 * np.pi * 220 * t)

    # Add rhythmic clicks at 120 BPM (every 0.5 seconds)
    beat_interval = 0.5
    for beat_time in np.arange(0, duration, beat_interval):
        beat_sample = int(beat_time * sample_rate)
        click_duration = int(0.02 * sample_rate)  # 20ms click
        if beat_sample + click_duration < samples:
            click_env = np.exp(-np.linspace(0, 5, click_duration))
            signal[beat_sample : beat_sample + click_duration] += 0.7 * click_env

    return np.clip(signal, -1.0, 1.0).astype(np.float32)


@pytest.fixture
def different_rhythm_audio(sample_rate: int) -> np.ndarray:
    """3-second audio with different rhythmic pattern (90 BPM)."""
    duration = 3.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, dtype=np.float32)

    # Base tone (different pitch)
    signal = 0.3 * np.sin(2 * np.pi * 330 * t)

    # Add rhythmic clicks at 90 BPM (every 0.667 seconds)
    beat_interval = 60.0 / 90.0
    for beat_time in np.arange(0, duration, beat_interval):
        beat_sample = int(beat_time * sample_rate)
        click_duration = int(0.02 * sample_rate)
        if beat_sample + click_duration < samples:
            click_env = np.exp(-np.linspace(0, 5, click_duration))
            signal[beat_sample : beat_sample + click_duration] += 0.7 * click_env

    return np.clip(signal, -1.0, 1.0).astype(np.float32)


@pytest.fixture
def white_noise(sample_rate: int) -> np.ndarray:
    """2-second white noise."""
    duration = 2.0
    samples = int(sample_rate * duration)
    rng = np.random.default_rng(42)
    return (rng.random(samples) * 2 - 1).astype(np.float32) * 0.5


@pytest.fixture
def silence(sample_rate: int) -> np.ndarray:
    """2-second silence (near-zero audio)."""
    duration = 2.0
    samples = int(sample_rate * duration)
    return np.zeros(samples, dtype=np.float32)
