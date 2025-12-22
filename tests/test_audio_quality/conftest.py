"""Shared fixtures for Audio Quality tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_audio_44k():
    """Generate clean sample audio at 44.1kHz.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # 440 Hz sine wave (A4 note)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def sample_audio_short():
    """Generate short audio sample (0.3 seconds).

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 0.3
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def sample_stereo_audio():
    """Generate stereo sample audio.

    Returns:
        Tuple of (audio array with shape (2, samples), sample rate).
    """
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = 0.5 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
    audio = np.stack([left, right])
    return audio, sample_rate


@pytest.fixture
def clipped_audio():
    """Generate audio with clipping artifacts.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Generate overdriven signal that clips
    audio = 2.0 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    audio = np.clip(audio, -0.99, 0.99)  # Hard clip
    return audio, sample_rate


@pytest.fixture
def audio_with_clicks():
    """Generate audio with click artifacts.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Add click artifacts (sudden spikes)
    click_positions = [int(0.5 * sample_rate), int(1.2 * sample_rate)]
    for pos in click_positions:
        if pos < len(audio) - 10:
            audio[pos : pos + 5] = 0.95  # Sharp transient

    return audio, sample_rate


@pytest.fixture
def quiet_audio():
    """Generate very quiet audio.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Very quiet signal (-30 dBFS)
    audio = 0.03 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def loud_audio():
    """Generate loud audio near 0 dBFS.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Near peak level
    audio = 0.95 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def silence():
    """Generate silence.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 2.0
    sample_rate = 44100
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    return audio, sample_rate


@pytest.fixture
def white_noise():
    """Generate white noise.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 2.0
    sample_rate = 44100
    rng = np.random.default_rng(42)
    audio = 0.3 * rng.standard_normal(int(sample_rate * duration)).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def bass_heavy_audio():
    """Generate bass-heavy audio.

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Low frequency content (80 Hz)
    audio = 0.5 * np.sin(2 * np.pi * 80 * t).astype(np.float32)
    return audio, sample_rate
