"""Shared fixtures for Prompt Adherence tests."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions.prompt_adherence import (
    CLAPEncoder,
    CLAPEncoderConfig,
)


@pytest.fixture(scope="session")
def clap_encoder():
    """Shared CLAPEncoder instance (session-scoped for efficiency).

    This loads the real CLAP model once per test session.
    """
    return CLAPEncoder()


@pytest.fixture
def sample_audio_48k():
    """Generate sample audio at 48kHz (CLAP's native sample rate).

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 1.0
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # 440 Hz sine wave (A4 note)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def sample_audio_44k():
    """Generate sample audio at 44.1kHz (common sample rate).

    Returns:
        Tuple of (audio array, sample rate).
    """
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # 440 Hz sine wave
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def sample_stereo_audio():
    """Generate stereo sample audio.

    Returns:
        Tuple of (audio array with shape (2, samples), sample rate).
    """
    duration = 1.0
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Different frequencies for left and right channels
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    audio = np.stack([left, right])
    return audio, sample_rate


@pytest.fixture
def sample_prompt():
    """Sample text prompt for testing."""
    return "a pure sine wave tone"


@pytest.fixture
def music_prompt():
    """Music-related prompt for testing."""
    return "upbeat electronic dance music with synthesizers"


@pytest.fixture
def mismatched_prompt():
    """Prompt that shouldn't match a sine wave."""
    return "heavy metal guitar solo with distortion"
