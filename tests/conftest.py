"""Pytest fixtures for AudioML tests."""

import jax
import pytest


@pytest.fixture
def rng():
    """Provide a JAX random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def rng_split(rng):
    """Provide multiple random keys."""
    return jax.random.split(rng, 4)
