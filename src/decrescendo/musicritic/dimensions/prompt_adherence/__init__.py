"""Prompt Adherence dimension evaluator.

This module provides the PromptAdherenceEvaluator for measuring how well
AI-generated audio matches a text prompt using CLAP embeddings.
"""

from .clap_encoder import CLAPEncoder
from .config import CLAPEncoderConfig, PromptAdherenceConfig
from .evaluator import PromptAdherenceEvaluator
from .exceptions import (
    CLAPDependencyError,
    CLAPEncoderError,
    CLAPLoadError,
    PromptRequiredError,
)

__all__ = [
    # Evaluator
    "PromptAdherenceEvaluator",
    # Encoder
    "CLAPEncoder",
    # Config
    "CLAPEncoderConfig",
    "PromptAdherenceConfig",
    # Exceptions
    "CLAPEncoderError",
    "CLAPLoadError",
    "CLAPDependencyError",
    "PromptRequiredError",
]
