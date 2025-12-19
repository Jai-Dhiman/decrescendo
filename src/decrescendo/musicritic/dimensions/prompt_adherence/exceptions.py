"""Custom exceptions for Prompt Adherence dimension."""


class CLAPEncoderError(Exception):
    """Base exception for CLAP encoder errors."""

    pass


class CLAPLoadError(CLAPEncoderError):
    """Raised when CLAP model cannot be loaded."""

    pass


class CLAPDependencyError(CLAPEncoderError):
    """Raised when laion-clap package is not installed."""

    pass


class PromptRequiredError(ValueError):
    """Raised when prompt is required but not provided."""

    pass
