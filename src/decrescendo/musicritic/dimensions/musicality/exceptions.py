"""Exception classes for Musicality dimension."""

from __future__ import annotations


class MusicalityError(Exception):
    """Base exception for musicality analysis errors."""

    pass


class AudioTooShortError(MusicalityError):
    """Raised when audio is too short for reliable musicality analysis.

    Attributes:
        actual_duration: The actual duration of the audio in seconds.
        required_duration: The minimum required duration in seconds.
    """

    def __init__(
        self,
        actual_duration: float,
        required_duration: float,
        message: str | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            actual_duration: The actual duration of the audio in seconds.
            required_duration: The minimum required duration in seconds.
            message: Optional custom message.
        """
        self.actual_duration = actual_duration
        self.required_duration = required_duration

        if message is None:
            message = (
                f"Audio duration ({actual_duration:.2f}s) is shorter than "
                f"the required minimum ({required_duration:.2f}s) for "
                f"reliable musicality analysis."
            )

        super().__init__(message)


class TISAnalysisError(MusicalityError):
    """Raised when Tonal Interval Space computation fails."""

    pass


class TensionAnalysisError(MusicalityError):
    """Raised when tension-resolution analysis fails."""

    pass


class ExpressionAnalysisError(MusicalityError):
    """Raised when expression analysis fails."""

    pass
