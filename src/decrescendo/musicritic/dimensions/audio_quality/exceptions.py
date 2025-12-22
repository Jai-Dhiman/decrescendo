"""Custom exceptions for Audio Quality dimension."""


class AudioQualityError(Exception):
    """Base exception for Audio Quality dimension errors."""

    pass


class AudioTooShortError(AudioQualityError):
    """Raised when audio is too short for reliable analysis.

    Attributes:
        duration: Actual audio duration in seconds.
        min_duration: Required minimum duration in seconds.
    """

    def __init__(
        self,
        duration: float,
        min_duration: float,
        message: str | None = None,
    ) -> None:
        self.duration = duration
        self.min_duration = min_duration
        if message is None:
            message = (
                f"Audio duration ({duration:.2f}s) is too short. "
                f"Minimum required: {min_duration:.2f}s"
            )
        super().__init__(message)


class LoudnessAnalysisError(AudioQualityError):
    """Raised when loudness analysis fails."""

    pass


class ArtifactDetectionError(AudioQualityError):
    """Raised when artifact detection fails."""

    pass


class PerceptualAnalysisError(AudioQualityError):
    """Raised when perceptual quality analysis fails."""

    pass
