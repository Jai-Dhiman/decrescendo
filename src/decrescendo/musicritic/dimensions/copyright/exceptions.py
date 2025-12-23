"""Custom exceptions for Copyright dimension."""


class CopyrightError(Exception):
    """Base exception for Copyright dimension errors."""

    pass


class AudioTooShortError(CopyrightError):
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


class FingerprintError(CopyrightError):
    """Raised when audio fingerprinting fails."""

    pass


class FingerprintNotAvailableError(FingerprintError):
    """Raised when Chromaprint/fpcalc is not available."""

    def __init__(self, message: str | None = None) -> None:
        if message is None:
            message = (
                "Chromaprint (fpcalc) is not available. Install with: uv pip install pyacoustid"
            )
        super().__init__(message)


class MelodySimilarityError(CopyrightError):
    """Raised when melody similarity analysis fails."""

    pass


class RhythmSimilarityError(CopyrightError):
    """Raised when rhythm similarity analysis fails."""

    pass


class DatabaseError(CopyrightError):
    """Raised when fingerprint database operations fail."""

    pass


class DatabaseNotFoundError(DatabaseError):
    """Raised when fingerprint database is not found."""

    pass
