"""Custom exceptions for Musical Coherence dimension."""


class MusicalCoherenceError(Exception):
    """Base exception for Musical Coherence dimension errors."""

    pass


class AudioTooShortError(MusicalCoherenceError):
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


class StructureAnalysisError(MusicalCoherenceError):
    """Raised when structure analysis fails."""

    pass


class HarmonyAnalysisError(MusicalCoherenceError):
    """Raised when harmony analysis fails."""

    pass


class RhythmAnalysisError(MusicalCoherenceError):
    """Raised when rhythm analysis fails."""

    pass


class MelodyAnalysisError(MusicalCoherenceError):
    """Raised when melody analysis fails."""

    pass


class DependencyNotAvailableError(MusicalCoherenceError):
    """Raised when an optional dependency is not available.

    Attributes:
        dependency: Name of the missing dependency.
        feature: Feature that requires the dependency.
    """

    def __init__(self, dependency: str, feature: str) -> None:
        self.dependency = dependency
        self.feature = feature
        message = (
            f"Optional dependency '{dependency}' is not available. "
            f"Install it with: pip install {dependency}. "
            f"Required for: {feature}"
        )
        super().__init__(message)
