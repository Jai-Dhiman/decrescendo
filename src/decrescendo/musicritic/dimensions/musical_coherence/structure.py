"""Structure analysis for Musical Coherence dimension.

This module analyzes musical structure including section detection,
repetition patterns, and structural clarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from .config import StructureConfig
from .exceptions import StructureAnalysisError


@dataclass
class StructureReport:
    """Report of structure analysis.

    Attributes:
        section_count: Number of detected sections.
        sections: List of (start_time, end_time, label) tuples.
        boundary_timestamps: Timestamps of section boundaries in seconds.
        repetition_ratio: Ratio of repeated vs unique sections (0.0-1.0).
            Higher values indicate more repetition (verse-chorus structure).
        structure_clarity: How clear the structure is (0.0-1.0).
            Higher values indicate clearer section boundaries.
        self_similarity_matrix: Self-similarity matrix (for visualization).
    """

    section_count: int = 0
    sections: list[tuple[float, float, str]] = field(default_factory=list)
    boundary_timestamps: list[float] = field(default_factory=list)
    repetition_ratio: float = 0.0
    structure_clarity: float = 0.0
    self_similarity_matrix: np.ndarray = field(default_factory=lambda: np.array([]))


class StructureAnalyzer:
    """Analyzes musical structure using self-similarity and novelty detection.

    Uses librosa's self-similarity matrix and novelty function to detect
    section boundaries. Labels sections based on repetition patterns.

    Example:
        >>> analyzer = StructureAnalyzer()
        >>> report = analyzer.analyze(audio, sample_rate=22050)
        >>> print(f"Detected {report.section_count} sections")
        >>> print(f"Structure clarity: {report.structure_clarity:.2%}")
    """

    def __init__(self, config: StructureConfig | None = None) -> None:
        """Initialize the structure analyzer.

        Args:
            config: Structure analysis configuration. Uses defaults if None.
        """
        self.config = config or StructureConfig()

    def compute_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Compute features for self-similarity analysis.

        Uses MFCCs and chroma features for robust structure detection.

        Args:
            audio: Audio samples (mono, float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Feature matrix (n_features x n_frames).

        Raises:
            StructureAnalysisError: If feature computation fails.
        """
        try:
            # Compute MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=13,
                hop_length=self.config.hop_length,
                n_fft=self.config.n_fft,
            )

            # Compute chroma
            chroma = librosa.feature.chroma_cqt(
                y=audio,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            # Stack features
            features = np.vstack([mfcc, chroma])

            return features

        except Exception as e:
            raise StructureAnalysisError(f"Feature computation failed: {e}") from e

    def compute_self_similarity(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Compute self-similarity matrix from features.

        Uses cosine similarity between feature vectors.

        Args:
            features: Feature matrix (n_features x n_frames).

        Returns:
            Self-similarity matrix (n_frames x n_frames).

        Raises:
            StructureAnalysisError: If computation fails.
        """
        try:
            # Normalize features
            norms = np.linalg.norm(features, axis=0, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            features_norm = features / norms

            # Compute cosine similarity
            similarity = np.dot(features_norm.T, features_norm)

            return similarity

        except Exception as e:
            raise StructureAnalysisError(f"Self-similarity computation failed: {e}") from e

    def detect_boundaries(
        self,
        similarity: np.ndarray,
        sample_rate: int,
    ) -> list[float]:
        """Detect section boundaries using novelty function.

        Uses a checkerboard kernel to detect transitions in the
        self-similarity matrix.

        Args:
            similarity: Self-similarity matrix.
            sample_rate: Sample rate in Hz.

        Returns:
            List of boundary timestamps in seconds.

        Raises:
            StructureAnalysisError: If boundary detection fails.
        """
        try:
            n_frames = similarity.shape[0]

            if n_frames < 10:
                return []

            # Compute novelty function using checkerboard kernel
            kernel_size = min(32, n_frames // 4)
            if kernel_size < 4:
                return []

            # Create checkerboard kernel
            half = kernel_size // 2
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[:half, :half] = 1
            kernel[half:, half:] = 1
            kernel[:half, half:] = -1
            kernel[half:, :half] = -1

            # Convolve with similarity matrix diagonal
            novelty = np.zeros(n_frames)

            for i in range(half, n_frames - half):
                # Extract local region
                region = similarity[i - half : i + half, i - half : i + half]
                # Compute novelty as correlation with checkerboard
                if region.shape == (kernel_size, kernel_size):
                    novelty[i] = np.sum(region * kernel)

            # Normalize novelty
            if np.max(np.abs(novelty)) > 0:
                novelty = novelty / np.max(np.abs(novelty))

            # Smooth novelty function
            novelty = median_filter(novelty, size=5)

            # Find peaks
            min_distance = int(
                self.config.min_section_duration * sample_rate / self.config.hop_length
            )
            min_distance = max(1, min_distance)

            peaks, _ = signal.find_peaks(
                novelty,
                height=self.config.novelty_threshold,
                distance=min_distance,
            )

            # Convert to timestamps
            timestamps = librosa.frames_to_time(
                peaks,
                sr=sample_rate,
                hop_length=self.config.hop_length,
            )

            return timestamps.tolist()

        except Exception as e:
            raise StructureAnalysisError(f"Boundary detection failed: {e}") from e

    def label_sections(
        self,
        similarity: np.ndarray,
        boundaries: list[float],
        sample_rate: int,
        audio_duration: float,
    ) -> list[tuple[float, float, str]]:
        """Label sections based on similarity patterns.

        Assigns labels (A, B, C, ...) based on inter-section similarity.
        Similar sections get the same label.

        Args:
            similarity: Self-similarity matrix.
            boundaries: List of boundary timestamps.
            sample_rate: Sample rate in Hz.
            audio_duration: Total audio duration in seconds.

        Returns:
            List of (start_time, end_time, label) tuples.
        """
        if len(boundaries) == 0:
            # No boundaries = one section
            return [(0.0, audio_duration, "A")]

        # Create section boundaries including start and end
        all_boundaries = [0.0] + sorted(boundaries) + [audio_duration]

        # Convert to frame indices
        boundary_frames = librosa.time_to_frames(
            all_boundaries,
            sr=sample_rate,
            hop_length=self.config.hop_length,
        )

        n_sections = len(all_boundaries) - 1
        section_features = []

        # Compute average feature for each section
        for i in range(n_sections):
            start_frame = boundary_frames[i]
            end_frame = boundary_frames[i + 1]

            if start_frame >= similarity.shape[0]:
                start_frame = similarity.shape[0] - 1
            if end_frame >= similarity.shape[0]:
                end_frame = similarity.shape[0]

            if start_frame < end_frame:
                # Average similarity within section
                section_sim = np.mean(similarity[start_frame:end_frame, start_frame:end_frame])
                section_features.append(section_sim)
            else:
                section_features.append(0.0)

        # Cluster sections by similarity
        section_features = np.array(section_features)
        labels = [""] * n_sections
        current_label = "A"
        label_representatives = {}

        for i in range(n_sections):
            best_match = None
            best_similarity = 0.5  # Threshold for matching

            # Check if similar to any existing section
            for label, rep_idx in label_representatives.items():
                # Compute cross-section similarity
                start_i = boundary_frames[i]
                end_i = boundary_frames[i + 1]
                start_rep = boundary_frames[rep_idx]
                end_rep = boundary_frames[rep_idx + 1]

                # Clamp indices
                start_i = min(start_i, similarity.shape[0] - 1)
                end_i = min(end_i, similarity.shape[0])
                start_rep = min(start_rep, similarity.shape[0] - 1)
                end_rep = min(end_rep, similarity.shape[0])

                if start_i < end_i and start_rep < end_rep:
                    cross_sim = np.mean(similarity[start_i:end_i, start_rep:end_rep])
                    if cross_sim > best_similarity:
                        best_similarity = cross_sim
                        best_match = label

            if best_match is not None:
                labels[i] = best_match
            else:
                labels[i] = current_label
                label_representatives[current_label] = i
                # Move to next label
                current_label = chr(ord(current_label) + 1)
                if current_label > "Z":
                    current_label = "A"  # Wrap around

        # Create section list
        sections = []
        for i in range(n_sections):
            start_time = all_boundaries[i]
            end_time = all_boundaries[i + 1]
            label = labels[i]
            sections.append((start_time, end_time, label))

        return sections

    def compute_repetition_ratio(
        self,
        sections: list[tuple[float, float, str]],
    ) -> float:
        """Compute ratio of repeated sections.

        Higher ratio indicates more verse-chorus-like structure.

        Args:
            sections: List of (start, end, label) tuples.

        Returns:
            Repetition ratio from 0.0 to 1.0.
        """
        if len(sections) <= 1:
            return 0.0

        labels = [s[2] for s in sections]
        unique_labels = set(labels)

        # Count total vs unique labels
        n_total = len(labels)
        n_unique = len(unique_labels)

        # Ratio of repeated labels
        # If all unique: ratio = 0
        # If all same: ratio = 1
        repetition_ratio = 1.0 - (n_unique / n_total)

        return float(repetition_ratio)

    def compute_structure_clarity(
        self,
        similarity: np.ndarray,
        boundaries: list[float],
        sample_rate: int,
    ) -> float:
        """Compute clarity of structure.

        Measures how clearly defined section boundaries are.

        Args:
            similarity: Self-similarity matrix.
            boundaries: List of boundary timestamps.
            sample_rate: Sample rate in Hz.

        Returns:
            Clarity score from 0.0 to 1.0.
        """
        if len(boundaries) == 0:
            # No structure detected - check if audio is coherent
            # Coherent audio with no sections still has some structure
            if similarity.size > 0:
                # High overall self-similarity = coherent but no sections
                overall_sim = np.mean(similarity)
                return float(overall_sim * 0.5)
            return 0.0

        # Measure contrast at boundaries
        boundary_frames = librosa.time_to_frames(
            boundaries,
            sr=sample_rate,
            hop_length=self.config.hop_length,
        )

        contrasts = []
        window = 8  # Frames to look before/after boundary

        for frame in boundary_frames:
            if frame < window or frame >= similarity.shape[0] - window:
                continue

            # Similarity within sections (before and after boundary)
            before = np.mean(similarity[frame - window : frame, frame - window : frame])
            after = np.mean(similarity[frame : frame + window, frame : frame + window])

            # Similarity across boundary
            across = np.mean(similarity[frame - window : frame, frame : frame + window])

            # Contrast: high within, low across
            within = (before + after) / 2
            contrast = within - across

            contrasts.append(contrast)

        if len(contrasts) == 0:
            return 0.3  # Default moderate clarity

        avg_contrast = np.mean(contrasts)

        # Normalize contrast to 0-1 range
        # Contrast of 0.5 = very clear, 0 = no contrast
        clarity = min(1.0, max(0.0, avg_contrast * 2))

        return float(clarity)

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> StructureReport:
        """Run complete structure analysis.

        Args:
            audio: Audio samples (mono, float32, normalized).
            sample_rate: Sample rate in Hz.

        Returns:
            StructureReport with all structure analysis results.

        Raises:
            StructureAnalysisError: If analysis fails.
        """
        audio_duration = len(audio) / sample_rate

        # Compute features
        features = self.compute_features(audio, sample_rate)

        # Compute self-similarity
        similarity = self.compute_self_similarity(features)

        # Detect boundaries
        boundaries = self.detect_boundaries(similarity, sample_rate)

        # Label sections
        sections = self.label_sections(similarity, boundaries, sample_rate, audio_duration)

        # Compute metrics
        repetition_ratio = self.compute_repetition_ratio(sections)
        structure_clarity = self.compute_structure_clarity(similarity, boundaries, sample_rate)

        return StructureReport(
            section_count=len(sections),
            sections=sections,
            boundary_timestamps=boundaries,
            repetition_ratio=repetition_ratio,
            structure_clarity=structure_clarity,
            self_similarity_matrix=similarity,
        )

    def compute_score(
        self,
        report: StructureReport,
        audio_duration: float = 30.0,
    ) -> float:
        """Convert structure report to quality score.

        Score is based on:
        - Structure clarity (40% weight) - clear boundaries
        - Repetition balance (30% weight) - some but not too much repetition
        - Section appropriateness (30% weight) - reasonable number of sections

        Args:
            report: StructureReport from analyze().
            audio_duration: Audio duration in seconds.

        Returns:
            Quality score from 0.0 to 1.0 (higher = better structure).
        """
        # Clarity score (direct)
        clarity_score = report.structure_clarity

        # Repetition score - some repetition is good, too much or none is bad
        # Ideal: 0.3-0.6 repetition ratio
        if 0.2 <= report.repetition_ratio <= 0.7:
            repetition_score = 1.0
        elif report.repetition_ratio < 0.2:
            # Too little repetition
            repetition_score = 0.5 + report.repetition_ratio * 2.5
        else:
            # Too much repetition
            repetition_score = 0.5 + (1.0 - report.repetition_ratio) * 1.7

        # Section appropriateness - reasonable number for duration
        # Expect roughly 1 section per 10-15 seconds
        expected_sections = max(1, audio_duration / 12.0)
        section_ratio = report.section_count / expected_sections

        if 0.5 <= section_ratio <= 2.0:
            section_score = 1.0
        elif section_ratio < 0.5:
            section_score = section_ratio * 2
        else:
            section_score = max(0.0, 1.0 - (section_ratio - 2.0) * 0.5)

        # Weighted combination
        score = 0.40 * clarity_score + 0.30 * repetition_score + 0.30 * section_score

        return float(np.clip(score, 0.0, 1.0))
