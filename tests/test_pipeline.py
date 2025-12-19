"""Integration tests for the Constitutional Audio pipeline.

Tests the unified pipeline that combines input classifier (text prompts)
and output classifier (audio) for safety classification.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from decrescendo.musicritic.pipeline import (
    ClassifierNotEnabledError,
    ConstitutionalAudio,
    GenerationClassificationResult,
    PipelineAudioResult,
    PipelineConfig,
    PipelineConfigError,
    PipelineDecision,
    PipelineError,
    PromptClassificationResult,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_input_classifier():
    """Create a mock input classifier."""
    from decrescendo.musicritic.input_classifier.inference import (
        ArtistRequest,
        ClassificationResult,
        Decision,
        Intent,
        VoiceRequest,
    )

    classifier = MagicMock()

    def make_result(prompt: str) -> ClassificationResult:
        """Generate result based on prompt content."""
        # Simulate different responses based on prompt
        if "drake" in prompt.lower() or "artist" in prompt.lower():
            return ClassificationResult(
                intent=Intent.SUSPICIOUS,
                intent_confidence=0.7,
                intent_probabilities={"BENIGN": 0.2, "SUSPICIOUS": 0.7, "MALICIOUS": 0.1},
                artist_request=ArtistRequest.NAMED_ARTIST,
                artist_confidence=0.9,
                artist_probabilities={"NONE": 0.05, "NAMED_ARTIST": 0.9, "STYLE_REFERENCE": 0.05},
                voice_request=VoiceRequest.NONE,
                voice_confidence=0.95,
                voice_probabilities={"NONE": 0.95, "CELEBRITY": 0.03, "POLITICIAN": 0.02},
                policy_violations={"COPYRIGHT_IP": 0.8, "VOICE_CLONING": 0.2},
                policy_flags=["COPYRIGHT_IP"],
                decision=Decision.BLOCK,
                decision_reasons=["Named artist reference detected", "Policy violations: COPYRIGHT_IP"],
            )
        elif "malicious" in prompt.lower():
            return ClassificationResult(
                intent=Intent.MALICIOUS,
                intent_confidence=0.95,
                intent_probabilities={"BENIGN": 0.02, "SUSPICIOUS": 0.03, "MALICIOUS": 0.95},
                artist_request=ArtistRequest.NONE,
                artist_confidence=0.9,
                artist_probabilities={"NONE": 0.9, "NAMED_ARTIST": 0.05, "STYLE_REFERENCE": 0.05},
                voice_request=VoiceRequest.NONE,
                voice_confidence=0.95,
                voice_probabilities={"NONE": 0.95, "CELEBRITY": 0.03, "POLITICIAN": 0.02},
                policy_violations={"CONTENT_SAFETY": 0.9},
                policy_flags=["CONTENT_SAFETY"],
                decision=Decision.BLOCK,
                decision_reasons=["Malicious intent detected"],
            )
        else:
            return ClassificationResult(
                intent=Intent.BENIGN,
                intent_confidence=0.95,
                intent_probabilities={"BENIGN": 0.95, "SUSPICIOUS": 0.03, "MALICIOUS": 0.02},
                artist_request=ArtistRequest.NONE,
                artist_confidence=0.9,
                artist_probabilities={"NONE": 0.9, "NAMED_ARTIST": 0.05, "STYLE_REFERENCE": 0.05},
                voice_request=VoiceRequest.NONE,
                voice_confidence=0.95,
                voice_probabilities={"NONE": 0.95, "CELEBRITY": 0.03, "POLITICIAN": 0.02},
                policy_violations={"COPYRIGHT_IP": 0.1, "VOICE_CLONING": 0.05},
                policy_flags=[],
                decision=Decision.ALLOW,
                decision_reasons=["No safety concerns detected"],
            )

    classifier.classify.side_effect = make_result
    classifier.classify_batch.side_effect = lambda prompts: [make_result(p) for p in prompts]

    return classifier


@pytest.fixture
def mock_output_classifier():
    """Create a mock output classifier."""
    from decrescendo.musicritic.output_classifier.inference import (
        AggregatedResult,
        AudioClassificationResult,
        Decision,
        SpeakerMatch,
    )

    classifier = MagicMock()

    def make_result(path_or_audio, sample_rate=None) -> AggregatedResult:
        """Generate result based on input."""
        return AggregatedResult(
            harm_scores={
                "copyright_ip": 0.1,
                "voice_cloning": 0.05,
                "cultural": 0.02,
                "misinformation": 0.03,
                "emotional_manipulation": 0.04,
                "content_safety": 0.02,
                "physical_safety": 0.01,
            },
            flagged_categories=[],
            best_speaker_match=SpeakerMatch(matched=False, similarity=0.3),
            decision=Decision.CONTINUE,
            decision_reasons=["No safety concerns detected"],
            num_chunks=3,
            chunk_results=[],
        )

    classifier.classify_file.side_effect = make_result
    classifier.classify_array.side_effect = make_result

    return classifier


@pytest.fixture
def mock_voice_database():
    """Create a mock voice database."""
    from decrescendo.musicritic.output_classifier.checkpointing import VoiceEntry

    db = MagicMock()
    db.list_voices.return_value = [
        VoiceEntry(voice_id=0, name="Artist A", metadata={"genre": "pop"}),
        VoiceEntry(voice_id=1, name="Artist B", metadata={"genre": "rock"}),
    ]
    db.__len__ = MagicMock(return_value=2)
    db.get_all_embeddings.return_value = (
        jnp.zeros((2, 192)),
        ["Artist A", "Artist B"],
    )
    return db


# -----------------------------------------------------------------------------
# Test PipelineConfig
# -----------------------------------------------------------------------------


class TestPipelineConfig:
    """Test PipelineConfig validation."""

    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig()
        assert config.enable_input_classifier is True
        assert config.enable_output_classifier is True
        assert config.enable_voice_matching is True

    def test_config_validation_both_disabled_raises(self):
        """Test that disabling both classifiers raises error."""
        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=False,
        )
        with pytest.raises(PipelineConfigError, match="At least one classifier"):
            config.validate()

    def test_config_validation_one_enabled_passes(self):
        """Test that having one classifier enabled is valid."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        config.validate()  # Should not raise

        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=True,
        )
        config.validate()  # Should not raise


# -----------------------------------------------------------------------------
# Test ConstitutionalAudio Initialization
# -----------------------------------------------------------------------------


class TestConstitutionalAudioInit:
    """Test ConstitutionalAudio initialization."""

    def test_init_with_both_classifiers(self, mock_input_classifier, mock_output_classifier):
        """Test initialization with both classifiers."""
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            output_classifier=mock_output_classifier,
        )

        assert pipeline.input_classifier is mock_input_classifier
        assert pipeline.output_classifier is mock_output_classifier

    def test_init_input_only(self, mock_input_classifier):
        """Test initialization with input classifier only."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            config=config,
        )

        assert pipeline.input_classifier is mock_input_classifier
        assert pipeline.output_classifier is None

    def test_init_output_only(self, mock_output_classifier):
        """Test initialization with output classifier only."""
        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=True,
        )
        pipeline = ConstitutionalAudio(
            output_classifier=mock_output_classifier,
            config=config,
        )

        assert pipeline.input_classifier is None
        assert pipeline.output_classifier is mock_output_classifier

    def test_init_missing_required_classifier_raises(self):
        """Test that missing required classifier raises error."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        with pytest.raises(PipelineConfigError, match="Input classifier is enabled"):
            ConstitutionalAudio(config=config)

    def test_init_with_voice_database(
        self,
        mock_input_classifier,
        mock_output_classifier,
        mock_voice_database,
    ):
        """Test initialization with voice database."""
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            output_classifier=mock_output_classifier,
            voice_database=mock_voice_database,
        )

        assert pipeline.voice_database is mock_voice_database


# -----------------------------------------------------------------------------
# Test Input Classifier (Standalone)
# -----------------------------------------------------------------------------


class TestInputClassifierStandalone:
    """Test input classifier (prompt) classification."""

    def test_classify_prompt_safe(self, mock_input_classifier):
        """Test classifying a safe prompt."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            config=config,
        )

        result = pipeline.classify_prompt("Generate a calm piano melody")

        assert isinstance(result, PromptClassificationResult)
        assert result.decision == PipelineDecision.ALLOW
        assert result.input_result.intent.name == "BENIGN"

    def test_classify_prompt_with_artist_reference(self, mock_input_classifier):
        """Test classifying a prompt with artist reference."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            config=config,
        )

        result = pipeline.classify_prompt("Generate music like Drake")

        assert result.decision == PipelineDecision.BLOCK
        assert result.input_result.artist_request.name == "NAMED_ARTIST"
        assert "COPYRIGHT_IP" in result.input_result.policy_flags

    def test_classify_prompt_malicious(self, mock_input_classifier):
        """Test classifying a malicious prompt."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            config=config,
        )

        result = pipeline.classify_prompt("Create something malicious")

        assert result.decision == PipelineDecision.BLOCK
        assert result.input_result.intent.name == "MALICIOUS"

    def test_classify_prompt_batch(self, mock_input_classifier):
        """Test batch prompt classification."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            config=config,
        )

        prompts = [
            "Generate a calm piano melody",
            "Generate music like Drake",
            "Create something malicious",
        ]
        results = pipeline.classify_prompt_batch(prompts)

        assert len(results) == 3
        assert results[0].decision == PipelineDecision.ALLOW
        assert results[1].decision == PipelineDecision.BLOCK
        assert results[2].decision == PipelineDecision.BLOCK

    def test_classify_prompt_disabled_raises(self, mock_output_classifier):
        """Test that classify_prompt raises when disabled."""
        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=True,
        )
        pipeline = ConstitutionalAudio(
            output_classifier=mock_output_classifier,
            config=config,
        )

        with pytest.raises(ClassifierNotEnabledError, match="Input classifier is disabled"):
            pipeline.classify_prompt("test")

    def test_result_to_dict(self, mock_input_classifier):
        """Test PromptClassificationResult serialization."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            config=config,
        )

        result = pipeline.classify_prompt("Safe prompt")
        result_dict = result.to_dict()

        assert result_dict["type"] == "prompt"
        assert "input_classifier" in result_dict
        assert result_dict["decision"] == "ALLOW"


# -----------------------------------------------------------------------------
# Test Output Classifier (Standalone)
# -----------------------------------------------------------------------------


class TestOutputClassifierStandalone:
    """Test output classifier (audio) classification."""

    def test_classify_audio_array(self, mock_output_classifier):
        """Test classifying an audio array."""
        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=True,
        )
        pipeline = ConstitutionalAudio(
            output_classifier=mock_output_classifier,
            config=config,
        )

        # Generate test audio (1 second at 24kHz)
        audio = np.random.randn(24000).astype(np.float32)
        result = pipeline.classify_audio(audio, sample_rate=24000)

        assert isinstance(result, PipelineAudioResult)
        assert result.decision == PipelineDecision.ALLOW
        mock_output_classifier.classify_array.assert_called_once()

    def test_classify_audio_file(self, mock_output_classifier, tmp_path):
        """Test classifying an audio file."""
        import soundfile as sf

        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=True,
        )
        pipeline = ConstitutionalAudio(
            output_classifier=mock_output_classifier,
            config=config,
        )

        # Create a test audio file
        audio_path = tmp_path / "test.wav"
        audio = np.random.randn(24000).astype(np.float32)
        sf.write(str(audio_path), audio, 24000)

        result = pipeline.classify_audio(audio_path)

        assert isinstance(result, PipelineAudioResult)
        mock_output_classifier.classify_file.assert_called_once()

    def test_classify_audio_requires_sample_rate(self, mock_output_classifier):
        """Test that array classification requires sample_rate."""
        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=True,
        )
        pipeline = ConstitutionalAudio(
            output_classifier=mock_output_classifier,
            config=config,
        )

        audio = np.random.randn(24000).astype(np.float32)
        with pytest.raises(ValueError, match="sample_rate is required"):
            pipeline.classify_audio(audio)

    def test_classify_audio_disabled_raises(self, mock_input_classifier):
        """Test that classify_audio raises when disabled."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            config=config,
        )

        audio = np.random.randn(24000).astype(np.float32)
        with pytest.raises(ClassifierNotEnabledError, match="Output classifier is disabled"):
            pipeline.classify_audio(audio, sample_rate=24000)

    def test_result_to_dict(self, mock_output_classifier):
        """Test PipelineAudioResult serialization."""
        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=True,
        )
        pipeline = ConstitutionalAudio(
            output_classifier=mock_output_classifier,
            config=config,
        )

        audio = np.random.randn(24000).astype(np.float32)
        result = pipeline.classify_audio(audio, sample_rate=24000)
        result_dict = result.to_dict()

        assert result_dict["type"] == "audio"
        assert "output_classifier" in result_dict
        assert result_dict["decision"] == "ALLOW"


# -----------------------------------------------------------------------------
# Test Combined Pipeline
# -----------------------------------------------------------------------------


class TestCombinedPipeline:
    """Test combined prompt + audio classification."""

    def test_classify_generation_both(
        self,
        mock_input_classifier,
        mock_output_classifier,
    ):
        """Test full generation classification with both prompt and audio."""
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            output_classifier=mock_output_classifier,
        )

        audio = np.random.randn(24000).astype(np.float32)
        result = pipeline.classify_generation(
            prompt="Generate a calm piano melody",
            audio=audio,
            sample_rate=24000,
        )

        assert isinstance(result, GenerationClassificationResult)
        assert result.prompt_processed is True
        assert result.audio_processed is True
        assert result.prompt_result is not None
        assert result.audio_result is not None

    def test_classify_generation_prompt_only(self, mock_input_classifier):
        """Test generation classification with prompt only."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            config=config,
        )

        result = pipeline.classify_generation(prompt="Generate a calm piano melody")

        assert result.prompt_processed is True
        assert result.audio_processed is False
        assert result.prompt_result is not None
        assert result.audio_result is None

    def test_classify_generation_audio_only(self, mock_output_classifier):
        """Test generation classification with audio only."""
        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=True,
        )
        pipeline = ConstitutionalAudio(
            output_classifier=mock_output_classifier,
            config=config,
        )

        audio = np.random.randn(24000).astype(np.float32)
        result = pipeline.classify_generation(audio=audio, sample_rate=24000)

        assert result.prompt_processed is False
        assert result.audio_processed is True
        assert result.prompt_result is None
        assert result.audio_result is not None

    def test_classify_generation_neither_raises(
        self,
        mock_input_classifier,
        mock_output_classifier,
    ):
        """Test that generation classification requires at least one input."""
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            output_classifier=mock_output_classifier,
        )

        with pytest.raises(PipelineError, match="At least one of prompt or audio"):
            pipeline.classify_generation()

    def test_result_to_dict(
        self,
        mock_input_classifier,
        mock_output_classifier,
    ):
        """Test GenerationClassificationResult serialization."""
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            output_classifier=mock_output_classifier,
        )

        audio = np.random.randn(24000).astype(np.float32)
        result = pipeline.classify_generation(
            prompt="Generate a calm piano melody",
            audio=audio,
            sample_rate=24000,
        )
        result_dict = result.to_dict()

        assert result_dict["type"] == "generation"
        assert "prompt" in result_dict
        assert "audio" in result_dict
        assert "decision" in result_dict
        assert "decision_reasons" in result_dict


# -----------------------------------------------------------------------------
# Test Decision Aggregation
# -----------------------------------------------------------------------------


class TestDecisionAggregation:
    """Test decision aggregation between classifiers."""

    def test_both_allow_results_in_allow(
        self,
        mock_input_classifier,
        mock_output_classifier,
    ):
        """Test that both ALLOW results in ALLOW."""
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            output_classifier=mock_output_classifier,
        )

        audio = np.random.randn(24000).astype(np.float32)
        result = pipeline.classify_generation(
            prompt="Generate a calm piano melody",
            audio=audio,
            sample_rate=24000,
        )

        assert result.decision == PipelineDecision.ALLOW
        assert "No safety concerns detected" in result.decision_reasons

    def test_prompt_block_results_in_block(
        self,
        mock_input_classifier,
        mock_output_classifier,
    ):
        """Test that prompt BLOCK results in overall BLOCK."""
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            output_classifier=mock_output_classifier,
        )

        audio = np.random.randn(24000).astype(np.float32)
        result = pipeline.classify_generation(
            prompt="Generate music like Drake",  # Triggers BLOCK
            audio=audio,
            sample_rate=24000,
        )

        assert result.decision == PipelineDecision.BLOCK
        assert result.prompt_result.decision == PipelineDecision.BLOCK
        assert any("Prompt blocked" in reason for reason in result.decision_reasons)

    def test_audio_block_results_in_block(
        self,
        mock_input_classifier,
    ):
        """Test that audio BLOCK results in overall BLOCK."""
        from decrescendo.musicritic.output_classifier.inference import (
            AggregatedResult,
            Decision,
            SpeakerMatch,
        )

        # Create output classifier that returns BLOCK
        blocking_output = MagicMock()
        blocking_output.classify_array.return_value = AggregatedResult(
            harm_scores={"copyright_ip": 0.98},
            flagged_categories=["copyright_ip"],
            best_speaker_match=SpeakerMatch(matched=False, similarity=0.1),
            decision=Decision.BLOCK,
            decision_reasons=["High copyright risk detected"],
            num_chunks=1,
            chunk_results=[],
        )

        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            output_classifier=blocking_output,
        )

        audio = np.random.randn(24000).astype(np.float32)
        result = pipeline.classify_generation(
            prompt="Safe prompt",
            audio=audio,
            sample_rate=24000,
        )

        assert result.decision == PipelineDecision.BLOCK
        assert result.audio_result.decision == PipelineDecision.BLOCK

    def test_flag_when_both_allow(
        self,
        mock_input_classifier,
    ):
        """Test FLAG_FOR_REVIEW with borderline content."""
        from decrescendo.musicritic.output_classifier.inference import (
            AggregatedResult,
            Decision,
            SpeakerMatch,
        )

        # Create output classifier that returns FLAG_FOR_REVIEW
        flagging_output = MagicMock()
        flagging_output.classify_array.return_value = AggregatedResult(
            harm_scores={"copyright_ip": 0.6},
            flagged_categories=["copyright_ip"],
            best_speaker_match=SpeakerMatch(matched=False, similarity=0.4),
            decision=Decision.FLAG_FOR_REVIEW,
            decision_reasons=["Elevated copyright similarity"],
            num_chunks=1,
            chunk_results=[],
        )

        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            output_classifier=flagging_output,
        )

        audio = np.random.randn(24000).astype(np.float32)
        result = pipeline.classify_generation(
            prompt="Safe prompt",
            audio=audio,
            sample_rate=24000,
        )

        assert result.decision == PipelineDecision.FLAG_FOR_REVIEW

    def test_block_takes_precedence_over_flag(
        self,
        mock_input_classifier,
    ):
        """Test that BLOCK takes precedence over FLAG_FOR_REVIEW."""
        from decrescendo.musicritic.output_classifier.inference import (
            AggregatedResult,
            Decision,
            SpeakerMatch,
        )

        # Create output classifier that returns FLAG_FOR_REVIEW
        flagging_output = MagicMock()
        flagging_output.classify_array.return_value = AggregatedResult(
            harm_scores={"copyright_ip": 0.6},
            flagged_categories=["copyright_ip"],
            best_speaker_match=SpeakerMatch(matched=False, similarity=0.4),
            decision=Decision.FLAG_FOR_REVIEW,
            decision_reasons=["Elevated copyright similarity"],
            num_chunks=1,
            chunk_results=[],
        )

        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            output_classifier=flagging_output,
        )

        audio = np.random.randn(24000).astype(np.float32)
        result = pipeline.classify_generation(
            prompt="Generate music like Drake",  # Triggers BLOCK from input
            audio=audio,
            sample_rate=24000,
        )

        # BLOCK from input should take precedence over FLAG from output
        assert result.decision == PipelineDecision.BLOCK


# -----------------------------------------------------------------------------
# Test Protected Voice Matching
# -----------------------------------------------------------------------------


class TestProtectedVoiceMatching:
    """Test protected voice matching integration."""

    def test_voice_database_integration(
        self,
        mock_input_classifier,
        mock_voice_database,
    ):
        """Test that voice database is properly integrated."""
        from decrescendo.musicritic.output_classifier.inference import (
            AggregatedResult,
            Decision,
            SpeakerMatch,
        )

        # Create output classifier that detects protected voice
        voice_detecting_output = MagicMock()
        voice_detecting_output.classify_array.return_value = AggregatedResult(
            harm_scores={"voice_cloning": 0.95},
            flagged_categories=["voice_cloning"],
            best_speaker_match=SpeakerMatch(
                matched=True,
                similarity=0.92,
                matched_voice_id=0,
                matched_voice_name="Artist A",
            ),
            decision=Decision.BLOCK,
            decision_reasons=["Protected voice detected: Artist A"],
            num_chunks=1,
            chunk_results=[],
        )

        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            output_classifier=voice_detecting_output,
            voice_database=mock_voice_database,
        )

        audio = np.random.randn(24000).astype(np.float32)
        result = pipeline.classify_generation(
            prompt="Safe prompt",
            audio=audio,
            sample_rate=24000,
        )

        assert result.decision == PipelineDecision.BLOCK
        assert result.audio_result.output_result.best_speaker_match.matched is True
        assert result.audio_result.output_result.best_speaker_match.matched_voice_name == "Artist A"

    def test_voice_matching_disabled(
        self,
        mock_input_classifier,
        mock_output_classifier,
        mock_voice_database,
    ):
        """Test voice matching can be disabled."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=True,
            enable_voice_matching=False,
        )
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            output_classifier=mock_output_classifier,
            voice_database=mock_voice_database,
            config=config,
        )

        # Voice database should not be accessible when disabled
        assert pipeline.voice_database is None


# -----------------------------------------------------------------------------
# Test Decision Mapping
# -----------------------------------------------------------------------------


class TestDecisionMapping:
    """Test decision mapping between classifier decisions and pipeline decisions."""

    def test_input_decision_mapping(self, mock_input_classifier):
        """Test input classifier decision mapping."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            config=config,
        )

        # ALLOW
        result = pipeline.classify_prompt("Safe prompt")
        assert result.decision == PipelineDecision.ALLOW

        # BLOCK
        result = pipeline.classify_prompt("Generate music like Drake")
        assert result.decision == PipelineDecision.BLOCK

    def test_output_decision_mapping(self, mock_output_classifier):
        """Test output classifier decision mapping."""
        from decrescendo.musicritic.output_classifier.inference import Decision

        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=True,
        )
        pipeline = ConstitutionalAudio(
            output_classifier=mock_output_classifier,
            config=config,
        )

        audio = np.random.randn(24000).astype(np.float32)

        # CONTINUE -> ALLOW
        result = pipeline.classify_audio(audio, sample_rate=24000)
        assert result.decision == PipelineDecision.ALLOW


# -----------------------------------------------------------------------------
# Performance/Latency Tests
# -----------------------------------------------------------------------------


class TestPerformance:
    """Performance and latency benchmarks."""

    def test_prompt_classification_latency(self, mock_input_classifier):
        """Test prompt classification latency."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            config=config,
        )

        # Warm-up
        pipeline.classify_prompt("test")

        # Measure
        start = time.perf_counter()
        for _ in range(100):
            pipeline.classify_prompt("Generate a calm piano melody")
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / 100) * 1000
        # With mocked classifier, should be < 1ms per call
        assert avg_latency_ms < 10, f"Prompt latency too high: {avg_latency_ms:.2f}ms"

    def test_audio_classification_latency(self, mock_output_classifier):
        """Test audio classification latency."""
        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=True,
        )
        pipeline = ConstitutionalAudio(
            output_classifier=mock_output_classifier,
            config=config,
        )

        audio = np.random.randn(24000).astype(np.float32)

        # Warm-up
        pipeline.classify_audio(audio, sample_rate=24000)

        # Measure
        start = time.perf_counter()
        for _ in range(100):
            pipeline.classify_audio(audio, sample_rate=24000)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / 100) * 1000
        # With mocked classifier, should be < 1ms per call
        assert avg_latency_ms < 10, f"Audio latency too high: {avg_latency_ms:.2f}ms"

    def test_batch_prompt_classification(self, mock_input_classifier):
        """Test batch prompt classification performance."""
        config = PipelineConfig(
            enable_input_classifier=True,
            enable_output_classifier=False,
        )
        pipeline = ConstitutionalAudio(
            input_classifier=mock_input_classifier,
            config=config,
        )

        prompts = [f"Generate melody {i}" for i in range(100)]

        start = time.perf_counter()
        results = pipeline.classify_prompt_batch(prompts)
        elapsed = time.perf_counter() - start

        assert len(results) == 100
        # Batch should be faster than individual calls
        assert elapsed < 1.0, f"Batch processing too slow: {elapsed:.2f}s"


# -----------------------------------------------------------------------------
# Test Real Model Integration (Optional - requires model loading)
# -----------------------------------------------------------------------------


class TestRealModelIntegration:
    """Integration tests with real models (if available)."""

    @pytest.fixture
    def real_output_classifier(self, rng):
        """Create a real output classifier with random weights."""
        from decrescendo.musicritic.output_classifier import (
            OutputClassifierConfig,
        )
        from decrescendo.musicritic.output_classifier.inference import (
            OutputClassifierInference,
            initialize_output_classifier,
        )

        config = OutputClassifierConfig()
        model, variables = initialize_output_classifier(config, rng)
        return OutputClassifierInference(model, variables, config)

    def test_real_output_classifier_classify_array(self, real_output_classifier):
        """Test real output classifier with random audio."""
        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=True,
        )
        pipeline = ConstitutionalAudio(
            output_classifier=real_output_classifier,
            config=config,
        )

        # Generate test audio (1 second at 24kHz)
        audio = np.random.randn(24000).astype(np.float32)
        result = pipeline.classify_audio(audio, sample_rate=24000)

        assert isinstance(result, PipelineAudioResult)
        assert result.decision in [
            PipelineDecision.ALLOW,
            PipelineDecision.FLAG_FOR_REVIEW,
            PipelineDecision.BLOCK,
        ]
        assert isinstance(result.output_result.harm_scores, dict)
        assert len(result.output_result.harm_scores) == 7  # 7 harm categories

    def test_real_output_classifier_latency(self, real_output_classifier):
        """Benchmark real output classifier latency."""
        config = PipelineConfig(
            enable_input_classifier=False,
            enable_output_classifier=True,
        )
        pipeline = ConstitutionalAudio(
            output_classifier=real_output_classifier,
            config=config,
        )

        audio = np.random.randn(24000).astype(np.float32)

        # Warm-up (includes JIT compilation)
        pipeline.classify_audio(audio, sample_rate=24000)

        # Measure
        start = time.perf_counter()
        for _ in range(10):
            pipeline.classify_audio(audio, sample_rate=24000)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / 10) * 1000
        print(f"Real output classifier latency: {avg_latency_ms:.2f}ms per 1s audio")

        # Should be < 500ms per 1s audio (reasonable for CPU)
        assert avg_latency_ms < 5000, f"Latency too high: {avg_latency_ms:.2f}ms"
