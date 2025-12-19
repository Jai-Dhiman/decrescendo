"""Tests for Prompt Adherence evaluator."""

import numpy as np
import pytest

from decrescendo.musicritic.dimensions import (
    DimensionCategory,
    DimensionResult,
    QualityDimension,
)
from decrescendo.musicritic.dimensions.prompt_adherence import (
    PromptAdherenceConfig,
    PromptAdherenceEvaluator,
    PromptRequiredError,
)


class TestPromptAdherenceEvaluatorAttributes:
    """Tests for evaluator attributes."""

    def test_dimension_attribute(self):
        """Test that dimension attribute is correct."""
        evaluator = PromptAdherenceEvaluator()

        assert evaluator.dimension == QualityDimension.PROMPT_ADHERENCE

    def test_category_attribute(self):
        """Test that category attribute is correct."""
        evaluator = PromptAdherenceEvaluator()

        assert evaluator.category == DimensionCategory.QUALITY


class TestPromptAdherenceEvaluatorValidation:
    """Tests for input validation."""

    def test_evaluate_without_prompt_raises(self, clap_encoder, sample_audio_48k):
        """Test that missing prompt raises PromptRequiredError."""
        evaluator = PromptAdherenceEvaluator(encoder=clap_encoder)
        audio, sample_rate = sample_audio_48k

        with pytest.raises(PromptRequiredError) as exc_info:
            evaluator.evaluate(audio, sample_rate, prompt=None)

        assert "Prompt is required" in str(exc_info.value)

    def test_evaluate_empty_prompt_raises(self, clap_encoder, sample_audio_48k):
        """Test that empty prompt raises PromptRequiredError."""
        evaluator = PromptAdherenceEvaluator(encoder=clap_encoder)
        audio, sample_rate = sample_audio_48k

        with pytest.raises(PromptRequiredError):
            evaluator.evaluate(audio, sample_rate, prompt="")

        with pytest.raises(PromptRequiredError):
            evaluator.evaluate(audio, sample_rate, prompt="   ")


class TestPromptAdherenceEvaluatorEvaluation:
    """Tests for evaluation functionality."""

    def test_evaluate_with_prompt(self, clap_encoder, sample_audio_48k, sample_prompt):
        """Test successful evaluation with prompt."""
        evaluator = PromptAdherenceEvaluator(encoder=clap_encoder)
        audio, sample_rate = sample_audio_48k

        result = evaluator.evaluate(audio, sample_rate, prompt=sample_prompt)

        assert isinstance(result, DimensionResult)
        assert result.dimension == QualityDimension.PROMPT_ADHERENCE
        assert 0.0 <= result.score <= 1.0

    def test_evaluate_result_structure(self, clap_encoder, sample_audio_48k, sample_prompt):
        """Test that result contains all expected fields."""
        evaluator = PromptAdherenceEvaluator(encoder=clap_encoder)
        audio, sample_rate = sample_audio_48k

        result = evaluator.evaluate(audio, sample_rate, prompt=sample_prompt)

        # Check sub_scores
        assert "clap_similarity" in result.sub_scores

        # Check metadata
        assert "prompt" in result.metadata
        assert result.metadata["prompt"] == sample_prompt
        assert "adherence_level" in result.metadata
        assert result.metadata["adherence_level"] in ["strong", "moderate", "poor"]

        # Check explanation
        assert len(result.explanation) > 0
        assert "adherence" in result.explanation.lower()

    def test_evaluate_confidence_calculation(
        self, clap_encoder, sample_audio_48k, sample_prompt
    ):
        """Test that confidence is calculated reasonably."""
        evaluator = PromptAdherenceEvaluator(encoder=clap_encoder)
        audio, sample_rate = sample_audio_48k

        result = evaluator.evaluate(audio, sample_rate, prompt=sample_prompt)

        # Confidence should be between 0.5 and 1.0
        assert 0.5 <= result.confidence <= 1.0

    def test_evaluate_stereo_handled_by_base_class(
        self, clap_encoder, sample_stereo_audio, sample_prompt
    ):
        """Test that stereo audio is handled by base class conversion."""
        evaluator = PromptAdherenceEvaluator(encoder=clap_encoder)
        audio, sample_rate = sample_stereo_audio

        # Stereo audio should be converted to mono by base class
        result = evaluator.evaluate(audio, sample_rate, prompt=sample_prompt)

        assert isinstance(result, DimensionResult)
        assert 0.0 <= result.score <= 1.0


class TestPromptAdherenceEvaluatorClassification:
    """Tests for adherence classification."""

    def test_classify_adherence_strong(self, clap_encoder):
        """Test strong adherence classification."""
        evaluator = PromptAdherenceEvaluator(encoder=clap_encoder)

        # Score above strong threshold
        level = evaluator._classify_adherence(0.8)
        assert level == "strong"

        level = evaluator._classify_adherence(0.7)
        assert level == "strong"

    def test_classify_adherence_moderate(self, clap_encoder):
        """Test moderate adherence classification."""
        evaluator = PromptAdherenceEvaluator(encoder=clap_encoder)

        # Score between moderate and strong thresholds
        level = evaluator._classify_adherence(0.6)
        assert level == "moderate"

        level = evaluator._classify_adherence(0.5)
        assert level == "moderate"

    def test_classify_adherence_poor(self, clap_encoder):
        """Test poor adherence classification."""
        evaluator = PromptAdherenceEvaluator(encoder=clap_encoder)

        # Score below moderate threshold
        level = evaluator._classify_adherence(0.4)
        assert level == "poor"

        level = evaluator._classify_adherence(0.0)
        assert level == "poor"


class TestPromptAdherenceEvaluatorCustomConfig:
    """Tests for custom configuration."""

    def test_custom_thresholds(self, clap_encoder):
        """Test evaluator with custom thresholds."""
        config = PromptAdherenceConfig(
            strong_adherence_threshold=0.9,
            moderate_adherence_threshold=0.7,
        )
        evaluator = PromptAdherenceEvaluator(config=config, encoder=clap_encoder)

        # With higher thresholds, 0.8 should be moderate, not strong
        level = evaluator._classify_adherence(0.8)
        assert level == "moderate"

        # And 0.6 should be poor, not moderate
        level = evaluator._classify_adherence(0.6)
        assert level == "poor"
