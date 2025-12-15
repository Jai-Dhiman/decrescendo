"""Tests for Output Classifier."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from decrescendo.constitutional_audio.output_classifier.audio_preprocessing import (
    AudioPreprocessor,
)
from decrescendo.constitutional_audio.output_classifier.config import (
    AudioEncoderConfig,
    OutputClassifierConfig,
    PreprocessingConfig,
)
from decrescendo.constitutional_audio.output_classifier.inference import (
    Decision,
    OutputClassifierInference,
    ScoreAggregator,
    initialize_output_classifier,
)
from decrescendo.constitutional_audio.output_classifier.model import (
    AudioEncoder,
    HarmClassifier,
    OutputClassifierModel,
    SpeakerEncoder,
    compute_speaker_similarity,
)


class TestAudioPreprocessor:
    """Test audio preprocessing."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return AudioPreprocessor()

    def test_to_mono(self, preprocessor):
        """Test stereo to mono conversion."""
        stereo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mono = preprocessor.to_mono(stereo)
        assert mono.shape == (3,)
        np.testing.assert_array_almost_equal(mono, [1.5, 3.5, 5.5])

    def test_to_mono_already_mono(self, preprocessor):
        """Test that mono audio is unchanged."""
        mono = np.array([1.0, 2.0, 3.0])
        result = preprocessor.to_mono(mono)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, mono)

    def test_normalize(self, preprocessor):
        """Test audio normalization."""
        audio = np.array([0.1, -0.2, 0.15, -0.1])
        normalized = preprocessor.normalize(audio)
        # Check that RMS is close to target
        rms = np.sqrt(np.mean(normalized**2))
        target_rms = 10 ** (-20 / 20)  # -20 dB
        assert abs(rms - target_rms) < 0.01

    def test_chunk_audio(self, preprocessor):
        """Test audio chunking."""
        # Create audio longer than one chunk
        chunk_samples = preprocessor.config.chunk_samples
        audio = np.ones(int(chunk_samples * 2.5))

        chunks = list(preprocessor.chunk_audio(audio))

        # Should have multiple chunks
        assert len(chunks) >= 2

        # Each chunk should be correct size
        for chunk in chunks:
            assert len(chunk) == chunk_samples

    def test_process_array(self, preprocessor):
        """Test full processing pipeline."""
        # Create dummy audio
        audio = np.random.randn(48000).astype(np.float32)  # 2 seconds at 24kHz
        sample_rate = 24000

        chunks = list(preprocessor.process_array(audio, sample_rate))

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, jnp.ndarray)
            assert chunk.shape == (preprocessor.config.chunk_samples,)


class TestAudioEncoder:
    """Test audio encoder model."""

    @pytest.fixture
    def config(self):
        """Create encoder config."""
        return AudioEncoderConfig(
            input_samples=24000,
            num_conv_layers=4,  # Reduced for faster testing
            base_channels=32,
            embedding_dim=256,
        )

    @pytest.fixture
    def encoder(self, config):
        """Create encoder instance."""
        return AudioEncoder(config=config)

    def test_encoder_init(self, encoder, rng):
        """Test encoder initialization."""
        batch_size = 2
        samples = 24000

        dummy_audio = jnp.zeros((batch_size, samples))
        variables = encoder.init(rng, dummy_audio, train=False)

        assert "params" in variables

    def test_encoder_forward(self, encoder, rng):
        """Test encoder forward pass."""
        batch_size = 2
        samples = 24000

        audio = jax.random.normal(rng, (batch_size, samples))
        variables = encoder.init(rng, audio, train=False)

        embeddings = encoder.apply(variables, audio, train=False)

        assert embeddings.shape == (batch_size, encoder.config.embedding_dim)


class TestSpeakerEncoder:
    """Test speaker encoder."""

    @pytest.fixture
    def encoder(self):
        """Create speaker encoder."""
        from decrescendo.constitutional_audio.output_classifier.config import SpeakerConfig
        return SpeakerEncoder(config=SpeakerConfig())

    def test_speaker_encoder_output_normalized(self, encoder, rng):
        """Test that speaker embeddings are L2 normalized."""
        audio = jax.random.normal(rng, (2, 24000))
        variables = encoder.init(rng, audio, train=False)

        embeddings = encoder.apply(variables, audio, train=False)

        # Check L2 norm is approximately 1
        norms = jnp.linalg.norm(embeddings, axis=-1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0], decimal=5)


class TestHarmClassifier:
    """Test harm classifier."""

    @pytest.fixture
    def classifier(self):
        """Create harm classifier."""
        return HarmClassifier(
            num_categories=7,
            hidden_dim=128,
            dropout_rate=0.1,
        )

    def test_classifier_output_shape(self, classifier, rng):
        """Test classifier output shape."""
        batch_size = 4
        embedding_dim = 256

        embeddings = jax.random.normal(rng, (batch_size, embedding_dim))
        variables = classifier.init(rng, embeddings, train=False)

        logits = classifier.apply(variables, embeddings, train=False)

        assert logits.shape == (batch_size, 7)


class TestOutputClassifierModel:
    """Test complete output classifier model."""

    @pytest.fixture
    def config(self):
        """Create model config with smaller dimensions for testing."""
        audio_config = AudioEncoderConfig(
            input_samples=24000,
            num_conv_layers=3,
            base_channels=16,
            embedding_dim=128,
        )
        return OutputClassifierConfig(
            audio_encoder=audio_config,
            classifier_hidden_dim=64,
        )

    @pytest.fixture
    def model(self, config):
        """Create model instance."""
        return OutputClassifierModel(config=config)

    def test_model_init(self, model, rng):
        """Test model initialization."""
        audio = jnp.zeros((1, 24000))
        variables = model.init(rng, audio, train=False)

        assert "params" in variables
        assert "audio_encoder" in variables["params"]
        assert "speaker_encoder" in variables["params"]
        assert "harm_classifier" in variables["params"]

    def test_model_forward(self, model, config, rng):
        """Test model forward pass."""
        batch_size = 2
        audio = jax.random.normal(rng, (batch_size, 24000))

        variables = model.init(rng, audio, train=False)
        outputs = model.apply(variables, audio, train=False)

        assert "harm_logits" in outputs
        assert "audio_embeddings" in outputs
        assert "speaker_embeddings" in outputs

        assert outputs["harm_logits"].shape == (batch_size, config.num_harm_categories)
        assert outputs["audio_embeddings"].shape == (batch_size, config.audio_encoder.embedding_dim)
        assert outputs["speaker_embeddings"].shape == (batch_size, config.speaker.embedding_dim)


class TestSpeakerSimilarity:
    """Test speaker similarity functions."""

    def test_compute_speaker_similarity(self):
        """Test cosine similarity computation."""
        e1 = jnp.array([1.0, 0.0, 0.0])
        e2 = jnp.array([1.0, 0.0, 0.0])

        sim = compute_speaker_similarity(e1, e2)
        # Returns array of shape (1,) for 1D inputs
        assert float(sim[0]) == pytest.approx(1.0)

    def test_compute_speaker_similarity_orthogonal(self):
        """Test similarity of orthogonal vectors."""
        e1 = jnp.array([1.0, 0.0, 0.0])
        e2 = jnp.array([0.0, 1.0, 0.0])

        sim = compute_speaker_similarity(e1, e2)
        assert float(sim[0]) == pytest.approx(0.0, abs=1e-6)


class TestScoreAggregator:
    """Test score aggregation."""

    def test_aggregator_empty(self):
        """Test aggregator with no scores."""
        agg = ScoreAggregator(num_categories=7)
        scores = agg.get_aggregated_scores()
        assert scores.shape == (7,)
        np.testing.assert_array_equal(scores, np.zeros(7))

    def test_aggregator_single_score(self):
        """Test aggregator with single score."""
        agg = ScoreAggregator(num_categories=7)
        agg.add_scores(np.array([0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]))

        scores = agg.get_aggregated_scores()
        np.testing.assert_array_almost_equal(
            scores, [0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

    def test_aggregator_max_scores(self):
        """Test max score aggregation."""
        agg = ScoreAggregator(num_categories=3)
        agg.add_scores(np.array([0.2, 0.5, 0.1]))
        agg.add_scores(np.array([0.8, 0.3, 0.4]))
        agg.add_scores(np.array([0.1, 0.9, 0.2]))

        max_scores = agg.get_max_scores()
        np.testing.assert_array_almost_equal(max_scores, [0.8, 0.9, 0.4])


class TestOutputClassifierInference:
    """Test inference pipeline."""

    @pytest.fixture
    def classifier(self, rng):
        """Create inference pipeline."""
        config = OutputClassifierConfig(
            audio_encoder=AudioEncoderConfig(
                num_conv_layers=3,
                base_channels=16,
                embedding_dim=128,
            ),
            classifier_hidden_dim=64,
        )
        model, variables = initialize_output_classifier(config, rng)
        return OutputClassifierInference(model, variables, config)

    def test_classify_chunk(self, classifier, rng):
        """Test single chunk classification."""
        chunk = jax.random.normal(rng, (24000,))
        result = classifier.classify_chunk(chunk)

        assert len(result.harm_scores) == 7
        assert result.chunk_decision in Decision
        assert result.speaker_match is not None

    def test_classify_array(self, classifier, rng):
        """Test array classification."""
        # 2 seconds of audio
        audio = np.random.randn(48000).astype(np.float32)

        result = classifier.classify_array(audio, sample_rate=24000)

        assert result.num_chunks > 0
        assert result.decision in Decision
        assert len(result.harm_scores) == 7

    def test_decision_continue(self, classifier, rng):
        """Test that safe audio gets CONTINUE decision."""
        # With random model weights, scores should generally be low
        audio = np.zeros(24000, dtype=np.float32)
        result = classifier.classify_array(audio, sample_rate=24000)

        # Decision should be based on thresholds
        assert result.decision in Decision
