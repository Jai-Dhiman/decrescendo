"""Flax Linen model architecture for the Input Classifier."""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import InputClassifierConfig, TransformerConfig


class TransformerEmbeddings(nn.Module):
    """Embeddings for transformer: word + position + token_type."""

    config: TransformerConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        token_type_ids: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)

        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        word_embeds = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="word_embeddings",
        )(input_ids)

        position_embeds = nn.Embed(
            num_embeddings=self.config.max_position_embeddings,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="position_embeddings",
        )(position_ids)

        token_type_embeds = nn.Embed(
            num_embeddings=self.config.type_vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="token_type_embeddings",
        )(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)(embeddings)
        embeddings = nn.Dropout(rate=self.config.hidden_dropout_prob)(
            embeddings, deterministic=deterministic
        )

        return embeddings


class TransformerSelfAttention(nn.Module):
    """Multi-head self attention mechanism."""

    config: TransformerConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        batch_size, seq_length, _ = hidden_states.shape
        head_dim = self.config.head_dim
        num_heads = self.config.num_attention_heads

        # Project to Q, K, V
        query = nn.Dense(
            self.config.hidden_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="query",
        )(hidden_states)

        key = nn.Dense(
            self.config.hidden_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="key",
        )(hidden_states)

        value = nn.Dense(
            self.config.hidden_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="value",
        )(hidden_states)

        # Reshape for multi-head attention: (batch, seq, hidden) -> (batch, heads, seq, head_dim)
        query = query.reshape(batch_size, seq_length, num_heads, head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch_size, seq_length, num_heads, head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch_size, seq_length, num_heads, head_dim).transpose(0, 2, 1, 3)

        # Compute attention scores
        attn_weights = jnp.matmul(query, key.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(rate=self.config.attention_probs_dropout_prob)(
            attn_weights, deterministic=deterministic
        )

        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, value)

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, -1)

        # Output projection
        attn_output = nn.Dense(
            self.config.hidden_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="output",
        )(attn_output)

        return attn_output


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward network."""

    config: TransformerConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # Self-attention with residual connection
        attn_output = TransformerSelfAttention(
            config=self.config, dtype=self.dtype, name="attention"
        )(hidden_states, attention_mask, deterministic)

        attn_output = nn.Dropout(rate=self.config.hidden_dropout_prob)(
            attn_output, deterministic=deterministic
        )
        hidden_states = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype, name="attention_layer_norm"
        )(hidden_states + attn_output)

        # Feed-forward network with residual connection
        ffn_output = nn.Dense(
            self.config.intermediate_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="intermediate",
        )(hidden_states)
        ffn_output = jax.nn.gelu(ffn_output)

        ffn_output = nn.Dense(
            self.config.hidden_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="output",
        )(ffn_output)

        ffn_output = nn.Dropout(rate=self.config.hidden_dropout_prob)(
            ffn_output, deterministic=deterministic
        )
        hidden_states = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype, name="output_layer_norm"
        )(hidden_states + ffn_output)

        return hidden_states


class TransformerEncoder(nn.Module):
    """Stack of transformer layers."""

    config: TransformerConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        for i in range(self.config.num_hidden_layers):
            hidden_states = TransformerLayer(
                config=self.config, dtype=self.dtype, name=f"layer_{i}"
            )(hidden_states, attention_mask, deterministic)

        return hidden_states


class Pooler(nn.Module):
    """Pooler that extracts [CLS] token representation."""

    hidden_size: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        # Take [CLS] token (first token)
        cls_hidden = hidden_states[:, 0]
        pooled = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
        )(cls_hidden)
        return jax.nn.tanh(pooled)


class ClassificationHead(nn.Module):
    """Classification head with dropout and dense layer."""

    num_classes: int
    dropout_rate: float
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self, pooled_output: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        x = nn.Dropout(rate=self.dropout_rate)(pooled_output, deterministic=deterministic)
        return nn.Dense(
            self.num_classes,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
        )(x)


class InputClassifier(nn.Module):
    """Constitutional Audio Input Classifier.

    Analyzes text prompts to detect:
    - Intent: benign / suspicious / malicious
    - Artist requests: none / named_artist / style_reference
    - Voice requests: none / celebrity / politician
    - Policy violations: 7 harm categories (multi-label)

    Args:
        config: InputClassifierConfig with model hyperparameters
        dtype: Data type for computations (default: float32)
    """

    config: InputClassifierConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        token_type_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> dict[str, jnp.ndarray]:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_length)
            attention_mask: Mask of shape (batch, seq_length), 1 for real tokens, 0 for padding
            token_type_ids: Token type IDs of shape (batch, seq_length)
            deterministic: If True, disable dropout (for inference)

        Returns:
            Dictionary with logits for each classification head:
            - intent_logits: (batch, num_intent_classes)
            - artist_logits: (batch, num_artist_classes)
            - voice_logits: (batch, num_voice_classes)
            - policy_logits: (batch, num_policy_labels)
            - pooled_output: (batch, hidden_size)
            - last_hidden_state: (batch, seq_length, hidden_size)
        """
        transformer_config = self.config.transformer
        cls_config = self.config.classification

        # Create extended attention mask for transformer
        # Convert (batch, seq) -> (batch, 1, 1, seq) for broadcasting in attention
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            # Convert 0/1 mask to additive mask (-inf for masked positions)
            extended_attention_mask = (1.0 - extended_attention_mask) * jnp.finfo(self.dtype).min
        else:
            extended_attention_mask = None

        # Embeddings
        hidden_states = TransformerEmbeddings(
            config=transformer_config, dtype=self.dtype, name="embeddings"
        )(input_ids, token_type_ids=token_type_ids, deterministic=deterministic)

        # Encoder
        hidden_states = TransformerEncoder(
            config=transformer_config, dtype=self.dtype, name="encoder"
        )(hidden_states, attention_mask=extended_attention_mask, deterministic=deterministic)

        # Pooling
        pooled_output = Pooler(
            hidden_size=transformer_config.hidden_size, dtype=self.dtype, name="pooler"
        )(hidden_states)

        # Classification heads
        intent_logits = ClassificationHead(
            num_classes=cls_config.num_intent_classes,
            dropout_rate=cls_config.classifier_dropout,
            dtype=self.dtype,
            name="intent_classifier",
        )(pooled_output, deterministic=deterministic)

        artist_logits = ClassificationHead(
            num_classes=cls_config.num_artist_classes,
            dropout_rate=cls_config.classifier_dropout,
            dtype=self.dtype,
            name="artist_classifier",
        )(pooled_output, deterministic=deterministic)

        voice_logits = ClassificationHead(
            num_classes=cls_config.num_voice_classes,
            dropout_rate=cls_config.classifier_dropout,
            dtype=self.dtype,
            name="voice_classifier",
        )(pooled_output, deterministic=deterministic)

        policy_logits = ClassificationHead(
            num_classes=cls_config.num_policy_labels,
            dropout_rate=cls_config.classifier_dropout,
            dtype=self.dtype,
            name="policy_classifier",
        )(pooled_output, deterministic=deterministic)

        return {
            "intent_logits": intent_logits,
            "artist_logits": artist_logits,
            "voice_logits": voice_logits,
            "policy_logits": policy_logits,
            "pooled_output": pooled_output,
            "last_hidden_state": hidden_states,
        }
