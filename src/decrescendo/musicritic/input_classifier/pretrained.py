"""Utilities for loading pretrained weights from HuggingFace."""

from typing import Any

import jax
import jax.numpy as jnp
from transformers import FlaxRobertaModel, RobertaTokenizerFast

from .config import InputClassifierConfig
from .model import InputClassifier


class PretrainedLoadError(Exception):
    """Raised when pretrained weights cannot be loaded."""

    pass


def load_pretrained_roberta(
    model_name: str = "roberta-base",
) -> tuple[dict[str, Any], RobertaTokenizerFast]:
    """Load pretrained RoBERTa weights and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model identifier (e.g., "roberta-base", "roberta-large")

    Returns:
        Tuple of (params_dict, tokenizer)

    Raises:
        PretrainedLoadError: If the model or tokenizer cannot be loaded
    """
    try:
        hf_model = FlaxRobertaModel.from_pretrained(model_name)
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        return hf_model.params, tokenizer
    except Exception as e:
        raise PretrainedLoadError(f"Failed to load pretrained model '{model_name}': {e}") from e


def _transfer_embedding_params(
    hf_embeddings: dict[str, Any],
    our_embeddings: dict[str, Any],
) -> dict[str, Any]:
    """Transfer embedding parameters from HuggingFace to our model."""
    result = dict(our_embeddings)

    # Word embeddings
    if "word_embeddings" in hf_embeddings:
        result["word_embeddings"] = {"embedding": hf_embeddings["word_embeddings"]["embedding"]}

    # Position embeddings
    if "position_embeddings" in hf_embeddings:
        result["position_embeddings"] = {
            "embedding": hf_embeddings["position_embeddings"]["embedding"]
        }

    # Token type embeddings
    if "token_type_embeddings" in hf_embeddings:
        result["token_type_embeddings"] = {
            "embedding": hf_embeddings["token_type_embeddings"]["embedding"]
        }

    # Layer norm
    if "LayerNorm" in hf_embeddings:
        result["LayerNorm_0"] = {
            "scale": hf_embeddings["LayerNorm"]["scale"],
            "bias": hf_embeddings["LayerNorm"]["bias"],
        }

    return result


def _transfer_attention_params(
    hf_attention: dict[str, Any],
    our_attention: dict[str, Any],
) -> dict[str, Any]:
    """Transfer attention parameters from HuggingFace to our model."""
    result = dict(our_attention)

    hf_self = hf_attention.get("self", {})

    # Query, Key, Value projections
    for proj in ["query", "key", "value"]:
        if proj in hf_self:
            result[proj] = {
                "kernel": hf_self[proj]["kernel"],
                "bias": hf_self[proj]["bias"],
            }

    # Output projection
    if "output" in hf_attention and "dense" in hf_attention["output"]:
        result["output"] = {
            "kernel": hf_attention["output"]["dense"]["kernel"],
            "bias": hf_attention["output"]["dense"]["bias"],
        }

    return result


def _transfer_layer_params(
    hf_layer: dict[str, Any],
    our_layer: dict[str, Any],
) -> dict[str, Any]:
    """Transfer single transformer layer parameters."""
    result = dict(our_layer)

    # Attention
    if "attention" in hf_layer and "attention" in our_layer:
        result["attention"] = _transfer_attention_params(
            hf_layer["attention"], our_layer["attention"]
        )

    # Attention layer norm (in HF it's under attention/output/LayerNorm)
    if (
        "attention" in hf_layer
        and "output" in hf_layer["attention"]
        and "LayerNorm" in hf_layer["attention"]["output"]
    ):
        hf_ln = hf_layer["attention"]["output"]["LayerNorm"]
        result["attention_layer_norm"] = {
            "scale": hf_ln["scale"],
            "bias": hf_ln["bias"],
        }

    # Intermediate (FFN first layer)
    if "intermediate" in hf_layer and "dense" in hf_layer["intermediate"]:
        result["intermediate"] = {
            "kernel": hf_layer["intermediate"]["dense"]["kernel"],
            "bias": hf_layer["intermediate"]["dense"]["bias"],
        }

    # Output (FFN second layer)
    if "output" in hf_layer and "dense" in hf_layer["output"]:
        result["output"] = {
            "kernel": hf_layer["output"]["dense"]["kernel"],
            "bias": hf_layer["output"]["dense"]["bias"],
        }

    # Output layer norm
    if "output" in hf_layer and "LayerNorm" in hf_layer["output"]:
        hf_ln = hf_layer["output"]["LayerNorm"]
        result["output_layer_norm"] = {
            "scale": hf_ln["scale"],
            "bias": hf_ln["bias"],
        }

    return result


def _transfer_encoder_params(
    hf_encoder: dict[str, Any],
    our_encoder: dict[str, Any],
    num_layers: int,
) -> dict[str, Any]:
    """Transfer encoder (all layers) parameters."""
    result = dict(our_encoder)

    if "layer" not in hf_encoder:
        return result

    hf_layers = hf_encoder["layer"]

    for i in range(num_layers):
        layer_key = f"layer_{i}"
        hf_layer_key = str(i)

        if hf_layer_key in hf_layers and layer_key in our_encoder:
            result[layer_key] = _transfer_layer_params(hf_layers[hf_layer_key], our_encoder[layer_key])

    return result


def _transfer_pooler_params(
    hf_pooler: dict[str, Any],
    our_pooler: dict[str, Any],
) -> dict[str, Any]:
    """Transfer pooler parameters."""
    result = dict(our_pooler)

    if "dense" in hf_pooler:
        result["Dense_0"] = {
            "kernel": hf_pooler["dense"]["kernel"],
            "bias": hf_pooler["dense"]["bias"],
        }

    return result


def initialize_from_pretrained(
    config: InputClassifierConfig,
    rng: jax.Array,
    max_length: int = 512,
) -> tuple[InputClassifier, dict[str, Any], RobertaTokenizerFast]:
    """Initialize InputClassifier with pretrained RoBERTa weights.

    This function:
    1. Loads pretrained RoBERTa weights from HuggingFace
    2. Creates an InputClassifier model
    3. Initializes the model with random weights
    4. Transfers pretrained weights to the backbone
    5. Keeps classification heads randomly initialized

    Args:
        config: Model configuration
        rng: JAX random key for initialization
        max_length: Maximum sequence length for initialization

    Returns:
        Tuple of (model, params, tokenizer)

    Raises:
        PretrainedLoadError: If pretrained weights cannot be loaded
    """
    # Load pretrained weights
    pretrained_params, tokenizer = load_pretrained_roberta(config.pretrained_model_name)

    # Create model
    model = InputClassifier(config=config)

    # Initialize with random weights
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.ones((1, max_length), dtype=jnp.int32)
    variables = model.init(init_rng, dummy_input, deterministic=True)
    params = variables["params"]

    # Transfer pretrained weights
    if config.use_pretrained:
        # The HuggingFace model has params under 'roberta' key
        hf_roberta = pretrained_params.get("roberta", pretrained_params)

        # Transfer embeddings
        if "embeddings" in hf_roberta and "embeddings" in params:
            params["embeddings"] = _transfer_embedding_params(
                hf_roberta["embeddings"], params["embeddings"]
            )

        # Transfer encoder
        if "encoder" in hf_roberta and "encoder" in params:
            params["encoder"] = _transfer_encoder_params(
                hf_roberta["encoder"],
                params["encoder"],
                config.transformer.num_hidden_layers,
            )

        # Transfer pooler
        if "pooler" in hf_roberta and "pooler" in params:
            params["pooler"] = _transfer_pooler_params(hf_roberta["pooler"], params["pooler"])

    return model, params, tokenizer


def get_tokenizer(model_name: str = "roberta-base") -> RobertaTokenizerFast:
    """Load just the tokenizer without model weights.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        RobertaTokenizerFast tokenizer

    Raises:
        PretrainedLoadError: If tokenizer cannot be loaded
    """
    try:
        return RobertaTokenizerFast.from_pretrained(model_name)
    except Exception as e:
        raise PretrainedLoadError(f"Failed to load tokenizer '{model_name}': {e}") from e
