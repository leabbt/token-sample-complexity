"""Load BigBird or BERT in inference mode and extract layer-0 embeddings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, BertModel, BigBirdModel


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    family: str        # "bigbird" or "bert"
    hidden_size: int
    num_attention_heads: int
    max_position_embeddings: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "bigbird-base":  ModelSpec("google/bigbird-roberta-base",  "bigbird", 768,  12, 4096),
    "bigbird-large": ModelSpec("google/bigbird-roberta-large", "bigbird", 1024, 16, 4096),
    "bert-base":     ModelSpec("bert-base-uncased",            "bert",    768,  12, 512),
    "bert-large":    ModelSpec("bert-large-uncased",           "bert",    1024, 16, 512),
}


def get_spec(model_key: str) -> ModelSpec:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model key: {model_key!r}. Choices: {list(MODEL_SPECS)}")
    return MODEL_SPECS[model_key]


def default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(
    model_key: str = "bigbird-base",
    *,
    attention_type: str = "block_sparse",
    device: Optional[torch.device] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    """Load a BigBird or BERT model for inference.

    `attention_type` is forwarded to BigBird's `set_attention_type`. It is
    ignored for BERT.
    """
    spec = get_spec(model_key)
    if device is None:
        device = default_device()

    if spec.family == "bigbird":
        model = BigBirdModel.from_pretrained(spec.hf_id, torch_dtype=torch_dtype)
        model.set_attention_type(attention_type)
    elif spec.family == "bert":
        model = BertModel.from_pretrained(spec.hf_id, torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Unsupported family: {spec.family}")

    model.eval()
    model.to(device)
    return model


def load_tokenizer(model_key: str = "bigbird-base"):
    spec = get_spec(model_key)
    return AutoTokenizer.from_pretrained(spec.hf_id)


@torch.no_grad()
def get_layer0_embeddings(model, tokens) -> torch.Tensor:
    """Token + position (+ token-type) embeddings after LayerNorm.

    These are the input to the first transformer layer for both BigBird and BERT.
    `tokens` is a dict containing `input_ids` (and optionally `token_type_ids`).
    """
    input_ids = tokens["input_ids"]
    embeds = model.embeddings.word_embeddings(input_ids)
    seq_length = input_ids.shape[1]
    position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
    embeds = embeds + model.embeddings.position_embeddings(position_ids)
    if hasattr(model.embeddings, "token_type_embeddings"):
        token_type_ids = tokens.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embeds = embeds + model.embeddings.token_type_embeddings(token_type_ids)
    return model.embeddings.LayerNorm(embeds)
