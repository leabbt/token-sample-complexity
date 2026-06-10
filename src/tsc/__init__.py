"""token-sample-complexity: shared utilities for the paper experiments."""

__version__ = "0.1.0"

from .attention import (
    chunked_attention_auto,
    chunked_attention_cpu_offload,
    chunked_attention_memory_efficient,
    cross_attention_chunked,
    full_attention_output_chunked,
    get_mean_cov,
)
from .fit import WLSFit, fit_convergence, wls_fit_full
from .model_utils import (
    MODEL_SPECS,
    ModelSpec,
    default_device,
    get_layer0_embeddings,
    get_spec,
    load_model,
    load_tokenizer,
)
from .sampling import sample_iid, sample_window_and_random
from .text_loaders import load_text

__all__ = [
    "MODEL_SPECS",
    "ModelSpec",
    "WLSFit",
    "chunked_attention_auto",
    "chunked_attention_cpu_offload",
    "chunked_attention_memory_efficient",
    "cross_attention_chunked",
    "default_device",
    "full_attention_output_chunked",
    "fit_convergence",
    "get_layer0_embeddings",
    "get_mean_cov",
    "get_spec",
    "load_model",
    "load_text",
    "load_tokenizer",
    "sample_iid",
    "sample_window_and_random",
    "wls_fit_full",
]
