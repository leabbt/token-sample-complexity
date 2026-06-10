#!/usr/bin/env python3
"""
Text Loading Module for BigBird Convergence Experiments.

This module provides various text loading strategies to study the effect of
text distribution on attention convergence rates.

Strategies:
    1. Single source (wiki, news, books, code, reddit, scientific)
    2. Multilingual (en, de, fr, zh, es, ar)
    3. Mixed provenance (controlled mixtures of sources)
    4. Shuffled text (sentence, paragraph, or full word shuffle)
    5. Synthetic embeddings (for theoretical validation)

Usage:
    from bigbird_mc.text_loaders import load_text, get_available_sources

    # Single source
    text = load_text(max_length=4096, source="wiki")

    # Multilingual
    text = load_text(max_length=4096, source="wiki", language="de")

    # Mixed sources
    text = load_text(max_length=4096, source="mixed",
                     mix_config={"wiki": 0.5, "news": 0.5})

    # Shuffled
    text = load_text(max_length=4096, source="wiki", shuffle="sentence")
"""

import json
import math
import os
import random
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from datasets import load_dataset, Dataset, load_from_disk
import warnings

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


# ============== CONFIGURATION ==============

# Available text sources with their HuggingFace dataset specifications
TEXT_SOURCES = {
    # English sources
    "wiki": {
        "dataset": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "text_field": "text",
        "description": "Wikipedia Good/Featured articles (English)",
    },
    "news": {
        "dataset": "cc_news",
        "config": None,
        "split": "train",
        "text_field": "text",
        "description": "Common Crawl News articles",
    },
    "books": {
        "dataset": "bookcorpus",
        "config": None,
        "split": "train",
        "text_field": "text",
        "description": "BookCorpus - fiction books",
    },
    "code": {
        "dataset": "codeparrot/github-code",
        "config": "all-all",
        "split": "train",
        "text_field": "code",
        "description": "GitHub code (mixed languages)",
        "streaming": True,  # Large dataset, use streaming
    },
    "reddit": {
        "dataset": "reddit",
        "config": None,
        "split": "train",
        "text_field": "content",
        "description": "Reddit comments (casual/informal)",
    },
    "scientific": {
        "dataset": "scientific_papers",
        "config": "arxiv",
        "split": "train",
        "text_field": "article",
        "description": "ArXiv scientific papers",
    },
    "openwebtext": {
        "dataset": "openwebtext",
        "config": None,
        "split": "train",
        "text_field": "text",
        "description": "OpenWebText - web pages",
    },
    "mr_niah": {
        "dataset": "MiniMaxAI/MR-NIAH",
        "config": None,
        "split": "train",
        "text_field": "messages",  # Special handling - extract from dialogue
        "description": "MR-NIAH benchmark haystack text",
        "special_loader": True,  # Flag for custom loading logic
    },
}

# MR-NIAH available token lengths
MR_NIAH_TOKEN_LENGTHS = [2048, 10240, 20480, 51200, 102400, 204800, 512000, 1024000]

# Multilingual Wikipedia sources
MULTILINGUAL_SOURCES = {
    "en": {
        "dataset": "wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "text_field": "text",
        "description": "English Wikipedia",
    },
    "de": {
        "dataset": "wikipedia",
        "config": "20220301.de",
        "split": "train",
        "text_field": "text",
        "description": "German Wikipedia",
    },
    "fr": {
        "dataset": "wikipedia",
        "config": "20220301.fr",
        "split": "train",
        "text_field": "text",
        "description": "French Wikipedia",
    },
    "es": {
        "dataset": "wikipedia",
        "config": "20220301.es",
        "split": "train",
        "text_field": "text",
        "description": "Spanish Wikipedia",
    },
    "zh": {
        "dataset": "wikipedia",
        "config": "20220301.zh",
        "split": "train",
        "text_field": "text",
        "description": "Chinese Wikipedia",
    },
    "ar": {
        "dataset": "wikipedia",
        "config": "20220301.ar",
        "split": "train",
        "text_field": "text",
        "description": "Arabic Wikipedia",
    },
    "ru": {
        "dataset": "wikipedia",
        "config": "20220301.ru",
        "split": "train",
        "text_field": "text",
        "description": "Russian Wikipedia",
    },
    "ja": {
        "dataset": "wikipedia",
        "config": "20220301.ja",
        "split": "train",
        "text_field": "text",
        "description": "Japanese Wikipedia",
    },
}

# Shuffle strategies
SHUFFLE_STRATEGIES = ["none", "sentence", "paragraph", "word"]


# ============== UTILITY FUNCTIONS ==============

def get_available_sources() -> Dict[str, str]:
    """Return dictionary of available sources with descriptions."""
    sources = {}
    for name, config in TEXT_SOURCES.items():
        sources[name] = config["description"]
    return sources


def get_available_languages() -> Dict[str, str]:
    """Return dictionary of available languages with descriptions."""
    languages = {}
    for lang, config in MULTILINGUAL_SOURCES.items():
        languages[lang] = config["description"]
    return languages


def estimate_entries_needed(max_length: int, chars_per_token: float = 4.0) -> int:
    """
    Estimate number of dataset entries needed to fill max_length tokens.

    Args:
        max_length: Target number of tokens
        chars_per_token: Estimated characters per token (varies by language)

    Returns:
        Estimated number of entries to load
    """
    # Assume average entry is ~500 characters
    avg_entry_chars = 500
    target_chars = max_length * chars_per_token
    estimated_entries = int(target_chars / avg_entry_chars * 2)  # 2x safety margin
    return max(10000, estimated_entries)


def _estimate_chars_per_token(language: str) -> float:
    if language in ["zh", "ja"]:
        return 1.5
    if language in ["ar"]:
        return 3.0
    return 4.0


def _extract_text(entry: Dict, text_field: str, fallback_fields: Optional[List[str]] = None) -> str:
    if text_field in entry and entry[text_field]:
        return entry[text_field]
    if fallback_fields:
        for field in fallback_fields:
            if field in entry and entry[field]:
                return entry[field]
    return ""


# ============== LOCAL WIKIPEDIA DATASETS ==============

def _get_local_wiki_path(language: str) -> Optional[str]:
    """
    Check for locally saved Wikipedia datasets (created via streaming + save_to_disk).
    Returns path if found, None otherwise.
    """
    # Try $WORK/code_for_bigbird first (Jean Zay)
    work_dir = os.environ.get("WORK", "")
    possible_paths = [
        os.path.join(work_dir, "code_for_bigbird", f"wikipedia_{language}_10k"),
        os.path.join(os.path.dirname(__file__), f"wikipedia_{language}_10k"),
        f"./wikipedia_{language}_10k",
    ]

    for path in possible_paths:
        if path and os.path.isdir(path):
            # Check if it's a valid datasets directory
            if os.path.exists(os.path.join(path, "dataset_info.json")) or \
               os.path.exists(os.path.join(path, "state.json")):
                return path
    return None


def _load_local_wikipedia(
    language: str,
    max_entries: int,
    verbose: bool = True
) -> Optional[List[str]]:
    """
    Load Wikipedia from locally saved dataset (created via streaming + save_to_disk).
    Returns list of texts if found, None if not available.
    """
    local_path = _get_local_wiki_path(language)
    if not local_path:
        return None

    if verbose:
        print(f"  Loading local Wikipedia dataset from: {local_path}")

    try:
        ds = load_from_disk(local_path)
        texts = []

        # Try common field names
        text_field = None
        for field in ["text", "content", "article"]:
            if field in ds.column_names:
                text_field = field
                break

        if text_field is None:
            warnings.warn(f"Could not find text field in local Wikipedia dataset. Columns: {ds.column_names}")
            return None

        for i in range(min(len(ds), max_entries)):
            text = ds[i].get(text_field, "")
            if text and text.strip():
                texts.append(text.strip())

        if verbose:
            print(f"  Loaded {len(texts)} entries from local Wikipedia ({language})")

        return texts

    except Exception as e:
        warnings.warn(f"Failed to load local Wikipedia ({language}): {e}")
        return None


# ============== CORE LOADING FUNCTIONS ==============

def _is_offline_mode() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes") or \
        os.environ.get("HF_DATASETS_OFFLINE", "").lower() in ("1", "true", "yes")


def _get_datasets_cache_dir() -> str:
    return os.environ.get(
        "HF_DATASETS_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets"),
    )


def _get_hub_cache_dir() -> str:
    """Get HuggingFace hub cache directory."""
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return os.path.join(hf_home, "hub")


def _load_from_hub_cache(
    dataset_name: str,
    text_field: str,
    max_entries: int,
    fallback_text_fields: Optional[List[str]] = None,
) -> List[str]:
    """
    Load dataset from HuggingFace hub cache (datasets--{name} format).
    Supports both .arrow and .parquet files.
    """
    hub_dir = _get_hub_cache_dir()
    dataset_dir = os.path.join(hub_dir, f"datasets--{dataset_name}")

    if not os.path.isdir(dataset_dir):
        return []

    # Find snapshots directory
    snapshots_dir = os.path.join(dataset_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return []

    # Get the latest snapshot
    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    if not snapshots:
        return []

    # Find data files in snapshots (arrow or parquet)
    data_files = []
    for snapshot in sorted(snapshots, reverse=True):  # Try newest first
        snapshot_path = os.path.join(snapshots_dir, snapshot)
        for root, dirs, files in os.walk(snapshot_path):
            for f in files:
                if f.endswith(".arrow") or f.endswith(".parquet"):
                    data_files.append(os.path.join(root, f))
        if data_files:
            break

    if not data_files:
        return []

    texts = []
    remaining = max_entries

    for data_path in sorted(data_files):
        if remaining <= 0:
            break
        try:
            if data_path.endswith(".parquet"):
                import pandas as pd
                df = pd.read_parquet(data_path)
                # Find the text column
                col = None
                for c in [text_field] + (fallback_text_fields or []):
                    if c in df.columns:
                        col = c
                        break
                if col is None:
                    continue
                for text in df[col].head(remaining):
                    if text and str(text).strip():
                        texts.append(str(text).strip())
                        if len(texts) >= max_entries:
                            break
            else:
                ds = Dataset.from_file(data_path)
                take = min(len(ds), remaining)
                for i in range(take):
                    text = _extract_text(ds[i], text_field, fallback_text_fields)
                    if text and text.strip():
                        texts.append(text.strip())
        except Exception as e:
            continue
        remaining = max_entries - len(texts)

    return texts


def _load_wikitext_from_cache(
    config: str,
    split: str,
    text_field: str,
    max_entries: int,
    fallback_text_fields: Optional[List[str]] = None,
) -> List[str]:
    cache_root = _get_datasets_cache_dir()
    base_dir = os.path.join(cache_root, "wikitext", config, "0.0.0")
    if not os.path.isdir(base_dir):
        return []

    candidate_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    arrow_files: List[str] = []
    for candidate in sorted(candidate_dirs):
        for name in os.listdir(candidate):
            if name.endswith(".arrow") and name.startswith(f"wikitext-{split}"):
                arrow_files.append(os.path.join(candidate, name))
        if arrow_files:
            break

    if not arrow_files:
        return []

    texts: List[str] = []
    remaining = max_entries
    for arrow_path in sorted(arrow_files):
        if remaining <= 0:
            break
        try:
            ds = Dataset.from_file(arrow_path)
        except Exception:
            continue
        take = min(len(ds), remaining)
        for i in range(take):
            text = _extract_text(ds[i], text_field, fallback_text_fields)
            if text and text.strip():
                texts.append(text.strip())
        remaining = max_entries - len(texts)

    return texts


def _load_from_hf_dataset(
    dataset_name: str,
    config: Optional[str],
    split: str,
    text_field: str,
    max_entries: int,
    streaming: bool = False,
    fallback_text_fields: Optional[List[str]] = None,
    fallback_to_wikitext: bool = True
) -> List[str]:
    """
    Load text entries from a HuggingFace dataset.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        config: Dataset configuration (e.g., "wikitext-103-raw-v1")
        split: Dataset split to use
        text_field: Field containing the text
        max_entries: Maximum number of entries to load
        streaming: Whether to use streaming mode

    Returns:
        List of text strings
    """
    try:
        if config:
            dataset = load_dataset(dataset_name, config, split=split, streaming=streaming)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    except Exception as e:
        if _is_offline_mode():
            # Try loading from hub cache first
            cached = _load_from_hub_cache(
                dataset_name=dataset_name,
                text_field=text_field,
                max_entries=max_entries,
                fallback_text_fields=fallback_text_fields,
            )
            if cached:
                warnings.warn(f"Offline mode: loaded {dataset_name} from hub cache.")
                return cached

            # Try wikitext-specific cache
            if dataset_name == "wikitext" and config:
                cached = _load_wikitext_from_cache(
                    config=config,
                    split=split,
                    text_field=text_field,
                    max_entries=max_entries,
                    fallback_text_fields=fallback_text_fields,
                )
                if cached:
                    warnings.warn("Offline mode: loaded cached wikitext Arrow files.")
                    return cached

            # Fallback to wikitext
            if fallback_to_wikitext:
                cached = _load_wikitext_from_cache(
                    config="wikitext-103-raw-v1",
                    split="train",
                    text_field="text",
                    max_entries=max_entries,
                    fallback_text_fields=None,
                )
                if cached:
                    warnings.warn("Offline mode: falling back to cached wikitext Arrow files.")
                    return cached
            raise
        if fallback_to_wikitext:
            warnings.warn(f"Failed to load {dataset_name}: {e}. Falling back to wikitext.")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            text_field = "text"
            streaming = False
            fallback_text_fields = None
        else:
            raise

    texts = []

    if streaming:
        # Streaming mode for large datasets
        for i, entry in enumerate(dataset):
            if i >= max_entries:
                break
            text = _extract_text(entry, text_field, fallback_text_fields)
            if text and text.strip():
                texts.append(text.strip())
    else:
        # Regular mode
        max_entries = min(max_entries, len(dataset))
        for i in range(max_entries):
            text = _extract_text(dataset[i], text_field, fallback_text_fields)
            if text and text.strip():
                texts.append(text.strip())

    return texts


def load_single_source(
    max_length: int,
    source: str = "wiki",
    language: str = "en",
    verbose: bool = True
) -> str:
    """
    Load text from a single source.

    Args:
        max_length: Target number of tokens
        source: Source name (wiki, news, books, code, reddit, scientific)
        language: Language code for multilingual sources
        verbose: Print loading progress

    Returns:
        Concatenated text string
    """
    # Determine dataset configuration
    if source == "scientific":
        return load_scientific_source(max_length=max_length, verbose=verbose)

    # Estimate entries needed
    chars_per_token = _estimate_chars_per_token(language)
    estimated_entries = estimate_entries_needed(max_length, chars_per_token)

    if verbose:
        print(f"Loading {source} ({language}): ~{estimated_entries} entries for {max_length} tokens...")

    # For non-English Wikipedia, try local datasets first
    if source == "wiki" and language != "en":
        local_texts = _load_local_wikipedia(language, estimated_entries, verbose)
        if local_texts:
            if verbose:
                print(f"  Loaded {len(local_texts)} non-empty entries from local dataset")
            return " ".join(local_texts)

        # Fall back to HuggingFace
        if language not in MULTILINGUAL_SOURCES:
            raise ValueError(f"Language '{language}' not available. "
                           f"Available: {list(MULTILINGUAL_SOURCES.keys())}")
        config = MULTILINGUAL_SOURCES[language]
    else:
        if source not in TEXT_SOURCES:
            raise ValueError(f"Source '{source}' not available. "
                           f"Available: {list(TEXT_SOURCES.keys())}")
        config = TEXT_SOURCES[source]

    texts = _load_from_hf_dataset(
        dataset_name=config["dataset"],
        config=config.get("config"),
        split=config["split"],
        text_field=config["text_field"],
        max_entries=estimated_entries,
        streaming=config.get("streaming", False)
    )

    if verbose:
        print(f"  Loaded {len(texts)} non-empty entries")

    return " ".join(texts)


def load_scientific_source(max_length: int, verbose: bool = True) -> str:
    """
    Load scientific text with a robust fallback order.
    """
    chars_per_token = _estimate_chars_per_token("en")
    estimated_entries = estimate_entries_needed(max_length, chars_per_token)

    if verbose:
        print(f"Loading scientific (en): ~{estimated_entries} entries for {max_length} tokens...")

    # Try scientific_papers first (legacy), then arxiv (more robust).
    try:
        texts = _load_from_hf_dataset(
            dataset_name="scientific_papers",
            config="arxiv",
            split="train",
            text_field="article",
            max_entries=estimated_entries,
            streaming=False,
            fallback_text_fields=["abstract", "text"],
            fallback_to_wikitext=False
        )
    except Exception as e1:
        warnings.warn(f"Failed to load scientific_papers/arxiv: {e1}. Trying arxiv dataset.")
        try:
            texts = _load_from_hf_dataset(
                dataset_name="arxiv",
                config=None,
                split="train",
                text_field="abstract",
                max_entries=estimated_entries,
                streaming=True,
                fallback_text_fields=["article", "text"],
                fallback_to_wikitext=False
            )
        except Exception as e2:
            warnings.warn(f"Failed to load arxiv dataset: {e2}. Falling back to wikitext.")
            texts = _load_from_hf_dataset(
                dataset_name="wikitext",
                config="wikitext-103-raw-v1",
                split="train",
                text_field="text",
                max_entries=estimated_entries,
                streaming=False
            )

    if verbose:
        print(f"  Loaded {len(texts)} non-empty entries")
    return " ".join(texts)


def load_mr_niah(
    max_length: int,
    language: str = "en",
    verbose: bool = True
) -> str:
    """
    Load haystack text from MR-NIAH dataset.

    The MR-NIAH dataset contains pre-generated haystack texts at various token lengths.
    This function selects the appropriate token-length file based on max_length.

    Args:
        max_length: Target number of tokens. Will select the smallest available
                   token length that is >= max_length.
        language: Language code - "en" for English, "zh" for Chinese
        verbose: Print loading progress

    Returns:
        Concatenated haystack text string

    Available token lengths: 2048, 10240, 20480, 51200, 102400, 204800, 512000, 1024000
    """
    # Select appropriate token length
    selected_length = None
    for length in MR_NIAH_TOKEN_LENGTHS:
        if length >= max_length:
            selected_length = length
            break

    if selected_length is None:
        # Use largest available if max_length exceeds all options
        selected_length = MR_NIAH_TOKEN_LENGTHS[-1]
        if verbose:
            print(f"  Warning: max_length {max_length} exceeds largest MR-NIAH length. "
                  f"Using {selected_length} tokens.")

    if verbose:
        print(f"Loading MR-NIAH haystack ({language}): selected {selected_length} token file for {max_length} target...")

    # Map language to folder name
    lang_folder = "english" if language == "en" else "chinese" if language == "zh" else "english"

    try:
        # Try loading specific file directly using data_files parameter
        # This avoids loading the entire dataset which may have corrupted files
        data_file = f"{lang_folder}/{selected_length}_tokens.jsonl"

        if verbose:
            print(f"  Attempting to load: {data_file}")

        dataset = None
        load_errors = []

        # Method 1: Try loading specific file directly
        try:
            dataset = load_dataset(
                "MiniMaxAI/MR-NIAH",
                data_files={"train": data_file},
                split="train"
            )
        except Exception as e1:
            load_errors.append(f"Direct file load: {e1}")

        # Method 2: Try with streaming
        if dataset is None:
            try:
                if verbose:
                    print(f"  Trying streaming mode...")
                stream_dataset = load_dataset(
                    "MiniMaxAI/MR-NIAH",
                    data_files={"train": data_file},
                    split="train",
                    streaming=True
                )
                # Convert streaming dataset to list (take first few entries)
                dataset = list(stream_dataset.take(10))
            except Exception as e2:
                load_errors.append(f"Streaming load: {e2}")

        # Method 3: Try loading all English files (avoiding corrupted Chinese files)
        if dataset is None and language == "en":
            try:
                if verbose:
                    print(f"  Trying to load all English files...")
                dataset = load_dataset(
                    "MiniMaxAI/MR-NIAH",
                    data_files={"train": "english/*.jsonl"},
                    split="train"
                )
                # Filter by token length
                dataset = [
                    entry for entry in dataset
                    if entry.get('token_len') == selected_length
                ][:10]
            except Exception as e3:
                load_errors.append(f"All English files: {e3}")

        # Method 4: Lenient JSONL parsing via hf_hub_download
        if dataset is None and hf_hub_download is not None:
            try:
                if verbose:
                    print(f"  Trying lenient JSONL parse...")
                local_path = hf_hub_download(
                    repo_id="MiniMaxAI/MR-NIAH",
                    filename=data_file,
                )
                entries = []
                with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            continue
                        if len(entries) >= 10:
                            break
                dataset = entries
            except Exception as e4:
                load_errors.append(f"Lenient JSONL parse: {e4}")

        if dataset is None:
            error_msg = "; ".join(load_errors)
            raise ValueError(f"All loading methods failed: {error_msg}")

        # Extract text from the dataset
        texts = []

        # Handle both list and dataset objects
        entries = dataset if isinstance(dataset, list) else list(dataset.take(10) if hasattr(dataset, 'take') else dataset[:10])

        for entry in entries:
            # Try different field names that might contain the haystack text
            text_content = None

            # Check for 'haystack' field (common in NIAH datasets)
            if 'haystack' in entry and entry['haystack']:
                text_content = entry['haystack']

            # Check for 'context' field
            elif 'context' in entry and entry['context']:
                text_content = entry['context']

            # Check for 'text' field
            elif 'text' in entry and entry['text']:
                text_content = entry['text']

            # Check for 'messages' field (dialogue format)
            elif 'messages' in entry:
                messages = entry['messages']
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict):
                            content = msg.get('content', '')
                            if content and isinstance(content, str) and len(content) > 100:
                                texts.append(content.strip())
                elif isinstance(messages, str):
                    text_content = messages

            if text_content and isinstance(text_content, str):
                texts.append(text_content.strip())

        if not texts:
            raise ValueError(f"Could not extract text from MR-NIAH dataset for {selected_length} tokens")

        if verbose:
            print(f"  Loaded {len(texts)} text segments from MR-NIAH")
            total_chars = sum(len(t) for t in texts)
            print(f"  Total characters: {total_chars}")

        return " ".join(texts)

    except Exception as e:
        warnings.warn(f"Failed to load MR-NIAH dataset: {e}. Falling back to wikitext.")
        return load_single_source(max_length, source="wiki", language="en", verbose=verbose)


def apply_repetition(
    text: str,
    max_length: int,
    language: str,
    repeat_config: Dict,
    verbose: bool = True
) -> str:
    """
    Repeat a text block to amplify covariance / low-entropy structure.

    Modes:
        - "block": Repeat a small block (default 20k chars)
        - "tile": Tile the ENTIRE text to fill max_length (useful when source is small)
        - "none": No repetition
    """
    if not text:
        return text

    mode = repeat_config.get("mode", "block")
    if mode == "none":
        return text

    separator = repeat_config.get("separator", "\n")
    seed = repeat_config.get("seed", None)
    repeats = repeat_config.get("repeats", None)

    if seed is not None:
        random.seed(seed)

    # Target characters (with extra margin for tokenizer compression)
    target_chars = int(max_length * _estimate_chars_per_token(language) * 1.5)

    if mode == "tile":
        # Tile the ENTIRE text to fill target length
        if len(text) == 0:
            return text

        if repeats is None:
            repeats = max(1, math.ceil(target_chars / len(text)))

        if verbose:
            print(f"Applying repetition: mode=tile, text_chars={len(text)}, repeats={repeats}, target={target_chars}")

        tiled = (text + separator) * repeats
        return tiled.strip()

    else:  # mode == "block"
        block_chars = int(repeat_config.get("block_chars", 20000))
        block_strategy = repeat_config.get("block_strategy", "prefix")

        if block_chars <= 0:
            return text

        if block_strategy == "random_span" and len(text) > block_chars:
            start = random.randint(0, max(0, len(text) - block_chars))
            block = text[start:start + block_chars]
        else:
            block = text[:block_chars]

        if not block:
            return text

        if repeats is None:
            repeats = max(1, math.ceil(target_chars / max(1, len(block))))

        if verbose:
            print(f"Applying repetition: mode=block, block_chars={len(block)}, repeats={repeats}, target={target_chars}")

        repeated = (block + separator) * repeats
        return repeated.strip()


def load_mixed_languages(
    max_length: int,
    language_config: Dict[str, float],
    source: str = "wiki",
    mix_strategy: str = "concat",
    verbose: bool = True
) -> str:
    """
    Load Wikipedia text from multiple languages with controlled proportions.

    Args:
        max_length: Target number of tokens
        language_config: Dictionary mapping language codes to proportions
            Example: {"en": 0.5, "de": 0.3, "fr": 0.2}
        source: Base source to use (currently only "wiki" supports multilingual)
        mix_strategy: How to combine languages
            - "concat": Concatenate blocks from each language
            - "interleave": Interleave sentences from languages
            - "random": Randomly sample from languages
        verbose: Print loading progress

    Returns:
        Mixed language text string
    """
    # Normalize proportions
    total = sum(language_config.values())
    language_config = {k: v / total for k, v in language_config.items()}

    if verbose:
        print(f"Loading mixed languages: {language_config}")
        print(f"Mix strategy: {mix_strategy}")

    language_texts = {}

    for lang, proportion in language_config.items():
        if lang not in MULTILINGUAL_SOURCES:
            warnings.warn(f"Language '{lang}' not available, skipping. "
                         f"Available: {list(MULTILINGUAL_SOURCES.keys())}")
            continue

        lang_max_length = int(max_length * proportion * 1.2)  # 20% margin
        language_texts[lang] = load_single_source(
            max_length=lang_max_length,
            source=source,
            language=lang,
            verbose=verbose
        )

    if not language_texts:
        raise ValueError("No valid languages found in language_config")

    if mix_strategy == "concat":
        # Simple concatenation
        return " ".join(language_texts.values())

    elif mix_strategy == "interleave":
        # Interleave sentences from different languages
        all_sentences = []
        for lang, text in language_texts.items():
            sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
            all_sentences.extend(sentences)

        random.shuffle(all_sentences)
        return " ".join(all_sentences)

    elif mix_strategy == "random":
        # Random sampling at word level
        all_words = []
        for lang, text in language_texts.items():
            words = text.split()
            # Sample proportionally
            n_words = int(len(words) * language_config[lang])
            sampled = random.sample(words, min(n_words, len(words)))
            all_words.extend(sampled)

        random.shuffle(all_words)
        return " ".join(all_words)

    else:
        raise ValueError(f"Unknown mix_strategy: {mix_strategy}")


def load_mixed_sources(
    max_length: int,
    mix_config: Dict[str, float],
    mix_strategy: str = "concat",
    verbose: bool = True
) -> str:
    """
    Load text from multiple sources with controlled mixing.

    Args:
        max_length: Target number of tokens
        mix_config: Dictionary mapping source names to proportions (should sum to 1.0)
            Example: {"wiki": 0.5, "news": 0.3, "reddit": 0.2}
        mix_strategy: How to combine sources
            - "concat": Concatenate blocks from each source
            - "interleave": Interleave sentences from sources
            - "random": Randomly sample from sources
        verbose: Print loading progress

    Returns:
        Mixed text string
    """
    # Normalize proportions
    total = sum(mix_config.values())
    mix_config = {k: v / total for k, v in mix_config.items()}

    if verbose:
        print(f"Loading mixed sources: {mix_config}")
        print(f"Mix strategy: {mix_strategy}")

    source_texts = {}

    for source, proportion in mix_config.items():
        source_max_length = int(max_length * proportion * 1.2)  # 20% margin
        source_texts[source] = load_single_source(
            max_length=source_max_length,
            source=source,
            verbose=verbose
        )

    if mix_strategy == "concat":
        # Simple concatenation
        return " ".join(source_texts.values())

    elif mix_strategy == "interleave":
        # Interleave sentences
        all_sentences = []
        for source, text in source_texts.items():
            sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
            # Tag sentences with source (for potential analysis)
            all_sentences.extend(sentences)

        random.shuffle(all_sentences)
        return " ".join(all_sentences)

    elif mix_strategy == "random":
        # Random sampling at word level (destroys all structure)
        all_words = []
        for source, text in source_texts.items():
            words = text.split()
            # Sample proportionally
            n_words = int(len(words) * mix_config[source])
            sampled = random.sample(words, min(n_words, len(words)))
            all_words.extend(sampled)

        random.shuffle(all_words)
        return " ".join(all_words)

    else:
        raise ValueError(f"Unknown mix_strategy: {mix_strategy}")


def shuffle_text(
    text: str,
    shuffle_level: str = "none",
    seed: Optional[int] = None
) -> str:
    """
    Shuffle text at different granularities.

    Args:
        text: Input text
        shuffle_level: Level of shuffling
            - "none": No shuffling
            - "sentence": Shuffle sentences
            - "paragraph": Shuffle paragraphs
            - "word": Shuffle all words (destroys grammar)
        seed: Random seed for reproducibility

    Returns:
        Shuffled text
    """
    if seed is not None:
        random.seed(seed)

    if shuffle_level == "none":
        return text

    elif shuffle_level == "sentence":
        # Split on sentence boundaries
        sentences = []
        for sent in text.replace("!", ".").replace("?", ".").split("."):
            sent = sent.strip()
            if sent:
                sentences.append(sent + ".")
        random.shuffle(sentences)
        return " ".join(sentences)

    elif shuffle_level == "paragraph":
        # Split on double newlines or long whitespace
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) < 2:
            # Fallback: split on single newlines
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        random.shuffle(paragraphs)
        return "\n\n".join(paragraphs)

    elif shuffle_level == "word":
        # Shuffle all words (destroys grammar completely)
        words = text.split()
        random.shuffle(words)
        return " ".join(words)

    else:
        raise ValueError(f"Unknown shuffle_level: {shuffle_level}. "
                        f"Available: {SHUFFLE_STRATEGIES}")


# ============== SYNTHETIC EMBEDDINGS ==============

def generate_synthetic_embeddings(
    n_tokens: int,
    d: int = 768,
    cov_type: str = "identity",
    cov_params: Optional[Dict] = None,
    seed: int = 42
) -> torch.Tensor:
    """
    Generate synthetic embeddings with controlled covariance structure.

    This is useful for theoretical validation where you want to test
    with known covariance matrices.

    Args:
        n_tokens: Number of tokens to generate
        d: Embedding dimension
        cov_type: Type of covariance structure
            - "identity": Σ = I
            - "scaled_identity": Σ = σ²I
            - "diagonal": Σ = diag(σ₁², ..., σ_d²)
            - "low_rank": Σ = UU^T + σ²I (rank k)
            - "power_law": eigenvalues decay as 1/i^α
            - "block_diagonal": Block structure
            - "toeplitz": Toeplitz structure (AR-like correlations)
            - "custom": User-provided covariance matrix
        cov_params: Parameters for covariance construction
            - scaled_identity: {"scale": float}
            - diagonal: {"scales": array}
            - low_rank: {"rank": int, "noise": float}
            - power_law: {"alpha": float, "scale": float}
            - block_diagonal: {"block_size": int, "within_corr": float}
            - toeplitz: {"rho": float} (correlation decay)
            - custom: {"cov": tensor}
        seed: Random seed

    Returns:
        Tensor of shape [1, n_tokens, d] with specified covariance
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cov_params = cov_params or {}

    if cov_type == "identity":
        Sigma = torch.eye(d)

    elif cov_type == "scaled_identity":
        scale = cov_params.get("scale", 1.0)
        Sigma = scale * torch.eye(d)

    elif cov_type == "diagonal":
        scales = cov_params.get("scales", None)
        if scales is None:
            scales = torch.rand(d) + 0.1
        Sigma = torch.diag(torch.tensor(scales, dtype=torch.float32))

    elif cov_type == "low_rank":
        rank = cov_params.get("rank", 50)
        noise = cov_params.get("noise", 0.1)
        U = torch.randn(d, rank) / np.sqrt(rank)
        Sigma = U @ U.T + noise * torch.eye(d)

    elif cov_type == "power_law":
        alpha = cov_params.get("alpha", 1.0)
        scale = cov_params.get("scale", 1.0)
        eigenvalues = scale * torch.tensor([1.0 / (i ** alpha) for i in range(1, d + 1)])
        # Random orthogonal basis
        Q, _ = torch.linalg.qr(torch.randn(d, d))
        Sigma = Q @ torch.diag(eigenvalues) @ Q.T

    elif cov_type == "block_diagonal":
        block_size = cov_params.get("block_size", 64)
        within_corr = cov_params.get("within_corr", 0.5)
        n_blocks = d // block_size

        Sigma = torch.zeros(d, d)
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            block = within_corr * torch.ones(block_size, block_size)
            block.fill_diagonal_(1.0)
            Sigma[start:end, start:end] = block

        # Handle remainder
        if d % block_size != 0:
            Sigma[n_blocks * block_size:, n_blocks * block_size:] = torch.eye(d % block_size)

    elif cov_type == "toeplitz":
        rho = cov_params.get("rho", 0.9)
        # Toeplitz: Σ[i,j] = rho^|i-j|
        indices = torch.arange(d)
        Sigma = rho ** torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)).float()

    elif cov_type == "custom":
        Sigma = cov_params["cov"]
        if isinstance(Sigma, np.ndarray):
            Sigma = torch.tensor(Sigma, dtype=torch.float32)

    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")

    # Ensure positive definiteness
    Sigma = Sigma + 1e-6 * torch.eye(d)

    # Cholesky decomposition for sampling
    L = torch.linalg.cholesky(Sigma)

    # Generate samples: X = Z @ L^T where Z ~ N(0, I)
    Z = torch.randn(1, n_tokens, d)
    X = Z @ L.T

    return X, Sigma


# ============== MAIN INTERFACE ==============

def load_text(
    max_length: int,
    source: str = "wiki",
    language: str = "en",
    mix_config: Optional[Dict[str, float]] = None,
    mix_strategy: str = "concat",
    shuffle: str = "none",
    shuffle_seed: Optional[int] = None,
    language_mix_config: Optional[Dict[str, float]] = None,
    repeat_config: Optional[Dict] = None,
    verbose: bool = True
) -> str:
    """
    Main text loading interface.

    Args:
        max_length: Target number of tokens
        source: Text source ("wiki", "news", "books", "code", "reddit",
                "scientific", "openwebtext", "mr_niah", or "mixed")
        language: Language code for wiki/mr_niah source ("en", "de", "fr", "es",
                  "zh", "ar", "ru", "ja") or "mixed" for mixed languages
        mix_config: For source="mixed", dictionary of source->proportion
        mix_strategy: For mixed sources/languages: "concat", "interleave", "random"
        shuffle: Shuffle level: "none", "sentence", "paragraph", "word"
        shuffle_seed: Random seed for shuffling
        language_mix_config: For language="mixed", dictionary of lang->proportion
            Example: {"en": 0.5, "de": 0.3, "fr": 0.2}
        verbose: Print loading progress

    Returns:
        Text string ready for tokenization

    Examples:
        # Simple Wikipedia
        text = load_text(4096, source="wiki")

        # German Wikipedia
        text = load_text(4096, source="wiki", language="de")

        # Mixed sources
        text = load_text(4096, source="mixed",
                        mix_config={"wiki": 0.5, "news": 0.5})

        # Mixed languages
        text = load_text(4096, source="wiki", language="mixed",
                        language_mix_config={"en": 0.5, "de": 0.5})

        # MR-NIAH benchmark
        text = load_text(4096, source="mr_niah")

        # Shuffled sentences
        text = load_text(4096, source="wiki", shuffle="sentence")
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Text Loading Configuration:")
        print(f"  Source: {source}")
        print(f"  Language: {language}")
        print(f"  Max length: {max_length}")
        print(f"  Shuffle: {shuffle}")
        if source == "mixed":
            print(f"  Mix config: {mix_config}")
            print(f"  Mix strategy: {mix_strategy}")
        if language == "mixed":
            print(f"  Language mix config: {language_mix_config}")
            print(f"  Language mix strategy: {mix_strategy}")
        if repeat_config:
            print(f"  Repeat config: {repeat_config}")
        print(f"{'='*60}\n")

    # Load text based on source type
    if source == "mr_niah":
        # MR-NIAH benchmark haystack
        text = load_mr_niah(
            max_length=max_length,
            language=language if language != "mixed" else "en",
            verbose=verbose
        )
    elif source == "mixed":
        if mix_config is None:
            raise ValueError("mix_config required for source='mixed'")
        text = load_mixed_sources(
            max_length=max_length,
            mix_config=mix_config,
            mix_strategy=mix_strategy,
            verbose=verbose
        )
    elif language == "mixed":
        # Mixed languages
        if language_mix_config is None:
            raise ValueError("language_mix_config required for language='mixed'")
        text = load_mixed_languages(
            max_length=max_length,
            language_config=language_mix_config,
            source=source,
            mix_strategy=mix_strategy,
            verbose=verbose
        )
    else:
        text = load_single_source(
            max_length=max_length,
            source=source,
            language=language,
            verbose=verbose
        )

    if repeat_config:
        text = apply_repetition(
            text=text,
            max_length=max_length,
            language=language,
            repeat_config=repeat_config,
            verbose=verbose
        )

    # Safety fallback: tile text if too short for target tokens
    # Estimate: ~4 chars per token, need max_length tokens
    min_chars_needed = int(max_length * 3.5)  # Conservative estimate
    if len(text) < min_chars_needed:
        if verbose:
            print(f"WARNING: Text too short ({len(text)} chars < {min_chars_needed} needed)")
            print(f"Auto-tiling to reach target length...")
        repeats = math.ceil(min_chars_needed / max(1, len(text)))
        text = (text + "\n") * repeats
        if verbose:
            print(f"Tiled {repeats}x -> {len(text)} chars")

    # Apply shuffling if requested
    if shuffle != "none":
        if verbose:
            print(f"Applying {shuffle}-level shuffle...")
        text = shuffle_text(text, shuffle_level=shuffle, seed=shuffle_seed)

    if verbose:
        print(f"Final text length: {len(text)} characters")

    return text


# ============== CONVENIENCE FUNCTIONS FOR EXPERIMENTS ==============

def get_experiment_configs() -> Dict[str, Dict]:
    """
    Return predefined experiment configurations for systematic testing.

    Returns:
        Dictionary of experiment name -> configuration
    """
    configs = {
        # Baseline
        "baseline_wiki": {
            "source": "wiki",
            "language": "en",
            "shuffle": "none",
            "description": "Standard Wikipedia (English)"
        },

        # Domain variation
        "domain_news": {
            "source": "news",
            "language": "en",
            "shuffle": "none",
            "description": "News articles"
        },
        "domain_scientific": {
            "source": "scientific",
            "language": "en",
            "shuffle": "none",
            "description": "ArXiv papers"
        },
        "domain_code": {
            "source": "code",
            "language": "en",
            "shuffle": "none",
            "description": "GitHub code"
        },

        # Language variation
        "lang_german": {
            "source": "wiki",
            "language": "de",
            "shuffle": "none",
            "description": "German Wikipedia"
        },
        "lang_french": {
            "source": "wiki",
            "language": "fr",
            "shuffle": "none",
            "description": "French Wikipedia"
        },
        "lang_chinese": {
            "source": "wiki",
            "language": "zh",
            "shuffle": "none",
            "description": "Chinese Wikipedia"
        },
        "lang_arabic": {
            "source": "wiki",
            "language": "ar",
            "shuffle": "none",
            "description": "Arabic Wikipedia"
        },

        # Mixed sources
        "mixed_wiki_news": {
            "source": "mixed",
            "mix_config": {"wiki": 0.5, "news": 0.5},
            "mix_strategy": "concat",
            "shuffle": "none",
            "description": "50% Wiki + 50% News"
        },
        "mixed_formal_informal": {
            "source": "mixed",
            "mix_config": {"scientific": 0.5, "reddit": 0.5},
            "mix_strategy": "concat",
            "shuffle": "none",
            "description": "50% Scientific + 50% Reddit"
        },
        "mixed_interleaved": {
            "source": "mixed",
            "mix_config": {"wiki": 0.5, "news": 0.5},
            "mix_strategy": "interleave",
            "shuffle": "none",
            "description": "Wiki+News interleaved sentences"
        },
        "mixed_imbalanced_wiki_news_95_5": {
            "source": "mixed",
            "mix_config": {"wiki": 0.95, "news": 0.05},
            "mix_strategy": "concat",
            "shuffle": "none",
            "description": "Imbalanced: 95% Wiki + 5% News"
        },
        "mixed_imbalanced_code_news_90_10": {
            "source": "mixed",
            "mix_config": {"code": 0.9, "news": 0.1},
            "mix_strategy": "concat",
            "shuffle": "none",
            "description": "Imbalanced: 90% Code + 10% News"
        },

        # Shuffle experiments
        "shuffle_sentence": {
            "source": "wiki",
            "language": "en",
            "shuffle": "sentence",
            "description": "Wiki with shuffled sentences"
        },
        "shuffle_paragraph": {
            "source": "wiki",
            "language": "en",
            "shuffle": "paragraph",
            "description": "Wiki with shuffled paragraphs"
        },
        "shuffle_word": {
            "source": "wiki",
            "language": "en",
            "shuffle": "word",
            "description": "Wiki with shuffled words (no grammar)"
        },

        # Repetition experiments
        "repeat_news_block": {
            "source": "news",
            "language": "en",
            "shuffle": "none",
            "repeat_config": {
                "mode": "block",
                "block_chars": 20000,
                "block_strategy": "random_span",
                "seed": 42
            },
            "description": "News with repeated block (high correlation)"
        },
    }

    return configs


def load_for_experiment(
    max_length: int,
    config_name: str,
    verbose: bool = True
) -> str:
    """
    Load text using a predefined experiment configuration.

    Args:
        max_length: Target number of tokens
        config_name: Name of experiment configuration
        verbose: Print loading progress

    Returns:
        Text string
    """
    configs = get_experiment_configs()

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. "
                        f"Available: {list(configs.keys())}")

    config = configs[config_name]

    if verbose:
        print(f"Loading experiment config: {config_name}")
        print(f"  Description: {config['description']}")

    return load_text(
        max_length=max_length,
        source=config.get("source", "wiki"),
        language=config.get("language", "en"),
        mix_config=config.get("mix_config"),
        mix_strategy=config.get("mix_strategy", "concat"),
        shuffle=config.get("shuffle", "none"),
        repeat_config=config.get("repeat_config"),
        verbose=verbose
    )


# ============== CLI FOR TESTING ==============

def parse_language_mix(mix_str: str) -> Optional[Dict[str, float]]:
    """Parse language mix string like 'en:0.5,de:0.5' to dict."""
    if not mix_str:
        return None
    mix_config = {}
    for part in mix_str.split(','):
        lang, prop = part.strip().split(':')
        mix_config[lang.strip()] = float(prop.strip())
    return mix_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Text Loaders Testing")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--source", type=str, default="wiki",
                       choices=list(TEXT_SOURCES.keys()) + ["mixed"])
    parser.add_argument("--language", type=str, default="en",
                       choices=list(MULTILINGUAL_SOURCES.keys()) + ["mixed"])
    parser.add_argument("--shuffle", type=str, default="none",
                       choices=SHUFFLE_STRATEGIES)
    parser.add_argument("--language_mix", type=str, default=None,
                       help="Language mix config as 'lang1:prop1,lang2:prop2' (e.g., 'en:0.5,de:0.5')")
    parser.add_argument("--config", type=str, default=None,
                       help="Use predefined experiment config")
    parser.add_argument("--list_configs", action="store_true",
                       help="List available experiment configs")
    parser.add_argument("--list_sources", action="store_true",
                       help="List available text sources")

    args = parser.parse_args()

    if args.list_configs:
        print("\nAvailable experiment configurations:")
        print("=" * 60)
        for name, config in get_experiment_configs().items():
            print(f"  {name}: {config['description']}")
        print()

    elif args.list_sources:
        print("\nAvailable text sources:")
        print("=" * 60)
        for name, desc in get_available_sources().items():
            print(f"  {name}: {desc}")
        print("\nAvailable languages:")
        print("=" * 60)
        for lang, desc in get_available_languages().items():
            print(f"  {lang}: {desc}")
        print("\nMR-NIAH available token lengths:")
        print("=" * 60)
        print(f"  {MR_NIAH_TOKEN_LENGTHS}")
        print()

    elif args.config:
        text = load_for_experiment(args.max_length, args.config)
        print(f"\nLoaded {len(text)} characters")
        print(f"First 500 chars:\n{text[:500]}...")

    else:
        # Parse language mix config if provided
        language_mix_config = parse_language_mix(args.language_mix)

        # Auto-set language to "mixed" if language_mix provided
        language = args.language
        if language_mix_config and language != "mixed":
            print(f"Note: --language_mix provided, setting language to 'mixed'")
            language = "mixed"

        text = load_text(
            max_length=args.max_length,
            source=args.source,
            language=language,
            shuffle=args.shuffle,
            language_mix_config=language_mix_config
        )
        print(f"\nLoaded {len(text)} characters")
        print(f"First 500 chars:\n{text[:500]}...")
