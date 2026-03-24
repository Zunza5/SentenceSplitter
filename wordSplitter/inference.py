"""
Inference pipeline for the Word Splitter.

Approach: The input spaceless text is expanded to have spaces between every
character. The MLP predicts which spaces to REMOVE.
"""

from pathlib import Path
import itertools
import math
from dataclasses import dataclass


import torch

try:
    from typing import Any
except ImportError:
    Any = object

import nltk

# Initialize Italian dictionary using WordNet
_valid_words = set()
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import wordnet
    _valid_words = set(wordnet.all_lemma_names(lang='ita'))
except Exception as e:
    print(f"Warning: Could not initialize NLTK Italian dictionary: {e}")

from wordSplitter.model import SpacePredictorMLP
from wordSplitter.embeddings import (
    load_language_model,
    extract_token_embeddings,
    expand_to_char_embeddings,
    compute_perplexity,
    compute_perplexity_batch,
    get_device,
)
from wordSplitter.data import build_char_to_token_map

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def load_mlp(checkpoint_path: Path | None = None, device: torch.device | None = None):
    """Load the trained MLP from a checkpoint."""
    if device is None:
        device = get_device()
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "best_mlp.pt"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    mlp = SpacePredictorMLP(
        hidden_dim=ckpt["hidden_dim"],
        dropout=ckpt.get("dropout", 0.2),
    )
    mlp.load_state_dict(ckpt["model_state_dict"])
    mlp.to(device)
    mlp.eval()
    print(f"Loaded MLP checkpoint (epoch {ckpt['epoch']}, F1={ckpt['f1']:.4f})")
    return mlp


def mlp_predict(
    text: str,
    mlp: SpacePredictorMLP,
    llm_model: Any,
    tokenizer: Any,
    device: torch.device,
    backend: str = "transformers"
) -> list[float]:
    """
    Run MLP inference on a spaced-out text string.

    The text is first expanded to "c i a o ..." format, then tokenized
    and fed through LLM + MLP.

    Returns:
        List of P(remove space) probabilities, one per character.
    """
    # Expand to spaced format
    spaced = " ".join(list(text.replace(" ", "")))

    # Tokenize and get char→token map
    input_ids, char_to_token = build_char_to_token_map(spaced, tokenizer)

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids_t)
    char_to_token_t = torch.tensor([char_to_token], dtype=torch.long, device=device)

    # Extract token embeddings from LLM
    tok_emb = extract_token_embeddings(llm_model, input_ids_t, attention_mask, backend=backend)

    # Expand to character level
    char_emb = expand_to_char_embeddings(tok_emb, char_to_token_t)

    # MLP prediction
    probs, _ = mlp(char_emb)  # (1, num_chars), aux_loss ignored at inference
    return probs.squeeze(0).cpu().tolist()


