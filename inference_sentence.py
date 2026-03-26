"""
Inference pipeline for Sentence Splitter.
"""

from pathlib import Path
from typing import Any

import torch
from model import SpacePredictorMLP
from sentence_embeddings import extract_token_embeddings, load_language_model, get_device

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BEST_SENTENCE_CKPT = CHECKPOINT_DIR / "best_sentence_mlp.pt"


def load_sentence_mlp(checkpoint_path: Path | None = None, device: torch.device | None = None):
    """Load the trained sentence MLP."""
    if checkpoint_path is None:
        checkpoint_path = BEST_SENTENCE_CKPT
    if device is None:
        device = get_device()

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    hidden_dim = checkpoint.get("hidden_dim", 2560)
    dropout = checkpoint.get("dropout", 0.2)
    d_model = checkpoint.get("d_model", checkpoint.get("cnn_dim", 256))
    num_experts = checkpoint.get("num_experts", 8)
    top_k = checkpoint.get("top_k", min(2, num_experts))

    mlp = SpacePredictorMLP(
        hidden_dim=hidden_dim,
        d_model=d_model,
        dropout=dropout,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)
    mlp.load_state_dict(state_dict)
    mlp.eval()
    return mlp


def _token_boundary_probs(
    chunk_text: str,
    mlp: SpacePredictorMLP,
    llm_model: Any,
    tokenizer: Any,
    device: torch.device,
    backend: str,
) -> list[tuple[int, float]]:
    """Return pairs of (chunk_char_end, boundary_prob) at token boundaries."""
    encoding = tokenizer(
        chunk_text,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offsets = encoding["offset_mapping"].squeeze(0).tolist()

    tok_emb = extract_token_embeddings(llm_model, input_ids, attention_mask, backend=backend)
    outputs = mlp(tok_emb, mask=attention_mask.bool())
    probs = outputs[0] if isinstance(outputs, tuple) else outputs
    probs = probs.squeeze(0).detach().cpu().tolist()

    boundaries: list[tuple[int, float]] = []
    for tok_idx, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            continue
        if end <= start:
            continue
        if tok_idx >= len(probs):
            break
        boundaries.append((end, float(probs[tok_idx])))
    return boundaries


def split_into_sentences(
    text: str,
    mlp: SpacePredictorMLP,
    llm_model: any,
    tokenizer: any,
    device: torch.device,
    backend: str = "transformers",
    threshold: float = 0.5,
    max_chars: int = 2048,
    stride_chars: int = 1024,
) -> list[str]:
    """
    Split text into sentences using token-level boundary probabilities.
    """
    if not text:
        return []

    boundary_scores: dict[int, list[float]] = {}

    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text = text[start:end]
        chunk_bounds = _token_boundary_probs(
            chunk_text=chunk_text,
            mlp=mlp,
            llm_model=llm_model,
            tokenizer=tokenizer,
            device=device,
            backend=backend,
        )

        for local_end, prob in chunk_bounds:
            global_end = start + local_end
            if 0 < global_end <= len(text):
                boundary_scores.setdefault(global_end, []).append(prob)

        if end == len(text):
            break

        start += stride_chars

    split_points = []
    for point, vals in boundary_scores.items():
        avg_prob = sum(vals) / max(1, len(vals))
        if avg_prob > threshold and point < len(text):
            split_points.append(point)
    split_points = sorted(set(split_points))

    sentences = []
    prev = 0
    for point in split_points:
        piece = text[prev:point].strip()
        if piece:
            sentences.append(piece)
        prev = point

    tail = text[prev:].strip()
    if tail:
        sentences.append(tail)

    return sentences
