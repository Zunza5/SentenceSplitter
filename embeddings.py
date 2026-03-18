"""
Minerva embedding extraction and perplexity computation.

Handles:
  - Loading the Minerva model
  - Extracting last-layer hidden states
  - Expanding token-level embeddings to character-level
  - Computing perplexity for the verification step
  - Offline embedding caching
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "sapienzanlp/Minerva-1B-base-v1.0"
CACHE_DIR = Path(__file__).parent / "embedding_cache"


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_minerva(device: torch.device | None = None):
    """
    Load the Minerva model and tokenizer.

    Returns:
        model: the Minerva causal LM in eval mode
        tokenizer: the corresponding tokenizer
    """
    if device is None:
        device = get_device()

    print(f"Loading Minerva model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model = model.to(device)
    model.eval()
    print("  → Model loaded.")
    return model, tokenizer


@torch.no_grad()
def extract_token_embeddings(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Extract last-layer hidden state embeddings.

    Args:
        model: Minerva model
        input_ids: (batch, seq_len) token IDs
        attention_mask: (batch, seq_len) attention mask

    Returns:
        (batch, seq_len, hidden_dim) last hidden state in float32
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # Last hidden state
    hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
    return hidden.float()  # convert from bfloat16 to float32 for MLP


def expand_to_char_embeddings(
    token_embeddings: torch.Tensor,
    char_to_token: torch.Tensor,
) -> torch.Tensor:
    """
    Expand token-level embeddings to character-level using the alignment map.

    Args:
        token_embeddings: (batch, num_tokens, hidden_dim)
        char_to_token: (batch, num_chars) mapping char_pos → token_idx

    Returns:
        (batch, num_chars, hidden_dim) character-level embeddings
    """
    batch_size, num_chars = char_to_token.shape
    hidden_dim = token_embeddings.shape[-1]

    # Clamp indices to valid range
    max_tok_idx = token_embeddings.shape[1] - 1
    safe_indices = char_to_token.clamp(0, max_tok_idx)

    # Gather: expand indices to (batch, num_chars, hidden_dim)
    idx = safe_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
    char_embeddings = torch.gather(token_embeddings, dim=1, index=idx)
    return char_embeddings


@torch.no_grad()
def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device | None = None,
) -> float:
    """
    Compute the perplexity of a given text string.

    Args:
        model: Minerva model
        tokenizer: Minerva tokenizer
        text: input text string
        device: computation device

    Returns:
        perplexity (float)
    """
    if device is None:
        device = next(model.parameters()).device

    encoding = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoding["input_ids"].to(device)

    if input_ids.shape[1] < 2:
        return float("inf")

    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss.item()
    return torch.exp(torch.tensor(loss)).item()


def extract_and_cache_embeddings(
    model: AutoModelForCausalLM,
    dataloader,
    device: torch.device,
    cache_name: str = "train",
) -> Path:
    """
    Extract embeddings for all batches in a dataloader and save to disk.

    Each batch is saved as a dict with:
        - char_embeddings: (batch, num_chars, hidden_dim)
        - char_labels: (batch, num_chars)
        - char_mask: (batch, num_chars)

    Returns:
        Path to the cache directory
    """
    cache_path = CACHE_DIR / cache_name
    cache_path.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        save_file = cache_path / f"batch_{batch_idx:05d}.pt"
        if save_file.exists():
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        char_to_token = batch["char_to_token"].to(device)

        # Extract token embeddings
        tok_emb = extract_token_embeddings(model, input_ids, attention_mask)

        # Expand to character level
        char_emb = expand_to_char_embeddings(tok_emb, char_to_token)

        # Save to disk (on CPU to save GPU memory)
        torch.save(
            {
                "char_embeddings": char_emb.cpu(),
                "char_labels": batch["char_labels"],
                "char_mask": batch["char_mask"],
            },
            save_file,
        )

        if (batch_idx + 1) % 10 == 0:
            print(f"  Cached {batch_idx + 1} batches...")

    print(f"  → Embeddings cached to {cache_path}")
    return cache_path
