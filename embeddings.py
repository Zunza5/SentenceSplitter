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
from typing import Any, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mlx-community/Qwen3.5-2B-MLX-8bit"
CACHE_DIR = Path(__file__).parent / "embedding_cache"


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_language_model(backend: str = "transformers", device: torch.device | None = None) -> Tuple[Any, Any]:
    """
    Load the Language Model and tokenizer dynamically handling transformers or mlx.
    """
    print(f"Loading {MODEL_NAME} with backend '{backend}'...")
    
    # We always use the Hugging Face tokenizer to maintain full API compatibility 
    # (like return_offsets_mapping and call semantics).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if backend == "mlx":
        import mlx_lm
        model, _ = mlx_lm.load(MODEL_NAME)
        print("  → MLX Model loaded.")
        return model, tokenizer
    else:
        if device is None:
            device = get_device()
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto"
        )
        model.eval()
        print("  → Transformers Model loaded.")
        return model, tokenizer


@torch.no_grad()
def extract_token_embeddings(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    backend: str = "transformers"
) -> torch.Tensor:
    """
    Extract last-layer hidden state embeddings safely mapped between backends.
    """
    if backend == "mlx":
        import mlx.core as mx
        import numpy as np
        
        # MLX strictly runs on np/mx arrays natively
        inputs = mx.array(input_ids.cpu().numpy())
        
        # Locate the transformer trunk depending on mlx-lm model architecture
        trunk = getattr(model, "model", None)
        if hasattr(model, "language_model"):
            trunk = getattr(model.language_model, "model", trunk)
            
        if trunk is not None:
            # Forward directly through Qwen/Llama trunk without lm_head
            hidden = trunk(inputs)
        else:
            raise ValueError("MLX model trunk not found.")
            
        # Float conversions: explicitly cast to float32 in MLX *before* numpy, 
        # because numpy/torch lack native bfloat16 stable support in many cases.
        hidden_fp32 = hidden.astype(mx.float32)
        hidden_torch = torch.from_numpy(np.array(hidden_fp32))
        return hidden_torch.to(input_ids.device)

    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        return hidden.float()


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
    model: Any,
    tokenizer: Any,
    text: str,
    device: torch.device | None = None,
    backend: str = "transformers"
) -> float:
    # We maintain this helper but now redirect to batch logic for simplicity.
    return compute_perplexity_batch(model, tokenizer, [text], device, backend)[0]


@torch.no_grad()
def compute_perplexity_batch(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: torch.device | None = None,
    backend: str = "transformers"
) -> list[float]:
    """
    Compute the perplexity of a list of text strings in batch.
    """
    if not texts:
        return []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    pad_id = getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id)

    if backend == "mlx":
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        
        encoding = tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            add_special_tokens=True
        )
        input_ids = mx.array(encoding["input_ids"])
        
        if input_ids.shape[1] < 2:
            # Not enough tokens to shift and compute loss
            return [1e6] * len(texts)
        
        logits = model(input_ids)
        
        shift_logits = logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]
        
        losses = nn.losses.cross_entropy(shift_logits, shift_labels)
        mask = (shift_labels != pad_id)
        
        masked_losses = losses * mask
        sum_loss = mx.sum(masked_losses, axis=1)
        valid_tokens = mx.sum(mask, axis=1)
        
        seq_loss = sum_loss / mx.maximum(valid_tokens, 1.0)
        perplexities = mx.exp(seq_loss)
        return np.array(perplexities).tolist()

    if device is None:
        device = next(model.parameters()).device

    # Tokenize with padding
    encoding = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    if input_ids.shape[1] < 2:
        return [1e6] * len(texts)

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    
    logits = outputs.logits 
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    token_loss = loss_fct(shift_logits, shift_labels)
    token_loss = token_loss.view(input_ids.shape[0], -1) 
    
    shift_mask = (shift_labels != -100).view(input_ids.shape[0], -1).float()
    sum_loss = (token_loss * shift_mask).sum(dim=1)
    active_tokens = shift_mask.sum(dim=1)
    
    seq_loss = sum_loss / torch.clamp(active_tokens, min=1.0)
    perplexities = torch.exp(seq_loss).cpu().tolist()
    
    return perplexities


def extract_and_cache_embeddings(
    model: Any,
    dataloader,
    device: torch.device,
    cache_name: str = "train",
    backend: str = "transformers"
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
        tok_emb = extract_token_embeddings(model, input_ids, attention_mask, backend=backend)

        # Expand to character level
        char_emb = expand_to_char_embeddings(tok_emb, char_to_token)

        # Save to disk (on CPU to save GPU memory)
        torch.save(
            {
                "char_embeddings": char_emb.cpu(),
                "char_labels": batch["char_labels"],
                "char_mask": batch["char_mask"],
                "spaceless": batch["spaceless"],
            },
            save_file,
        )

        if (batch_idx + 1) % 10 == 0:
            print(f"  Cached {batch_idx + 1} batches...")

    print(f"  → Embeddings cached to {cache_path}")
    return cache_path
