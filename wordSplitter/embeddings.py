"""
LLM embedding extraction.

Handles:
  - Loading the embedding model (Qwen3-Embedding-4B)
  - Extracting last-layer hidden states (per-token)
  - Expanding token-level embeddings to character-level
  - Offline embedding caching
"""

import os
from pathlib import Path
from typing import Any, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

MODEL_NAME = "mlx-community/Qwen3-Embedding-0.6B-mxfp8"

# Instruction prompt for sentence splitting task
INSTRUCT_PROMPT = "Instruct: Identify sentence boundaries in the following text\nQuery:"

CACHE_DIR = Path(__file__).parent.parent / "embedding_cache"


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_language_model(backend: str = "transformers", device: torch.device | None = None) -> Tuple[Any, Any]:
    """
    Load the Embedding Model and tokenizer.
    """
    print(f"Loading {MODEL_NAME} with backend '{backend}'...")
    
    if backend == "mlx":
        from mlx_embeddings import load as mlx_load
        model, tokenizer = mlx_load(MODEL_NAME)
        print("  → MLX Embedding Model loaded.")
        return model, tokenizer
    else:
        from transformers import AutoModel
        
        # Use HF tokenizer with left padding (required for embedding models)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
        
        if device is None:
            device = get_device()
        
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            device_map="auto"
        )
        model.eval()
        print("  → Transformers Embedding Model loaded.")
        return model, tokenizer


@torch.no_grad()
def extract_token_embeddings(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    backend: str = "transformers",
    layer_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Extract last-layer hidden state embeddings (per-token) from the embedding model.
    """
    if backend == "mlx":
        import mlx.core as mx
        import numpy as np
        
        inputs = mx.array(input_ids.cpu().numpy())
        
        # Access the transformer trunk of the embedding model
        trunk = getattr(model, "model", None)
        if hasattr(model, "language_model"):
            trunk = getattr(model.language_model, "model", trunk)
            
        if trunk is not None:
            if layer_idx is not None:
                # Manual forward pass up to layer_idx
                # This assumes a standard Llama/Qwen-like structure in MLX
                # embed_tokens -> layers -> norm
                x = trunk.embed_tokens(inputs)
                for i in range(layer_idx + 1):
                    x = trunk.layers[i](x, mask=None)
                hidden = x
            else:
                hidden = trunk(inputs)
        else:
            raise ValueError("MLX model trunk not found.")
            
        hidden_fp32 = hidden.astype(mx.float32)
        hidden_torch = torch.from_numpy(np.array(hidden_fp32))
        return hidden_torch.to(input_ids.device)

    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=(layer_idx is not None)
        )
        
        if layer_idx is not None:
            # hidden_states is a tuple: (embedding_layer, layer_0, ..., layer_N-1)
            # layer_idx = 0 usually means after the first transformer block
            # For Qwen, hidden_states[0] is the base embeddings.
            # layer_idx=N-1 would be the output of the last block.
            # transformers index them as 0 to num_hidden_layers
            # We want the output of the specific block
            hidden = outputs.hidden_states[layer_idx + 1]
        else:
            # AutoModel gives us last_hidden_state directly
            hidden = outputs.last_hidden_state
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


def extract_and_cache_embeddings(
    model: Any,
    dataloader,
    device: torch.device,
    cache_name: str = "train",
    backend: str = "transformers",
    base_cache_dir: Path | None = None,
    layer_idx: Optional[int] = None,
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
    if base_cache_dir is None:
        base_cache_dir = CACHE_DIR
    
    # Suffix cache name with layer if not default
    actual_cache_name = cache_name if layer_idx is None else f"{cache_name}_L{layer_idx}"
    cache_path = base_cache_dir / actual_cache_name
    cache_path.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        save_file = cache_path / f"batch_{batch_idx:05d}.pt"
        if save_file.exists():
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        char_to_token = batch["char_to_token"].to(device)

        # Extract token embeddings
        tok_emb = extract_token_embeddings(model, input_ids, attention_mask, backend=backend, layer_idx=layer_idx)

        # Expand to character level
        char_emb = expand_to_char_embeddings(tok_emb, char_to_token)

        # Save to disk (on CPU to save GPU memory)
        torch.save(
            {
                "char_embeddings": char_emb.cpu(),
                "char_labels": batch["char_labels"],
                "char_mask": batch["char_mask"],
                "spaceless": batch.get("spaceless", batch.get("text", [])),
            },
            save_file,
        )

        if (batch_idx + 1) % 10 == 0:
            print(f"  Cached {batch_idx + 1} batches...")

    print(f"  → Embeddings cached to {cache_path}")
    return cache_path


def extract_and_cache_token_embeddings(
    model: Any,
    dataloader,
    device: torch.device,
    cache_name: str = "train",
    backend: str = "transformers",
    base_cache_dir: Path | None = None,
    layer_idx: Optional[int] = None,
) -> Path:
    """
    Extract token-level embeddings and save to disk (no char expansion).

    Each batch is saved as a dict with:
        - token_embeddings: (batch, num_tokens, hidden_dim)
        - token_labels: (batch, num_tokens)
        - token_mask: (batch, num_tokens)
        - text: list of original text strings

    Returns:
        Path to the cache directory
    """
    if base_cache_dir is None:
        base_cache_dir = CACHE_DIR
    
    # Suffix cache name with layer if not default
    actual_cache_name = cache_name if layer_idx is None else f"{cache_name}_L{layer_idx}"
    cache_path = base_cache_dir / actual_cache_name
    cache_path.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        save_file = cache_path / f"batch_{batch_idx:05d}.pt"
        if save_file.exists():
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Extract token embeddings (no char expansion)
        tok_emb = extract_token_embeddings(model, input_ids, attention_mask, backend=backend, layer_idx=layer_idx)

        # Save to disk (on CPU to save GPU memory)
        torch.save(
            {
                "token_embeddings": tok_emb.cpu(),
                "token_labels": batch["token_labels"],
                "token_mask": batch["token_mask"],
                "token_offsets": batch["token_offsets"],
                "text": batch.get("text", []),
            },
            save_file,
        )

        if (batch_idx + 1) % 10 == 0:
            print(f"  Cached {batch_idx + 1} batches...")

    print(f"  → Token embeddings cached to {cache_path}")
    return cache_path
