"""
Sentence-level embedding utilities.
"""

from pathlib import Path
from typing import Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mlx-community/Qwen3.5-2B-MLX-8bit"
CACHE_DIR = Path(__file__).parent / "sentence_embedding_cache"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_language_model(backend: str = "transformers", device: torch.device | None = None) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if backend == "mlx":
        import mlx_lm

        model, _ = mlx_lm.load(MODEL_NAME)
        return model, tokenizer

    if device is None:
        device = get_device()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    model.eval()
    return model, tokenizer


@torch.no_grad()
def extract_token_embeddings(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    backend: str = "transformers",
) -> torch.Tensor:
    if backend == "mlx":
        import mlx.core as mx
        import numpy as np

        inputs = mx.array(input_ids.cpu().numpy())
        trunk = getattr(model, "model", None)
        if hasattr(model, "language_model"):
            trunk = getattr(model.language_model, "model", trunk)
        if trunk is None:
            raise ValueError("MLX model trunk not found")
        hidden = trunk(inputs).astype(mx.float32)
        hidden_torch = torch.from_numpy(np.array(hidden))
        return hidden_torch.to(input_ids.device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    return outputs.hidden_states[-1].float()


def expand_to_char_embeddings(token_embeddings: torch.Tensor, char_to_token: torch.Tensor) -> torch.Tensor:
    hidden_dim = token_embeddings.shape[-1]
    safe_indices = char_to_token.clamp(0, token_embeddings.shape[1] - 1)
    gather_idx = safe_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
    return torch.gather(token_embeddings, dim=1, index=gather_idx)


@torch.no_grad()
def extract_and_cache_embeddings(
    model: Any,
    dataloader,
    device: torch.device,
    cache_name: str = "train",
    backend: str = "transformers",
    base_cache_dir: Path | None = None,
) -> Path:
    cache_root = base_cache_dir or CACHE_DIR
    cache_path = cache_root / cache_name
    cache_path.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        save_file = cache_path / f"batch_{batch_idx:05d}.pt"
        if save_file.exists():
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        char_to_token = batch["char_to_token"].to(device)

        tok_emb = extract_token_embeddings(model, input_ids, attention_mask, backend=backend)
        char_emb = expand_to_char_embeddings(tok_emb, char_to_token)

        torch.save(
            {
                "char_embeddings": char_emb.cpu(),
                "char_labels": batch["char_labels"],
                "char_mask": batch["char_mask"],
                "spaceless": batch.get("spaceless", []),
            },
            save_file,
        )

    return cache_path
