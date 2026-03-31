"""
Sentence-level embedding utilities.
"""

from pathlib import Path
from typing import Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"
CACHE_DIR = Path(__file__).parent / "sentence_embedding_cache"


def _get_transformers_load_dtype(device: torch.device) -> torch.dtype:
    """Prefer bf16 for transformers; fallback only when the runtime cannot support it."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("Warning: CUDA device does not support bfloat16; falling back to float16.")
        return torch.float16

    if device.type == "mps":
        return torch.bfloat16

    # CPU path keeps bf16 as requested.
    return torch.bfloat16


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_language_model(
    backend: str = "transformers",
    device: torch.device | None = None,
    model_name: str | None = None,
    adapter_path: str | None = None,
) -> Tuple[Any, Any]:
    resolved_model_name = model_name or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)

    if backend == "mlx":
        import mlx_lm

        model, _ = mlx_lm.load(resolved_model_name, adapter_path=adapter_path)
        return model, tokenizer

    if device is None:
        device = get_device()
    load_dtype = _get_transformers_load_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_name,
        device_map="auto",
        torch_dtype=load_dtype,
    )
    model.eval()
    return model, tokenizer


def _get_transformer_backbone(model: Any) -> Any:
    """Return transformer trunk when model exposes an LM head wrapper."""
    current = model

    if hasattr(current, "get_base_model"):
        try:
            current = current.get_base_model()
        except Exception:
            pass

    for _ in range(6):
        next_model = getattr(current, "model", None)
        if next_model is None or next_model is current:
            break

        cls_name = current.__class__.__name__
        is_wrapper = cls_name.startswith("Peft") or cls_name.startswith("Lora")
        has_lm_head = hasattr(current, "lm_head") or cls_name.endswith("ForCausalLM")
        if is_wrapper or has_lm_head:
            current = next_model
            continue

        break

    return current


def forward_transformer_hidden_states_only(
    model: Any,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Forward only the transformer backbone and return hidden states."""
    backbone = _get_transformer_backbone(model)

    kwargs: dict[str, Any] = {
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if input_ids is not None:
        kwargs["input_ids"] = input_ids
    if inputs_embeds is not None:
        if hasattr(backbone, "dtype"):
            inputs_embeds = inputs_embeds.to(backbone.dtype)
        kwargs["inputs_embeds"] = inputs_embeds

    try:
        outputs = backbone(use_cache=False, **kwargs)
    except TypeError:
        outputs = backbone(**kwargs)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states

    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is not None:
        return (last_hidden,)

    raise ValueError("Model forward did not return hidden states")


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

    hidden_states = forward_transformer_hidden_states_only(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    num_layers_to_average = min(4, len(hidden_states))
    stacked = torch.stack(hidden_states[-num_layers_to_average:], dim=0)
    return stacked.mean(dim=0).float()


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
        token_labels = batch["token_labels"]
        token_mask = batch["token_mask"]

        tok_emb = extract_token_embeddings(model, input_ids, attention_mask, backend=backend)

        torch.save(
            {
                "token_embeddings": tok_emb.cpu(),
                "token_labels": token_labels,
                "token_mask": token_mask,
                "spaceless": batch.get("spaceless", []),
            },
            save_file,
        )

    return cache_path
