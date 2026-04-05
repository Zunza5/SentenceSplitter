"""
Sentence-level embedding utilities.
"""

import os
from pathlib import Path
from typing import Any, Tuple

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-2B"
CACHE_DIR = Path(__file__).parent / "sentence_embedding_cache"
GGUF_FILE_ENV = "SENTENCE_GGUF_FILE"


def _is_local_gguf(model_name: str) -> bool:
    return model_name.lower().endswith(".gguf") and Path(model_name).expanduser().exists()


def _select_4bit_gguf_filename(repo_id: str) -> str:
    """
    Pick a 4-bit GGUF file from a Hugging Face GGUF repository.

    Priority:
      1) explicit filename via SENTENCE_GGUF_FILE
      2) preferred naming order among available *.gguf files
    """
    explicit = os.getenv(GGUF_FILE_ENV)
    if explicit:
        return explicit

    try:
        from huggingface_hub import list_repo_files
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to auto-select a 4-bit GGUF file. "
            f"Install it or set {GGUF_FILE_ENV}."
        ) from exc

    files = list_repo_files(repo_id)
    gguf_files = [f for f in files if f.lower().endswith(".gguf")]
    if not gguf_files:
        raise ValueError(f"No GGUF files found in repo {repo_id}")

    lowered = {f.lower(): f for f in gguf_files}
    preferred_tokens = [
        "q4_k_m",
        "q4_k_s",
        "q4_1",
        "q4_0",
        "4bit",
    ]
    for token in preferred_tokens:
        for low_name, original in lowered.items():
            if token in low_name:
                return original

    raise ValueError(
        f"No 4-bit GGUF file found in repo {repo_id}. Available files: {gguf_files}"
    )


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

        if _is_local_gguf(MODEL_NAME):
            model, _ = mlx_lm.load(MODEL_NAME)
            return model, tokenizer

        if MODEL_NAME.lower().endswith("-gguf") or "gguf" in MODEL_NAME.lower():
            try:
                from huggingface_hub import hf_hub_download
            except Exception as exc:
                raise RuntimeError(
                    "huggingface_hub is required to download GGUF files for MLX."
                ) from exc

            gguf_file = _select_4bit_gguf_filename(MODEL_NAME)
            gguf_local = hf_hub_download(repo_id=MODEL_NAME, filename=gguf_file)
            model, _ = mlx_lm.load(gguf_local)
            return model, tokenizer

        # Non-GGUF repo path: keep default MLX loading behavior.
        model, _ = mlx_lm.load(MODEL_NAME)
        return model, tokenizer

    if device is None:
        device = get_device()

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config_type = type(config)

    # On macOS MPS, load on CPU first to avoid cache allocator issues during warmup,
    # then move to MPS for inference
    load_device = "cpu" if device.type == "mps" else "auto"
    
    # Prefer CausalLM for decoder-only checkpoints (e.g. Qwen3.5) because
    # AutoModel can resolve to a wrapper class with incompatible state_dict keys.
    if config_type in AutoModelForCausalLM._model_mapping:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
        )
        print(f"Loaded causal LM model '{MODEL_NAME}' for embedding extraction")
    elif config_type in AutoModel._model_mapping:
        model = AutoModel.from_pretrained(
            MODEL_NAME, 
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
        )
        print(f"Loaded encoder model '{MODEL_NAME}' for embedding extraction")
    else:
        raise ValueError(
            f"Unsupported model config for embedding extraction: {config_type.__name__}"
        )

    model.eval()
    
    # Move to target device if loaded on CPU
    if device.type == "mps" and load_device == "cpu":
        model = model.to(device)
    
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
    hidden_states = outputs.hidden_states
    num_layers_to_average = min(4, len(hidden_states))
    stacked = torch.stack(hidden_states[-num_layers_to_average:], dim=0)
    return stacked.mean(dim=0)


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
