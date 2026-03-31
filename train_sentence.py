"""
Training pipeline for Sentence Splitter MLP.

Phase 1: Extract LLM embeddings offline for sentence chunks and cache.
Phase 2: Train the MLP on cached sentence-level embeddings.
"""

import argparse
import functools
import importlib
import json
import math
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data_sentence import (
    get_sentence_dataloader,
    UD_URLS,
    SentenceSplitDataset,
    collate_sentence_fn,
)
from sentence_embeddings import (
    load_language_model,
    extract_and_cache_embeddings,
    extract_token_embeddings,
    forward_transformer_hidden_states_only,
    get_device,
    MODEL_NAME,
)
from transformers import AutoTokenizer
from model import SpacePredictorMLP, FocalLoss

SENTENCE_CACHE_DIR = Path(__file__).parent / "sentence_embedding_cache"
PREFIX_CACHE_DIR = SENTENCE_CACHE_DIR / "qwen_prefix_cache"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BEST_SENTENCE_CKPT = CHECKPOINT_DIR / "best_sentence_mlp.pt"
BEST_SENTENCE_LORA_CKPT = CHECKPOINT_DIR / "best_sentence_mlp_lora.pt"
QWEN_LORA_ADAPTER_DIR = CHECKPOINT_DIR / "qwen_lora_last2"
QWEN_MLX_LORA_ADAPTER_DIR = CHECKPOINT_DIR / "qwen_mlx_lora_last2"
MLX_LORA_DATA_DIR = Path(__file__).parent / "data_cache" / "mlx_lora_data"


def set_deterministic_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Keep deterministic behavior when possible without crashing on unsupported ops.
    torch.use_deterministic_algorithms(True, warn_only=True)


# Keep this cache small: large maxsize can retain many mmap'ed batch files in RAM.
_BATCH_FILE_CACHE_SIZE = int(os.getenv("SENT_SPLIT_BATCH_CACHE_SIZE", "8"))


@functools.lru_cache(maxsize=_BATCH_FILE_CACHE_SIZE)
def _load_batch_file(file_path: Path):
    return torch.load(file_path, weights_only=True, mmap=True)


class CachedEmbeddingDataset(Dataset):
    """Loads pre-extracted sentence embeddings from disk lazily."""

    def __init__(self, cache_path: Path):
        self.files = sorted(cache_path.glob("batch_*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No cached embeddings found in {cache_path}. Run extraction first."
            )

        self.index_map = []
        for f in self.files:
            data = torch.load(f, weights_only=True)
            batch_size = data["token_embeddings"].shape[0]
            for i in range(batch_size):
                self.index_map.append((f, i))
            del data

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, inner_idx = self.index_map[idx]
        data = _load_batch_file(file_path)
        mask = data["token_mask"][inner_idx]
        sample = {
            "token_embeddings": data["token_embeddings"][inner_idx][mask],
            "token_labels": data["token_labels"][inner_idx][mask],
        }
        if "spaceless" in data:
            if isinstance(data["spaceless"], (list, tuple)):
                sample["spaceless"] = data["spaceless"][inner_idx]
            else:
                sample["spaceless"] = data["spaceless"]
        return sample


class PrefixHiddenDataset(Dataset):
    """Loads cached hidden states at layer N-2 while preserving full token sequence."""

    def __init__(self, cache_path: Path):
        self.files = sorted(cache_path.glob("batch_*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No cached prefix hidden states found in {cache_path}."
            )

        self.index_map = []
        for f in self.files:
            data = torch.load(f, weights_only=True)
            batch_size = data["token_embeddings"].shape[0]
            for i in range(batch_size):
                self.index_map.append((f, i))
            del data

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, inner_idx = self.index_map[idx]
        data = _load_batch_file(file_path)
        return {
            "token_embeddings": data["token_embeddings"][inner_idx],
            "token_labels": data["token_labels"][inner_idx],
            "token_mask": data["token_mask"][inner_idx],
            "attention_mask": data["attention_mask"][inner_idx],
            "spaceless": data.get("spaceless", [""])[inner_idx] if isinstance(data.get("spaceless", []), (list, tuple)) else data.get("spaceless", ""),
        }


def collate_prefix_hidden_fn(batch):
    max_len = max(s["token_embeddings"].shape[0] for s in batch)
    hidden_dim = batch[0]["token_embeddings"].shape[-1]

    embeddings = torch.zeros(len(batch), max_len, hidden_dim)
    labels = torch.full((len(batch), max_len), -1.0)
    token_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, sample in enumerate(batch):
        length = sample["token_embeddings"].shape[0]
        embeddings[i, :length] = sample["token_embeddings"]
        labels[i, :length] = sample["token_labels"]
        token_mask[i, :length] = sample["token_mask"].bool()
        attention_mask[i, :length] = sample["attention_mask"].long()

    return {
        "token_embeddings": embeddings,
        "token_labels": labels,
        "token_mask": token_mask,
        "attention_mask": attention_mask,
        "spaceless": [s.get("spaceless", "") for s in batch],
    }


class _PassThroughDecoderLayer(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,)


class BalancedStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self, cache_dirs: list[Path], batch_size: int):
        self.cache_dirs = cache_dirs
        self.num_datasets = len(cache_dirs)
        self.samples_per_ds = max(1, batch_size // self.num_datasets)
        self.actual_batch_size = self.samples_per_ds * self.num_datasets
        
        max_samples = 0
        for d in cache_dirs:
            num_files = len(list(d.glob("batch_*.pt")))
            max_samples = max(max_samples, num_files * 8)
            
        self.total_batches = int(max_samples / self.samples_per_ds) if self.samples_per_ds else 0

    def __len__(self):
        return self.total_batches * self.actual_batch_size

    def __iter__(self):
        import random
        
        def file_generator(d_path):
            files = sorted(list(d_path.glob("batch_*.pt")))
            if not files:
                while True:
                    yield None
                    
            while True:
                random.shuffle(files)
                for f in files:
                    data = torch.load(f, weights_only=True, mmap=True)
                    bs = data["token_embeddings"].shape[0]
                    indices = list(range(bs))
                    random.shuffle(indices)
                    for i in indices:
                        spaceless_val = data.get("spaceless", [""])
                        if isinstance(spaceless_val, (list, tuple)):
                            s_val = spaceless_val[i] if i < len(spaceless_val) else ""
                        else:
                            s_val = spaceless_val
                            
                        yield {
                            "token_embeddings": data["token_embeddings"][i],
                            "token_labels": data["token_labels"][i],
                            "token_mask": data["token_mask"][i],
                            "attention_mask": data["attention_mask"][i],
                            "spaceless": s_val
                        }

        generators = [file_generator(d) for d in self.cache_dirs]
        
        for _ in range(self.total_batches):
            batch = []
            for g in generators:
                for _ in range(self.samples_per_ds):
                    val = next(g)
                    if val is not None:
                        batch.append(val)
            
            random.shuffle(batch)
            yield from batch


def _build_balanced_sample_weights(datasets: list[Dataset]) -> torch.Tensor:
    """Return per-sample weights so each source dataset has equal sampling mass."""
    if not datasets:
        raise ValueError("datasets must not be empty")

    weights: list[float] = []
    for ds in datasets:
        ds_len = len(ds)
        if ds_len == 0:
            raise ValueError("all datasets must contain at least one sample")
        weights.extend([1.0 / ds_len] * ds_len)

    return torch.tensor(weights, dtype=torch.double)


def cached_collate_fn(batch):
    max_len = max(s["token_embeddings"].shape[0] for s in batch)
    hidden_dim = batch[0]["token_embeddings"].shape[-1]

    embeddings = torch.zeros(len(batch), max_len, hidden_dim)
    labels = torch.full((len(batch), max_len), -1.0)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, sample in enumerate(batch):
        length = sample["token_embeddings"].shape[0]
        embeddings[i, :length] = sample["token_embeddings"]
        labels[i, :length] = sample["token_labels"]
        mask[i, :length] = True

    return {
        "token_embeddings": embeddings,
        "token_labels": labels,
        "token_mask": mask,
        "spaceless": [s.get("spaceless", "") for s in batch],
    }


def _get_transformer_layers(model) -> list:
    curr = model
    while curr is not None:
        if hasattr(curr, "layers"):
            return list(curr.layers)
        if hasattr(curr, "model") and hasattr(curr.model, "layers"):
            return list(curr.model.layers)
        
        if hasattr(curr, "base_model") and curr.base_model is not curr:
            curr = curr.base_model
        elif hasattr(curr, "model") and curr.model is not curr:
            curr = curr.model
        else:
            break
            
    raise ValueError("Unable to locate transformer layers for LoRA injection")


def _configure_qwen_lora_last_layers(
    model,
    layers_to_finetune: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    layers = _get_transformer_layers(model)
    if len(layers) < layers_to_finetune:
        raise ValueError(
            f"Requested last {layers_to_finetune} layers but model has only {len(layers)} layers"
        )

    layer_indexes = list(range(len(layers) - layers_to_finetune, len(layers)))
    try:
        peft_module = importlib.import_module("peft")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PEFT is required for LoRA finetuning. Install dependencies to continue."
        ) from exc

    peft_config = peft_module.LoraConfig(
        task_type=peft_module.TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        layers_to_transform=layer_indexes,
        layers_pattern=["layers"],
    )
    model = peft_module.get_peft_model(model, peft_config)
    return model, layer_indexes


def _replace_lower_layers_with_passthrough(model, layers_to_finetune: int = 2):
    layers = _get_transformer_layers(model)
    cutoff = len(layers) - layers_to_finetune
    if cutoff <= 0:
        raise ValueError("cutoff must be positive when using cached prefix training")

    for i in range(cutoff):
        layers[i] = _PassThroughDecoderLayer()
    return cutoff


def _write_mlx_jsonl(texts: list[str], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in texts:
            s = t.strip()
            if not s:
                continue
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")


def _collect_split_texts(splits: list[str]) -> list[str]:
    from data_sentence import get_sentences_for_split

    texts: list[str] = []
    for split in splits:
        for sent in get_sentences_for_split(split):
            text = " ".join(sent).strip()
            if text:
                texts.append(text)
    return texts


def _build_mlx_lora_dataset(train_splits: list[str], dev_splits: list[str]) -> Path:
    data_dir = MLX_LORA_DATA_DIR
    train_texts = _collect_split_texts(train_splits)
    dev_texts = _collect_split_texts(dev_splits)

    if not train_texts:
        raise ValueError("No train texts available to build MLX LoRA dataset")
    if not dev_texts:
        print("Warning: No dev texts found for MLX LoRA. Falling back to train subset for valid.jsonl")
        dev_texts = train_texts[: min(1000, len(train_texts))]

    _write_mlx_jsonl(train_texts, data_dir / "train.jsonl")
    _write_mlx_jsonl(dev_texts, data_dir / "valid.jsonl")
    if not (data_dir / "test.jsonl").exists():
        _write_mlx_jsonl(dev_texts[: min(1000, len(dev_texts))], data_dir / "test.jsonl")

    return data_dir


def _run_mlx_lora_finetune(
    model_name: str,
    train_splits: list[str],
    dev_splits: list[str],
    layers_to_finetune: int,
    batch_size: int,
    epochs: int,
    lr: float,
    max_chars: int,
    seed: int,
    adapter_dir: Path,
):
    data_dir = _build_mlx_lora_dataset(train_splits, dev_splits)

    with open(data_dir / "train.jsonl", "r", encoding="utf-8") as f:
        num_train = sum(1 for _ in f)
    iters = max(1, epochs * math.ceil(num_train / max(batch_size, 1)))
    steps_per_eval = max(20, min(200, iters // 4 if iters >= 4 else 20))
    steps_per_report = max(10, min(100, steps_per_eval // 2))

    adapter_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--train",
        "--model",
        model_name,
        "--data",
        str(data_dir),
        "--fine-tune-type",
        "lora",
        "--num-layers",
        str(layers_to_finetune),
        "--batch-size",
        str(batch_size),
        "--iters",
        str(iters),
        "--learning-rate",
        str(lr),
        "--max-seq-length",
        str(max_chars),
        "--adapter-path",
        str(adapter_dir),
        "--seed",
        str(seed),
        "--steps-per-report",
        str(steps_per_report),
        "--steps-per-eval",
        str(steps_per_eval),
        "--save-every",
        str(steps_per_eval),
    ]

    print("Running MLX LoRA:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _get_mlx_text_model(model):
    """Return the MLX text trunk exposing layers/embed/norm."""
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "model"):
        return language_model.model
    if hasattr(model, "model"):
        return model.model
    raise ValueError("Unable to locate MLX text model trunk")


def _mlx_prefix_hidden_states(
    model,
    input_ids: torch.Tensor,
    prefix_cutoff: int,
) -> Any:
    """Run MLX forward up to layer N-2 (exclusive of last trainable layers)."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    text_model = _get_mlx_text_model(model)
    mx_ids = mx.array(input_ids.cpu().numpy())
    hidden_states = text_model.embed_tokens(mx_ids)

    fa_mask = create_attention_mask(hidden_states, None)
    ssm_mask = create_ssm_mask(hidden_states, None)

    for layer in text_model.layers[:prefix_cutoff]:
        mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
        hidden_states = layer(hidden_states, mask=mask, cache=None)

    return hidden_states


def _mlx_tail_from_prefix_hidden(
    model,
    prefix_hidden: Any,
    prefix_cutoff: int,
) -> Any:
    """Run only the final trainable layers from cached prefix hidden states."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    text_model = _get_mlx_text_model(model)
    hidden_states = prefix_hidden

    fa_mask = create_attention_mask(hidden_states, None)
    ssm_mask = create_ssm_mask(hidden_states, None)

    for layer in text_model.layers[prefix_cutoff:]:
        mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
        hidden_states = layer(hidden_states, mask=mask, cache=None)

    hidden_states = text_model.norm(hidden_states)
    return hidden_states


@torch.no_grad()
def extract_and_cache_prefix_hidden_states_mlx(
    llm_model,
    dataloader,
    cache_name: str,
    prefix_cutoff: int,
    base_cache_dir: Path | None = None,
) -> Path:
    """Cache MLX hidden states up to layer N-2 for each batch."""
    import mlx.core as mx
    import numpy as np

    cache_root = base_cache_dir or PREFIX_CACHE_DIR
    cache_path = cache_root / cache_name
    cache_path.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        save_file = cache_path / f"batch_{batch_idx:05d}.pt"
        if save_file.exists():
            continue

        prefix_hidden = _mlx_prefix_hidden_states(
            llm_model,
            input_ids=batch["input_ids"],
            prefix_cutoff=prefix_cutoff,
        )

        torch.save(
            {
                "token_embeddings": torch.from_numpy(np.array(prefix_hidden.astype(mx.float32))),
                "token_labels": batch["token_labels"],
                "token_mask": batch["token_mask"],
                "attention_mask": batch["attention_mask"],
                "spaceless": batch.get("spaceless", []),
            },
            save_file,
        )

    return cache_path


@torch.no_grad()
def materialize_mlx_tail_embeddings_from_prefix_cache(
    llm_model,
    prefix_cache_path: Path,
    cache_name: str,
    prefix_cutoff: int,
    out_cache_dir: Path | None = None,
) -> Path:
    """Convert cached prefix hidden states into final token embeddings with MLX tail layers."""
    import mlx.core as mx
    import numpy as np

    prefix_files = sorted(prefix_cache_path.glob("batch_*.pt"))
    if not prefix_files:
        raise FileNotFoundError(f"No prefix cache files found in {prefix_cache_path}")

    out_root = out_cache_dir or SENTENCE_CACHE_DIR
    out_path = out_root / cache_name
    out_path.mkdir(parents=True, exist_ok=True)

    for prefix_file in prefix_files:
        out_file = out_path / prefix_file.name
        if out_file.exists():
            continue

        data = torch.load(prefix_file, weights_only=True)
        prefix_hidden = mx.array(data["token_embeddings"].numpy())

        final_hidden = _mlx_tail_from_prefix_hidden(
            llm_model,
            prefix_hidden=prefix_hidden,
            prefix_cutoff=prefix_cutoff,
        )

        torch.save(
            {
                "token_embeddings": torch.from_numpy(np.array(final_hidden.astype(mx.float32))),
                "token_labels": data["token_labels"],
                "token_mask": data["token_mask"],
                "spaceless": data.get("spaceless", []),
            },
            out_file,
        )

    return out_path


def _extract_sentence_embeddings_mlx_cached_last_layers(
    batch_size: int,
    model_name: str,
    adapter_path: str,
    splits_to_extract: list[str],
    layers_to_finetune: int,
    max_chars: int,
    stride_chars: int,
    augment_prob: float,
) -> tuple[int, Path]:
    """
    MLX-only cached embedding pipeline:
    1) cache hidden states up to layer N-2 with base model
    2) run only tail layers with finetuned adapter to build final token embeddings
    """
    base_model, tokenizer = load_language_model(
        backend="mlx",
        model_name=model_name,
        adapter_path=None,
    )
    lora_model, _ = load_language_model(
        backend="mlx",
        model_name=model_name,
        adapter_path=adapter_path,
    )

    text_model = _get_mlx_text_model(base_model)
    num_layers = len(text_model.layers)
    prefix_cutoff = num_layers - layers_to_finetune
    if prefix_cutoff <= 0:
        raise ValueError("layers_to_finetune must be smaller than total MLX transformer layers")

    for split in splits_to_extract:
        print(f"\n{'='*60}")
        print(f"MLX cached extraction for {split} split (prefix cutoff={prefix_cutoff})")
        print(f"{'='*60}")

        loader = get_sentence_dataloader(
            split=split,
            batch_size=batch_size,
            tokenizer=tokenizer,
            shuffle=False,
            max_chars=max_chars,
            stride_chars=stride_chars,
            augment_prob=0.0,
            augmentation_mode="original",
        )

        prefix_cache_name = f"mlx_prefix_{split}_last{layers_to_finetune}"
        prefix_cache_path = extract_and_cache_prefix_hidden_states_mlx(
            llm_model=base_model,
            dataloader=loader,
            cache_name=prefix_cache_name,
            prefix_cutoff=prefix_cutoff,
            base_cache_dir=PREFIX_CACHE_DIR,
        )
        materialize_mlx_tail_embeddings_from_prefix_cache(
            llm_model=lora_model,
            prefix_cache_path=prefix_cache_path,
            cache_name=split,
            prefix_cutoff=prefix_cutoff,
            out_cache_dir=SENTENCE_CACHE_DIR,
        )

        if augment_prob > 0:
            loader_aug = get_sentence_dataloader(
                split=split,
                batch_size=batch_size,
                tokenizer=tokenizer,
                shuffle=False,
                max_chars=max_chars,
                stride_chars=stride_chars,
                augment_prob=augment_prob,
                augmentation_mode="augmented",
            )
            prefix_cache_name_aug = f"mlx_prefix_{split}_aug_last{layers_to_finetune}"
            prefix_cache_path_aug = extract_and_cache_prefix_hidden_states_mlx(
                llm_model=base_model,
                dataloader=loader_aug,
                cache_name=prefix_cache_name_aug,
                prefix_cutoff=prefix_cutoff,
                base_cache_dir=PREFIX_CACHE_DIR,
            )
            materialize_mlx_tail_embeddings_from_prefix_cache(
                llm_model=lora_model,
                prefix_cache_path=prefix_cache_path_aug,
                cache_name=f"{split}_aug",
                prefix_cutoff=prefix_cutoff,
                out_cache_dir=SENTENCE_CACHE_DIR,
            )

    return prefix_cutoff, PREFIX_CACHE_DIR


def _resolve_transformers_model_name_for_cached_lora(
    backend: str,
    model_name: str,
) -> tuple[str, str]:
    """
    Cached LoRA training with prefix hidden-state reuse is implemented with
    transformers forward pass. For MLX aliases, try a deterministic fallback
    to an equivalent HF model id.
    """
    if backend == "transformers":
        return backend, model_name

    if backend != "mlx":
        raise ValueError(f"Unsupported backend for cached finetuning: {backend}")

    resolved_model_name = model_name
    if model_name.startswith("mlx-community/"):
        alias = model_name.split("/", 1)[1]
        alias = alias.split("-OptiQ", 1)[0]
        if "/" in alias:
            resolved_model_name = alias
        else:
            resolved_model_name = f"Qwen/{alias}"

    print(
        "MLX cached LoRA uses transformers-style prefix cache + last-layer finetuning. "
        f"Requested model='{model_name}', resolved model='{resolved_model_name}'."
    )
    return "transformers", resolved_model_name


@torch.no_grad()
def extract_and_cache_prefix_hidden_states(
    llm_model,
    dataloader,
    device: torch.device,
    cache_name: str,
    prefix_layer_idx: int,
    base_cache_dir: Path | None = None,
) -> Path:
    cache_root = base_cache_dir or PREFIX_CACHE_DIR
    cache_path = cache_root / cache_name
    cache_path.mkdir(parents=True, exist_ok=True)

    llm_model.eval()
    for batch_idx, batch in enumerate(dataloader):
        save_file = cache_path / f"batch_{batch_idx:05d}.pt"
        if save_file.exists():
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        hidden_states = forward_transformer_hidden_states_only(
            model=llm_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if prefix_layer_idx + 1 >= len(hidden_states):
            raise ValueError("Invalid prefix_layer_idx for hidden state extraction")

        prefix_hidden = hidden_states[prefix_layer_idx + 1].float().cpu()
        torch.save(
            {
                "token_embeddings": prefix_hidden,
                "token_labels": batch["token_labels"],
                "token_mask": batch["token_mask"],
                "attention_mask": batch["attention_mask"],
                "spaceless": batch.get("spaceless", []),
            },
            save_file,
        )

    return cache_path


@torch.no_grad()
def evaluate(
    model: SpacePredictorMLP,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        emb = batch["token_embeddings"].to(device)
        token_mask = batch["token_mask"].to(device)
        labels = batch["token_labels"].cpu()
        mask = batch["token_mask"].cpu()

        preds, _ = model(emb, mask=token_mask)
        preds = preds.cpu()

        for i in range(preds.shape[0]):
            valid = mask[i]
            p = (preds[i][valid] > 0.5).int().tolist()
            l = labels[i][valid].int().tolist()
            all_preds.extend(p)
            all_labels.extend(l)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def evaluate_end_to_end(
    mlp: SpacePredictorMLP,
    llm_model,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    mlp.eval()
    llm_model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_mask = batch["token_mask"].to(device)
        labels = batch["token_labels"].cpu()
        valid_mask = batch["token_mask"].cpu()

        tok_emb = extract_token_embeddings(
            llm_model,
            input_ids,
            attention_mask,
            backend="transformers",
        )
        preds, _ = mlp(tok_emb, mask=token_mask)
        preds = preds.cpu()

        for i in range(preds.shape[0]):
            valid = valid_mask[i]
            p = (preds[i][valid] > 0.5).int().tolist()
            l = labels[i][valid].int().tolist()
            all_preds.extend(p)
            all_labels.extend(l)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def evaluate_cached_prefix_end_to_end(
    mlp: SpacePredictorMLP,
    llm_model,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    mlp.eval()
    llm_model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        prefix_hidden = batch["token_embeddings"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_mask = batch["token_mask"].to(device)
        labels = batch["token_labels"].cpu()
        valid_mask = batch["token_mask"].cpu()

        hidden_states = forward_transformer_hidden_states_only(
            model=llm_model,
            inputs_embeds=prefix_hidden,
            attention_mask=attention_mask,
        )
        tok_emb = hidden_states[-1].float()
        preds, _ = mlp(tok_emb, mask=token_mask)
        preds = preds.cpu()

        for i in range(preds.shape[0]):
            valid = valid_mask[i]
            p = (preds[i][valid] > 0.5).int().tolist()
            l = labels[i][valid].int().tolist()
            all_preds.extend(p)
            all_labels.extend(l)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def finetune_qwen_lora_sentence_splitter(
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    d_model: int = 256,
    dropout: float = 0.2,
    pos_weight: float = 10.0,
    patience: int = 3,
    aux_weight: float = 0.01,
    train_splits: list[str] | None = None,
    dev_splits: list[str] | None = None,
    max_chars: int = 1024,
    stride_chars: int = 512,
    augment_prob: float = 0.0,
    balanced_batches: bool = True,
    model_name: str = MODEL_NAME,
    layers_to_finetune: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
    backend: str = "transformers",
):
    set_deterministic_seed(seed)

    if backend == "mlx":
        _run_mlx_lora_finetune(
            model_name=model_name,
            train_splits=train_splits or ["train"],
            dev_splits=dev_splits or ["dev"],
            layers_to_finetune=layers_to_finetune,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            max_chars=max_chars,
            seed=seed,
            adapter_dir=QWEN_MLX_LORA_ADAPTER_DIR,
        )

        # Recompute embeddings with the finetuned MLX adapters, then train the MLP as usual.
        extract_sentence_embeddings(
            batch_size=batch_size,
            backend="mlx",
            augment_prob=augment_prob,
            max_chars=max_chars,
            stride_chars=stride_chars,
            seed=seed,
            adapter_path=str(QWEN_MLX_LORA_ADAPTER_DIR),
            splits_to_extract=sorted(set((train_splits or ["train"]) + (dev_splits or ["dev"]))),
            model_name=model_name,
        )
        train_sentence_mlp(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            d_model=d_model,
            dropout=dropout,
            pos_weight=pos_weight,
            aux_weight=aux_weight,
            train_splits=train_splits,
            dev_splits=dev_splits,
            augment_prob=augment_prob,
            balanced_batches=balanced_batches,
            seed=seed,
        )
        return

    if backend != "transformers":
        raise ValueError(f"Unsupported backend for finetuning: {backend}")

    if train_splits is None:
        train_splits = ["train"]
    if dev_splits is None:
        dev_splits = ["dev"]

    device = get_device()
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    QWEN_LORA_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    llm_model, tokenizer = load_language_model(
        backend="transformers",
        device=device,
        model_name=model_name,
    )
    llm_model, lora_layers = _configure_qwen_lora_last_layers(
        llm_model,
        layers_to_finetune=layers_to_finetune,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    llm_model.train()

    if hasattr(llm_model, "print_trainable_parameters"):
        llm_model.print_trainable_parameters()
    print(f"LoRA enabled on transformer layers: {lora_layers}")

    train_datasets = [
        SentenceSplitDataset(
            split=s,
            tokenizer=tokenizer,
            max_chars=max_chars,
            stride_chars=stride_chars,
            augment_prob=0.0,
            augmentation_mode="original",
        )
        for s in train_splits
    ]
    if augment_prob > 0:
        train_datasets.extend(
            [
                SentenceSplitDataset(
                    split=s,
                    tokenizer=tokenizer,
                    max_chars=max_chars,
                    stride_chars=stride_chars,
                    augment_prob=augment_prob,
                    augmentation_mode="augmented",
                )
                for s in train_splits
            ]
        )

    dev_datasets = [
        SentenceSplitDataset(
            split=s,
            tokenizer=tokenizer,
            max_chars=max_chars,
            stride_chars=stride_chars,
            augment_prob=0.0,
            augmentation_mode="original",
        )
        for s in dev_splits
    ]

    train_ds = ConcatDataset(train_datasets)
    dev_ds = ConcatDataset(dev_datasets)

    hidden_dim = getattr(llm_model.config, "hidden_size", None)
    if hidden_dim is None:
        raise ValueError("Unable to infer hidden_size from Qwen config")

    mlp = SpacePredictorMLP(hidden_dim=hidden_dim, d_model=d_model, dropout=dropout).to(device)

    if balanced_batches:
        sample_weights = _build_balanced_sample_weights(train_datasets)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_sentence_fn,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_sentence_fn,
            num_workers=0,
        )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sentence_fn,
        num_workers=0,
    )

    criterion = FocalLoss(alpha=pos_weight, gamma=2.0, reduction="none")
    optimizer = torch.optim.Adam(
        list(mlp.parameters()) + [p for p in llm_model.parameters() if p.requires_grad],
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        mlp.train()
        llm_model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"LoRA Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_mask = batch["token_mask"].to(device)
            labels = batch["token_labels"].to(device)

            tok_emb = extract_token_embeddings(
                llm_model,
                input_ids,
                attention_mask,
                backend="transformers",
            )
            preds, moe_aux_loss = mlp(tok_emb, mask=token_mask)

            loss_all = criterion(preds, labels)
            bce_loss = (loss_all * token_mask.float()).sum() / max(token_mask.float().sum(), 1.0)
            loss_total = bce_loss + aux_weight * moe_aux_loss

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([p for p in llm_model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()

            total_loss += loss_total.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss_total.item():.4f}", "aux": f"{moe_aux_loss.item():.3f}"})

        avg_loss = total_loss / max(num_batches, 1)
        metrics = evaluate_end_to_end(mlp, llm_model, dev_loader, device)
        scheduler.step(metrics["f1"])

        print(
            f"LoRA Epoch {epoch:2d}/{epochs} │ "
            f"Loss: {avg_loss:.4f} │ "
            f"Acc: {metrics['accuracy']:.4f} │ "
            f"P: {metrics['precision']:.4f} │ "
            f"R: {metrics['recall']:.4f} │ "
            f"F1: {metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_counter = 0
            llm_model.save_pretrained(QWEN_LORA_ADAPTER_DIR)
            torch.save(
                {
                    "model_state_dict": mlp.state_dict(),
                    "hidden_dim": hidden_dim,
                    "d_model": mlp.cnn_dim,
                    "cnn_dim": mlp.cnn_dim,
                    "dropout": dropout,
                    "num_experts": mlp.num_experts,
                    "top_k": mlp.top_k,
                    "aux_weight": aux_weight,
                    "epoch": epoch,
                    "f1": best_f1,
                    "lora_adapter_dir": str(QWEN_LORA_ADAPTER_DIR),
                    "lora_layers": lora_layers,
                    "base_model_name": model_name,
                },
                BEST_SENTENCE_LORA_CKPT,
            )
            print(f"  → Saved best LoRA adapter + MLP checkpoint (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ LoRA early stopping at epoch {epoch}")
                break


def finetune_qwen_lora_cached_sentence_splitter(
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    d_model: int = 256,
    dropout: float = 0.2,
    pos_weight: float = 10.0,
    patience: int = 3,
    aux_weight: float = 0.01,
    train_splits: list[str] | None = None,
    dev_splits: list[str] | None = None,
    max_chars: int = 1024,
    stride_chars: int = 512,
    augment_prob: float = 0.0,
    balanced_batches: bool = True,
    model_name: str = MODEL_NAME,
    layers_to_finetune: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
    backend: str = "transformers",
):
    set_deterministic_seed(seed)

    if backend == "mlx":
        if train_splits is None:
            train_splits = ["train"]
        if dev_splits is None:
            dev_splits = ["dev"]

        _run_mlx_lora_finetune(
            model_name=model_name,
            train_splits=train_splits,
            dev_splits=dev_splits,
            layers_to_finetune=layers_to_finetune,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            max_chars=max_chars,
            seed=seed,
            adapter_dir=QWEN_MLX_LORA_ADAPTER_DIR,
        )

        target_splits = sorted(set(train_splits + dev_splits))
        prefix_cutoff, prefix_cache_root = _extract_sentence_embeddings_mlx_cached_last_layers(
            batch_size=batch_size,
            model_name=model_name,
            adapter_path=str(QWEN_MLX_LORA_ADAPTER_DIR),
            splits_to_extract=target_splits,
            layers_to_finetune=layers_to_finetune,
            max_chars=max_chars,
            stride_chars=stride_chars,
            augment_prob=augment_prob,
        )

        train_sentence_mlp(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            d_model=d_model,
            dropout=dropout,
            pos_weight=pos_weight,
            patience=patience,
            aux_weight=aux_weight,
            train_splits=train_splits,
            dev_splits=dev_splits,
            augment_prob=augment_prob,
            balanced_batches=balanced_batches,
            seed=seed,
        )

        if BEST_SENTENCE_LORA_CKPT.exists():
            checkpoint = torch.load(BEST_SENTENCE_LORA_CKPT, map_location="cpu", weights_only=False)
            checkpoint["lora_adapter_dir"] = str(QWEN_MLX_LORA_ADAPTER_DIR)
            checkpoint["requested_backend"] = "mlx"
            checkpoint["requested_model_name"] = model_name
            checkpoint["prefix_cutoff"] = prefix_cutoff
            checkpoint["prefix_cache_dir"] = str(prefix_cache_root)
            torch.save(checkpoint, BEST_SENTENCE_LORA_CKPT)

        print("MLX cached LoRA pipeline complete: prefix N-2 cached, tail-only MLX inference materialized, MLP trained.")
        return

    requested_backend = backend
    training_backend, resolved_model_name = _resolve_transformers_model_name_for_cached_lora(
        backend=backend,
        model_name=model_name,
    )
    adapter_dir = QWEN_MLX_LORA_ADAPTER_DIR if requested_backend == "mlx" else QWEN_LORA_ADAPTER_DIR

    if train_splits is None:
        train_splits = ["train"]
    if dev_splits is None:
        dev_splits = ["dev"]

    device = get_device()
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    llm_model, tokenizer = load_language_model(
        backend=training_backend,
        device=device,
        model_name=resolved_model_name,
    )

    layers = _get_transformer_layers(llm_model)
    prefix_cutoff = len(layers) - layers_to_finetune
    if prefix_cutoff <= 0:
        raise ValueError("layers_to_finetune must be smaller than total transformer layers")

    llm_model, lora_layers = _configure_qwen_lora_last_layers(
        llm_model,
        layers_to_finetune=layers_to_finetune,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # Cache hidden states up to layer N-2, loading datasets one at a time.
    cache_tag = f"last{layers_to_finetune}_{model_name.split('/')[-1]}"
    train_cache_dirs = []
    
    # Process train splits one at a time
    for split_idx, split in enumerate(train_splits):
        ds = SentenceSplitDataset(
            split=split,
            tokenizer=tokenizer,
            max_chars=max_chars,
            stride_chars=stride_chars,
            augment_prob=0.0,
            augmentation_mode="original",
        )
        raw_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sentence_fn, num_workers=0)
        cache_name = f"train_{split_idx}_{cache_tag}"
        train_cache_dirs.append(
            extract_and_cache_prefix_hidden_states(
                llm_model,
                raw_loader,
                device=device,
                cache_name=cache_name,
                prefix_layer_idx=prefix_cutoff - 1,
                base_cache_dir=PREFIX_CACHE_DIR,
            )
        )
        del ds
        
        # Process augmented version if requested
        if augment_prob > 0:
            ds_aug = SentenceSplitDataset(
                split=split,
                tokenizer=tokenizer,
                max_chars=max_chars,
                stride_chars=stride_chars,
                augment_prob=augment_prob,
                augmentation_mode="augmented",
            )
            raw_loader_aug = DataLoader(ds_aug, batch_size=batch_size, shuffle=False, collate_fn=collate_sentence_fn, num_workers=0)
            cache_name_aug = f"train_{split_idx}_aug_{cache_tag}"
            train_cache_dirs.append(
                extract_and_cache_prefix_hidden_states(
                    llm_model,
                    raw_loader_aug,
                    device=device,
                    cache_name=cache_name_aug,
                    prefix_layer_idx=prefix_cutoff - 1,
                    base_cache_dir=PREFIX_CACHE_DIR,
                )
            )
            del ds_aug

    # Process dev splits one at a time
    dev_cache_dirs = []
    for split_idx, split in enumerate(dev_splits):
        ds = SentenceSplitDataset(
            split=split,
            tokenizer=tokenizer,
            max_chars=max_chars,
            stride_chars=stride_chars,
            augment_prob=0.0,
            augmentation_mode="original",
        )
        raw_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sentence_fn, num_workers=0)
        cache_name = f"dev_{split_idx}_{cache_tag}"
        dev_cache_dirs.append(
            extract_and_cache_prefix_hidden_states(
                llm_model,
                raw_loader,
                device=device,
                cache_name=cache_name,
                prefix_layer_idx=prefix_cutoff - 1,
                base_cache_dir=PREFIX_CACHE_DIR,
            )
        )
        del ds

    _replace_lower_layers_with_passthrough(llm_model, layers_to_finetune=layers_to_finetune)
    llm_model.train()

    if hasattr(llm_model, "print_trainable_parameters"):
        llm_model.print_trainable_parameters()
    print(f"LoRA cached mode enabled on transformer layers: {lora_layers} (prefix cutoff={prefix_cutoff})")

    hidden_dim = getattr(llm_model.config, "hidden_size", None)
    if hidden_dim is None:
        raise ValueError("Unable to infer hidden_size from Qwen config")

    mlp = SpacePredictorMLP(hidden_dim=hidden_dim, d_model=d_model, dropout=dropout).to(device)

    if balanced_batches:
        from torch.utils.data import IterableDataset
        train_ds = BalancedStreamDataset(train_cache_dirs, batch_size)
        if train_ds.actual_batch_size != batch_size:
            print(f"Strict balanced batching enabled: adjusted batch_size from {batch_size} to {train_ds.actual_batch_size} to perfectly fit {len(train_cache_dirs)} datasets.")
        train_loader = DataLoader(
            train_ds,
            batch_size=train_ds.actual_batch_size,
            collate_fn=collate_prefix_hidden_fn,
            num_workers=0,
        )
    else:
        train_datasets = [PrefixHiddenDataset(p) for p in train_cache_dirs]
        train_ds = ConcatDataset(train_datasets)
        # Fallback to random loader for un-balanced (note: might hit I/O bottleneck if un-balanced is used)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_prefix_hidden_fn,
            num_workers=0,
        )

    dev_datasets = [PrefixHiddenDataset(p) for p in dev_cache_dirs]
    dev_ds = ConcatDataset(dev_datasets)
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_prefix_hidden_fn,
        num_workers=0,
    )

    criterion = FocalLoss(alpha=pos_weight, gamma=2.0, reduction="none")
    optimizer = torch.optim.Adam(
        list(mlp.parameters()) + [p for p in llm_model.parameters() if p.requires_grad],
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        mlp.train()
        llm_model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"LoRA Cached Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            prefix_hidden = batch["token_embeddings"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_mask = batch["token_mask"].to(device)
            labels = batch["token_labels"].to(device)

            hidden_states = forward_transformer_hidden_states_only(
                model=llm_model,
                inputs_embeds=prefix_hidden,
                attention_mask=attention_mask,
            )
            tok_emb = hidden_states[-1].float()
            preds, moe_aux_loss = mlp(tok_emb, mask=token_mask)

            loss_all = criterion(preds, labels)
            bce_loss = (loss_all * token_mask.float()).sum() / max(token_mask.float().sum(), 1.0)
            loss_total = bce_loss + aux_weight * moe_aux_loss

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([p for p in llm_model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()

            total_loss += loss_total.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss_total.item():.4f}", "aux": f"{moe_aux_loss.item():.3f}"})

        avg_loss = total_loss / max(num_batches, 1)

        metrics = evaluate_cached_prefix_end_to_end(mlp, llm_model, dev_loader, device)
        scheduler.step(metrics["f1"])

        print(
            f"LoRA Cached Epoch {epoch:2d}/{epochs} │ "
            f"Loss: {avg_loss:.4f} │ "
            f"Acc: {metrics['accuracy']:.4f} │ "
            f"P: {metrics['precision']:.4f} │ "
            f"R: {metrics['recall']:.4f} │ "
            f"F1: {metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_counter = 0
            llm_model.save_pretrained(adapter_dir)
            torch.save(
                {
                    "model_state_dict": mlp.state_dict(),
                    "hidden_dim": hidden_dim,
                    "d_model": mlp.cnn_dim,
                    "cnn_dim": mlp.cnn_dim,
                    "dropout": dropout,
                    "num_experts": mlp.num_experts,
                    "top_k": mlp.top_k,
                    "aux_weight": aux_weight,
                    "epoch": epoch,
                    "f1": best_f1,
                    "lora_adapter_dir": str(adapter_dir),
                    "lora_layers": lora_layers,
                    "base_model_name": resolved_model_name,
                    "requested_backend": requested_backend,
                    "requested_model_name": model_name,
                    "prefix_cutoff": prefix_cutoff,
                    "prefix_cache_dir": str(PREFIX_CACHE_DIR),
                },
                BEST_SENTENCE_LORA_CKPT,
            )
            print(f"  → Saved best LoRA adapter + MLP checkpoint (cached mode, F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ LoRA cached early stopping at epoch {epoch}")
                break


def extract_sentence_embeddings(
    batch_size: int = 8,
    backend: str = "transformers",
    augment_prob: float = 0.0,
    max_chars: int = 2048,
    stride_chars: int = 1024,
    seed: int = 42,
    adapter_path: str | None = None,
    splits_to_extract: list[str] | None = None,
    model_name: str | None = None,
):
    """Extract LLM embeddings for sentence documents and cache them."""
    set_deterministic_seed(seed)

    device = get_device()
    resolved_model_name = model_name or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)

    target_splits = splits_to_extract or list(UD_URLS.keys())

    for split in target_splits:
        print(f"\n{'='*60}")
        print(f"Extracting sentence embeddings for {split} split...")
        print(f"{'='*60}")

        # Load model for this split only
        model, _ = load_language_model(backend, device, model_name=resolved_model_name, adapter_path=adapter_path)

        # 1. Original
        loader = get_sentence_dataloader(
            split=split,
            batch_size=batch_size,
            tokenizer=tokenizer,
            shuffle=False,
            max_chars=max_chars,
            stride_chars=stride_chars,
            augment_prob=0.0,
            augmentation_mode="original"
        )
        extract_and_cache_embeddings(
            model=model,
            dataloader=loader,
            device=device,
            cache_name=split,
            backend=backend,
            base_cache_dir=SENTENCE_CACHE_DIR
        )

        # 2. Augmented (only for train split usually, but let's follow logic)
        if augment_prob > 0:
            print(f"Creating augmented dataset for {split}...")
            loader_aug = get_sentence_dataloader(
                split=split,
                batch_size=batch_size,
                tokenizer=tokenizer,
                shuffle=False,
                max_chars=max_chars,
                stride_chars=stride_chars,
                augment_prob=augment_prob,
                augmentation_mode="augmented"
            )
            extract_and_cache_embeddings(
                model=model,
                dataloader=loader_aug,
                device=device,
                cache_name=f"{split}_aug",
                backend=backend,
                base_cache_dir=SENTENCE_CACHE_DIR
            )

        # Free memory after each split
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n✓ All sentence embeddings extracted and cached.")


def train_sentence_mlp(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    d_model: int = 256,
    dropout: float = 0.2,
    pos_weight: float = 10.0, # Sentence boundaries are much rarer than word boundaries
    patience: int = 7,
    aux_weight: float = 0.01,
    train_splits: list[str] = None,
    dev_splits: list[str] = None,
    augment_prob: float = 0.0,
    balanced_batches: bool = True,
    seed: int = 42,
):
    set_deterministic_seed(seed)

    if train_splits is None:
        train_splits = ["train"]
    if dev_splits is None:
        dev_splits = ["dev"]
        
    device = get_device()
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Load cached embeddings from the sentence cache
    print(f"Loading cached sentence embeddings... Train: {train_splits}, Dev: {dev_splits}")
    
    train_datasets = []
    for s in train_splits:
        train_datasets.append(CachedEmbeddingDataset(SENTENCE_CACHE_DIR / s))
        # Automatically load augmented dataset if it exists
        aug_path = SENTENCE_CACHE_DIR / f"{s}_aug"
        if aug_path.exists():
            print(f"  → Also loading augmented samples from {aug_path}")
            train_datasets.append(CachedEmbeddingDataset(aug_path))
            
    train_ds = ConcatDataset(train_datasets)
    dev_ds = ConcatDataset([CachedEmbeddingDataset(SENTENCE_CACHE_DIR / s) for s in dev_splits])
    
    hidden_dim = train_ds[0]["token_embeddings"].shape[-1]
    print(f"Detected model hidden_dim: {hidden_dim}")

    if balanced_batches:
        sample_weights = _build_balanced_sample_weights(train_datasets)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=cached_collate_fn,
            num_workers=4,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=cached_collate_fn,
            num_workers=4,
        )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=cached_collate_fn,
        num_workers=2,
    )

    print(f"Train samples: {len(train_ds)}, Dev samples: {len(dev_ds)}")
    print(f"Balanced datasets: {balanced_batches}")
    print(f"MoE aux_weight: {aux_weight}")
    print(f"Model d_model: {d_model}")

    # Model
    mlp = SpacePredictorMLP(hidden_dim=hidden_dim, d_model=d_model, dropout=dropout).to(device)

    criterion = FocalLoss(alpha=pos_weight, gamma=2.0, reduction="none")
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        mlp.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            emb = batch["token_embeddings"].to(device)
            labels = batch["token_labels"].to(device)
            mask = batch["token_mask"].to(device)

            preds, moe_aux_loss = mlp(emb, mask=mask) 

            loss_all = criterion(preds, labels)
            bce_loss = (loss_all * mask.float()).sum() / max(mask.float().sum(), 1.0)
            
            # Total loss = BCE + load-balancing auxiliary loss
            loss_total = bce_loss + aux_weight * moe_aux_loss

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_total.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss_total.item():.4f}", "aux": f"{moe_aux_loss.item():.3f}"})

        avg_loss = total_loss / max(num_batches, 1)
        metrics = evaluate(mlp, dev_loader, device)
        scheduler.step(metrics["f1"])

        print(
            f"Epoch {epoch:2d}/{epochs} │ "
            f"Loss: {avg_loss:.4f} │ "
            f"Acc: {metrics['accuracy']:.4f} │ "
            f"P: {metrics['precision']:.4f} │ "
            f"R: {metrics['recall']:.4f} │ "
            f"F1: {metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": mlp.state_dict(),
                    "hidden_dim": hidden_dim,
                    "d_model": mlp.cnn_dim,
                    "cnn_dim": mlp.cnn_dim,
                    "dropout": dropout,
                    "num_experts": mlp.num_experts,
                    "top_k": mlp.top_k,
                    "aux_weight": aux_weight,
                    "epoch": epoch,
                    "f1": best_f1,
                },
                BEST_SENTENCE_CKPT,
            )
            print(f"  → Saved best checkpoint (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                break

    print(f"\n✓ Training complete. Best F1: {best_f1:.4f}")
