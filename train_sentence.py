"""
Training pipeline for Sentence Splitter MLP.

Phase 1: Extract LLM embeddings offline for sentence chunks and cache.
Phase 2: Train the MLP on cached sentence-level embeddings.
"""

import argparse
import functools
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data_sentence import get_sentence_dataloader, UD_URLS
from sentence_embeddings import (
    load_language_model,
    extract_and_cache_embeddings,
    get_device,
)
from model import SpacePredictorMLP, FocalLoss

SENTENCE_CACHE_DIR = Path(__file__).parent / "sentence_embedding_cache"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BEST_SENTENCE_CKPT = CHECKPOINT_DIR / "best_sentence_mlp.pt"


@functools.lru_cache(maxsize=128)
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


def extract_sentence_embeddings(batch_size: int = 8, backend: str = "transformers", augment_prob: float = 0.0, max_chars: int = 2048, stride_chars: int = 1024):
    """Extract LLM embeddings for sentence documents and cache them."""
    device = get_device()
    model, tokenizer = load_language_model(backend, device)

    splits_to_extract = list(UD_URLS.keys())

    for split in splits_to_extract:
        print(f"\n{'='*60}")
        print(f"Extracting sentence embeddings for {split} split...")
        print(f"{'='*60}")

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

    # Free memory
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
):
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
