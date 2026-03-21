"""
Training pipeline for Sentence Splitter MLP.

Phase 1: Extract LLM embeddings offline for sentence chunks and cache.
Phase 2: Train the MLP on cached sentence-level embeddings.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from data_sentence import get_sentence_dataloader
from wordSplitter.train import CachedEmbeddingDataset, cached_collate_fn, evaluate
from wordSplitter.embeddings import (
    load_language_model,
    extract_and_cache_embeddings,
    get_device,
)
from wordSplitter.model import SpacePredictorMLP, FocalLoss
from wordSplitter.data import UD_URLS

SENTENCE_CACHE_DIR = Path(__file__).parent / "sentence_embedding_cache"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BEST_SENTENCE_CKPT = CHECKPOINT_DIR / "best_sentence_mlp.pt"


def extract_sentence_embeddings(batch_size: int = 8, backend: str = "transformers", augment_prob: float = 0.0):
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
            max_chars=2048,
            chunk_size=10,
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
                max_chars=2048,
                chunk_size=10,
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
    dropout: float = 0.2,
    pos_weight: float = 10.0, # Sentence boundaries are much rarer than word boundaries
    patience: int = 7,
    train_splits: list[str] = None,
    dev_splits: list[str] = None,
    augment_prob: float = 0.0,
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
    
    hidden_dim = train_ds[0]["char_embeddings"].shape[-1]
    print(f"Detected model hidden_dim: {hidden_dim}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=cached_collate_fn,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=cached_collate_fn,
    )

    print(f"Train samples: {len(train_ds)}, Dev samples: {len(dev_ds)}")

    # Model
    mlp = SpacePredictorMLP(hidden_dim=hidden_dim, dropout=dropout).to(device)

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

        for batch in train_loader:
            emb = batch["char_embeddings"].to(device)
            labels = batch["char_labels"].to(device)
            mask = batch["char_mask"].to(device)

            preds = mlp(emb) 

            loss_all = criterion(preds, labels)
            loss_masked = (loss_all * mask.float()).sum() / max(mask.float().sum(), 1.0)

            optimizer.zero_grad()
            loss_masked.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_masked.item()
            num_batches += 1

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
                    "dropout": dropout,
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
