"""
Training pipeline for the Word Splitter MLP.

Phase 1: Extract Minerva embeddings offline and cache to disk.
Phase 2: Train the MLP on cached embeddings.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data import get_dataloader, MODEL_NAME, WordSplitDataset, collate_fn
from embeddings import (
    load_language_model,
    extract_and_cache_embeddings,
    get_device,
    CACHE_DIR,
)
from model import SpacePredictorMLP

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


# ── Cached Embedding Dataset ─────────────────────────────────────────────────

class CachedEmbeddingDataset(Dataset):
    """Loads pre-extracted character embeddings from disk."""

    def __init__(self, cache_path: Path):
        self.files = sorted(cache_path.glob("batch_*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No cached embeddings found in {cache_path}. "
                "Run Phase 1 (extract_embeddings) first."
            )

        # Flatten all batches into individual samples
        self.samples = []
        for f in self.files:
            data = torch.load(f, weights_only=True)
            batch_size = data["char_embeddings"].shape[0]
            for i in range(batch_size):
                mask = data["char_mask"][i]
                sample = {
                    "char_embeddings": data["char_embeddings"][i][mask],
                    "char_labels": data["char_labels"][i][mask],
                }
                if "spaceless" in data:
                    sample["spaceless"] = data["spaceless"][i]
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def cached_collate_fn(batch):
    """Pad cached embeddings and labels."""
    max_len = max(s["char_embeddings"].shape[0] for s in batch)
    hidden_dim = batch[0]["char_embeddings"].shape[-1]

    embeddings = torch.zeros(len(batch), max_len, hidden_dim)
    labels = torch.full((len(batch), max_len), -1.0)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, s in enumerate(batch):
        length = s["char_embeddings"].shape[0]
        embeddings[i, :length] = s["char_embeddings"]
        labels[i, :length] = s["char_labels"]
        mask[i, :length] = True

    return {
        "char_embeddings": embeddings,
        "char_labels": labels,
        "char_mask": mask,
        "spaceless": [s.get("spaceless", "") for s in batch],
    }


# ── Phase 1: Extract embeddings ──────────────────────────────────────────────

def extract_embeddings(batch_size: int = 16, backend: str = "transformers"):
    """Extract Minerva embeddings for all splits and cache to disk."""
    device = get_device()
    model, tokenizer = load_language_model(backend, device)

    from data import UD_URLS
    splits_to_extract = list(UD_URLS.keys())

    for split in splits_to_extract:
        print(f"\n{'='*60}")
        print(f"Extracting embeddings for {split} split...")
        print(f"{'='*60}")

        loader = get_dataloader(
            split=split,
            batch_size=batch_size,
            tokenizer=tokenizer,
            shuffle=False,
        )

        extract_and_cache_embeddings(
            model=model,
            dataloader=loader,
            device=device,
            cache_name=split,
            backend=backend,
        )

    # Free memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("\n✓ All embeddings extracted and cached.")


# ── Phase 2: Train MLP ───────────────────────────────────────────────────────

def train_mlp(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout: float = 0.2,
    pos_weight: float = 2.0,
    patience: int = 7,
    train_splits: list[str] = None,
    dev_splits: list[str] = None,
):
    if train_splits is None:
        train_splits = ["train"]
    if dev_splits is None:
        dev_splits = ["dev"]
    """Train the space-prediction MLP on cached embeddings."""
    device = get_device()
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Load cached embeddings
    print(f"Loading cached embeddings... Train: {train_splits}, Dev: {dev_splits}")
    train_ds = ConcatDataset([CachedEmbeddingDataset(CACHE_DIR / s) for s in train_splits])
    dev_ds = ConcatDataset([CachedEmbeddingDataset(CACHE_DIR / s) for s in dev_splits])
    
    # Auto-detect hidden dimension from extracted embeddings
    hidden_dim = train_ds[0]["char_embeddings"].shape[-1]
    print(f"Detected model hidden_dim: {hidden_dim}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=cached_collate_fn,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=cached_collate_fn,
        num_workers=0,
    )

    print(f"Train samples: {len(train_ds)}, Dev samples: {len(dev_ds)}")

    # Model
    mlp = SpacePredictorMLP(hidden_dim=hidden_dim, dropout=dropout).to(device)

    # Loss with positive class weighting (spaces are minority)
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────
        mlp.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            emb = batch["char_embeddings"].to(device)
            labels = batch["char_labels"].to(device)
            mask = batch["char_mask"].to(device)

            preds = mlp(emb)  # (batch, seq_len)

            # Compute loss only on valid positions
            loss_all = criterion(preds, labels)

            # Apply positive weighting
            weight = torch.where(labels == 1, pos_weight, 1.0)
            loss_masked = (loss_all * weight * mask.float()).sum() / mask.float().sum()

            optimizer.zero_grad()
            loss_masked.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_masked.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # ── Evaluate ──────────────────────────────────────────────
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

        # Save best checkpoint
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_counter = 0
            ckpt_path = CHECKPOINT_DIR / "best_mlp.pt"
            torch.save(
                {
                    "model_state_dict": mlp.state_dict(),
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "epoch": epoch,
                    "f1": best_f1,
                },
                ckpt_path,
            )
            print(f"  → Saved best checkpoint (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
                break

    print(f"\n✓ Training complete. Best F1: {best_f1:.4f}")


@torch.no_grad()
def evaluate(
    model: SpacePredictorMLP,
    dataloader: DataLoader,
    device: torch.device,
    top_k_errors: int = 0,
) -> dict:
    """Evaluate the MLP on a dataset, return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    exact_matches = 0
    total_samples = 0
    error_details = []  # List of (error_count, text, pred_text, true_text)

    for batch in dataloader:
        emb = batch["char_embeddings"].to(device)
        labels = batch["char_labels"]
        mask = batch["char_mask"]

        preds = model(emb).cpu()
        texts = batch.get("spaceless", [""] * preds.shape[0])

        # Collect valid predictions
        for i in range(preds.shape[0]):
            valid = mask[i]
            p = (preds[i][valid] > 0.5).int()
            l = labels[i][valid].int()
            
            p_list = p.tolist()
            l_list = l.tolist()
            all_preds.extend(p_list)
            all_labels.extend(l_list)
            
            error_count = (p != l).sum().item()
            if torch.equal(p, l):
                exact_matches += 1
            elif top_k_errors > 0:
                # Helper to reconstruct text from space mask
                def reconstruct(base, mask):
                    res = []
                    for char, m in zip(base, mask):
                        res.append(char)
                        if m == 0: res.append(" ")
                    return "".join(res)
                
                error_details.append({
                    "errors": error_count,
                    "text": texts[i],
                    "pred": reconstruct(texts[i], p_list),
                    "true": reconstruct(texts[i], l_list)
                })
            total_samples += 1

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    em = exact_matches / total_samples if total_samples > 0 else 0.0

    # Sort error details by number of errors descending
    worst_samples = sorted(error_details, key=lambda x: x["errors"], reverse=True)[:top_k_errors]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": em,
        "worst_samples": worst_samples,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Word Splitter MLP")
    parser.add_argument(
        "--phase",
        choices=["extract", "train", "both"],
        default="both",
        help="Which phase to run: extract embeddings, train MLP, or both",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pos-weight", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument(
        "--extract-batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction (lower if OOM)",
    )

    args = parser.parse_args()

    if args.phase in ("extract", "both"):
        extract_embeddings(batch_size=args.extract_batch_size)

    if args.phase in ("train", "both"):
        train_mlp(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            pos_weight=args.pos_weight,
            patience=args.patience,
        )
