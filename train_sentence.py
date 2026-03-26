"""
Training pipeline for Sentence Splitter MLP (token-level).

Phase 1: Extract LLM token embeddings offline and cache.
Phase 2: Train the MLP on cached token-level embeddings.
"""

import argparse
import functools
from pathlib import Path
from typing import Optional
from typing import Optional, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np # Added numpy
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from data_sentence import get_sentence_dataloader
from wordSplitter.embeddings import (
    load_language_model,
    extract_and_cache_token_embeddings,
    get_device,
    MODEL_NAME, # Added MODEL_NAME
)
from wordSplitter.model import SpacePredictorMLP, FineTuneSentenceSplitter, FocalLoss 
from wordSplitter.data import UD_URLS
try:
    import mlx.core as mx
    import mlx.nn as mnn
    import mlx.optimizers as mopt
    from wordSplitter.model_mlx import SpacePredictorMLP as SpacePredictorMLP_MLX
    from wordSplitter.model_mlx import FineTuneSentenceSplitterMLX, apply_lora_to_module
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

SENTENCE_CACHE_DIR = Path(__file__).parent / "sentence_token_cache"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BEST_SENTENCE_CKPT = CHECKPOINT_DIR / "best_sentence_mlp.pt"


# ── Cached Token Embedding Dataset ───────────────────────────────────────────

@functools.lru_cache(maxsize=128)
def _load_batch_file(file_path: Path):
    return torch.load(file_path, weights_only=True, mmap=True)


class CachedTokenEmbeddingDataset(Dataset):
    """Loads pre-extracted token-level embeddings from disk lazily."""

    def __init__(self, cache_path: Path):
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache path {cache_path} does not exist.")
        self.files = sorted(cache_path.glob("batch_*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No cached embeddings found in {cache_path}. "
                "Run Phase 1 (extract) first."
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
        return {
            "token_embeddings": data["token_embeddings"][inner_idx][mask],
            "token_labels": data["token_labels"][inner_idx][mask],
            "token_offsets": data["token_offsets"][inner_idx][mask] if "token_offsets" in data else None,
            "text": data["text"][inner_idx] if "text" in data else "",
        }


def cached_token_collate_fn(batch):
    """Pad cached token embeddings and labels."""
    max_len = max(s["token_embeddings"].shape[0] for s in batch)
    hidden_dim = batch[0]["token_embeddings"].shape[-1]

    embeddings = torch.zeros(len(batch), max_len, hidden_dim)
    labels = torch.full((len(batch), max_len), -1.0)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    offsets = torch.zeros(len(batch), max_len, 2, dtype=torch.long)
    texts = [s.get("text", "") for s in batch]

    for i, s in enumerate(batch):
        length = s["token_embeddings"].shape[0]
        embeddings[i, :length] = s["token_embeddings"]
        labels[i, :length] = s["token_labels"]
        mask[i, :length] = True
        if s["token_offsets"] is not None:
            offsets[i, :length] = s["token_offsets"]

    return {
        "token_embeddings": embeddings,
        "token_labels": labels,
        "token_mask": mask,
        "token_offsets": offsets,
        "texts": texts,
    }


def extract_sentence_embeddings(batch_size: int = 8, backend: str = "transformers", augment_prob: float = 0.0, max_chars: int = 2048, stride_chars: int = 1024, layer_idx: Optional[int] = None):
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
        extract_and_cache_token_embeddings(
            model=model,
            dataloader=loader,
            device=device,
            cache_name=split,
            backend=backend,
            base_cache_dir=SENTENCE_CACHE_DIR,
            layer_idx=layer_idx
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
            extract_and_cache_token_embeddings(
                model=model,
                dataloader=loader_aug,
                device=device,
                cache_name=f"{split}_aug",
                backend=backend,
                base_cache_dir=SENTENCE_CACHE_DIR,
                layer_idx=layer_idx
            )

    # Free memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("\n✓ All sentence token embeddings extracted and cached.")


def train_sentence_mlp(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout: float = 0.2,
    pos_weight: float = 10.0, # Sentence boundaries are much rarer than word boundaries
    patience: int = 7,
    aux_weight: float = 0.01,
    label_smoothing: float = 0.05,
    dev_splits: list[str] = None,
    d_model: int = 256,
    fine_tune_layers: int = 0,
    layer_idx: Optional[int] = None,
    backend: str = "transformers",
):
    if train_splits is None:
        train_splits = ["train"]
    if dev_splits is None:
        dev_splits = ["dev"]
        
    device = get_device()
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Load cached token embeddings
    print(f"Loading cached token embeddings... Train: {train_splits}, Dev: {dev_splits}")
    
    train_datasets = []
    suffix = "" if layer_idx is None else f"_L{layer_idx}"
    
    for s in train_splits:
        train_datasets.append(CachedTokenEmbeddingDataset(SENTENCE_CACHE_DIR / f"{s}{suffix}"))
        # Automatically load augmented dataset if it exists
        aug_path = SENTENCE_CACHE_DIR / f"{s}_aug{suffix}"
        if aug_path.exists():
            print(f"  → Also loading augmented samples from {aug_path}")
            train_datasets.append(CachedTokenEmbeddingDataset(aug_path))
            
    train_ds = ConcatDataset(train_datasets)
    dev_ds = ConcatDataset([CachedTokenEmbeddingDataset(SENTENCE_CACHE_DIR / f"{s}{suffix}") for s in dev_splits])
    
    hidden_dim = train_ds[0]["token_embeddings"].shape[-1]
    
    # Balanced Sampler for multi-dataset training
    dataset_sizes = [len(ds) for ds in train_ds.datasets]
    dataset_weights = [1.0 / s for s in dataset_sizes]
    
    sample_weights = []
    for i, size in enumerate(dataset_sizes):
        sample_weights.extend([dataset_weights[i]] * size)
        
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=cached_token_collate_fn,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=cached_token_collate_fn,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train samples: {len(train_ds)}, Dev samples: {len(dev_ds)}")
    print(f"MoE aux_weight: {aux_weight}")

    # Model
    mlp = SpacePredictorMLP(hidden_dim=hidden_dim, d_model=d_model, dropout=dropout)
    
    if fine_tune_layers > 0:
        print(f"Fine-tuning enabled: last {fine_tune_layers} layers of {MODEL_NAME}")
        llm_full, _ = load_language_model(backend, device)
        
        # Extract the last N layers
        # Qwen2/Qwen3 structure: llm_full.model.layers
        if hasattr(llm_full, "model") and hasattr(llm_full.model, "layers"):
            all_layers = llm_full.model.layers
            fine_tune_start = len(all_layers) - fine_tune_layers
            target_layers = nn.ModuleList([all_layers[i] for i in range(fine_tune_start, len(all_layers))])
            
            # Freeze all other parameters in llm_full if we were using it for more, 
            # but here target_layers are already part of it.
            # We wrap them into FineTuneSentenceSplitter.
            model = FineTuneSentenceSplitter(target_layers, mlp).to(device)
            # Ensure the specific layers are trainable
            for param in model.transformer_layers.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Could not find transformer layers in model of type {type(llm_full)}")
    else:
        model = mlp.to(device)

    criterion = FocalLoss(alpha=pos_weight, gamma=2.0, reduction="none", label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # AMP Scaler (Modern API)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    # For MPS, AMP (autocast) is available but GradScaler is typically not needed for 16-bit
    # although we'll use it if CUDA is present for safety/performance.

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            emb = batch["token_embeddings"].to(device, non_blocking=True)
            labels = batch["token_labels"].to(device, non_blocking=True)
            mask = batch["token_mask"].to(device, non_blocking=True)

            # Mixed Precision Forward
            with torch.autocast(device_type=device.type, enabled=(device.type in ["cuda", "mps"])):
                preds, moe_aux_loss = model(emb, mask=mask) 

                loss_all = criterion(preds, labels)
                bce_loss = (loss_all * mask.float()).sum() / max(mask.float().sum(), 1.0)
                
                # Total loss = BCE + load-balancing auxiliary loss
                loss_total = bce_loss + aux_weight * moe_aux_loss

            optimizer.zero_grad()
            
            if device.type == "cuda":
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss_total.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss_total.item():.4f}", "aux": f"{moe_aux_loss.item():.3f}"})

        avg_loss = total_loss / max(num_batches, 1)
        metrics = evaluate_tokens(model, dev_loader, device)
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
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), # Added optimizer state
                    "hidden_dim": hidden_dim,
                    "d_model": d_model, # Added d_model
                    "dropout": dropout,
                    "fine_tune_layers": fine_tune_layers,
                    "layer_idx": layer_idx,
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


def train_sentence_mlp_mlx(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout: float = 0.2,
    pos_weight: float = 10.0,
    patience: int = 7,
    aux_weight: float = 0.01,
    label_smoothing: float = 0.0,
    train_splits: list[str] = None,
    dev_splits: list[str] = None,
    d_model: int = 256,
    fine_tune_layers: int = 0,
    layer_idx: Optional[int] = None,
):
    """
    MLX version of the training loop for Sentence Splitter.
    """
    if not HAS_MLX:
        raise ImportError("MLX is not installed. Use --backend transformers.")

    # 1. Setup Data (still use Torch DataLoader, but convert to MLX in the loop)
    # This reuse logic from train_sentence_mlp
    train_datasets = []
    suffix = "" if layer_idx is None else f"_L{layer_idx}"
    for s in (train_splits or ["train"]):
        train_datasets.append(CachedTokenEmbeddingDataset(SENTENCE_CACHE_DIR / f"{s}{suffix}"))
        aug_path = SENTENCE_CACHE_DIR / f"{s}_aug{suffix}"
        if aug_path.exists():
            train_datasets.append(CachedTokenEmbeddingDataset(aug_path))
    
    train_ds = ConcatDataset(train_datasets)
    dev_ds = ConcatDataset([CachedTokenEmbeddingDataset(SENTENCE_CACHE_DIR / f"{s}{suffix}") for s in (dev_splits or ["dev"])])
    
    hidden_dim = train_ds[0]["token_embeddings"].shape[-1]
    
    # Simple DataLoader (WeightedRandomSampler might need more care in conversion, but let's use standard shuffle for now)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=cached_token_collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=cached_token_collate_fn)

    # 2. Model
    mlp = SpacePredictorMLP_MLX(hidden_dim=hidden_dim, d_model=d_model, dropout=dropout)
    
    if fine_tune_layers > 0:
        print(f"Fine-tuning enabled (MLX): {fine_tune_layers} layers")
        from wordSplitter.embeddings import load_language_model as mlx_load_lm
        llm_full, _ = mlx_load_lm("mlx")
        
        # MLX model trunk identification
        trunk = getattr(llm_full, "model", None)
        if hasattr(llm_full, "language_model"):
            trunk = getattr(llm_full.language_model, "model", trunk)
            
        all_layers = trunk.layers
        fine_tune_start = len(all_layers) - fine_tune_layers
        target_layers = [all_layers[i] for i in range(fine_tune_start, len(all_layers))]
        
        # Apply LoRA to the target layers
        print(f"Applying LoRA to {len(target_layers)} transformer layers...")
        for layer in target_layers:
            apply_lora_to_module(layer, r=16, lora_alpha=32)
            
        # Ensure the MLP is in float32 for training stability
        import mlx.utils as mutils
        mlp.update(mutils.tree_map(lambda x: x.astype(mx.float32) if hasattr(x, "astype") else x, mlp.parameters()))
        
        model = FineTuneSentenceSplitterMLX(target_layers, mlp)
    else:
        model = mlp

    # 3. Loss and Optimizer
    def loss_fn(model, emb, labels, mask):
        # Apply label smoothing: 0 -> ε/2, 1 -> 1 - ε/2
        if label_smoothing > 0:
            labels = labels * (1 - label_smoothing) + label_smoothing / 2

        # emb, labels, mask are MLX arrays
        preds, moe_aux_loss = model(emb, mask=mask)
        
        # Focal Loss equivalent in MLX
        gamma = 2.0
        alpha = pos_weight
        
        # BCE with weight
        # Clipped predictions to avoid log(0)
        p = mx.clip(preds, 1e-6, 1 - 1e-6)
        bce = - (labels * mx.log(p) + (1 - labels) * mx.log(1 - p))
        
        pt = mx.where(labels >= 0.5, p, 1 - p)
        focal_weight = mx.power(mx.clip(1 - pt, 0, 1), gamma) # Added clip for power
        alpha_weight = mx.where(labels >= 0.5, alpha, 1.0)
        
        loss = alpha_weight * focal_weight * bce
        
        # Masking
        mask_f = mask.astype(mx.float32)
        valid_count = mx.sum(mask_f)
        masked_loss = mx.sum(loss * mask_f) / (valid_count + 1e-6)
        
        total_loss = masked_loss + aux_weight * moe_aux_loss
        return total_loss, moe_aux_loss

    loss_and_grad_fn = mnn.value_and_grad(model, loss_fn)
    optimizer = mopt.Adam(learning_rate=lr)

    print(f"Starting MLX Training (Samples: {len(train_ds)})")
    
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (MLX)", leave=False)
        for batch in pbar:
            # Convert Torch batch to MLX
            emb = mx.array(batch["token_embeddings"].numpy())
            labels = mx.array(batch["token_labels"].numpy())
            mask = mx.array(batch["token_mask"].numpy())

            (loss_val, aux_val), grads = loss_and_grad_fn(model, emb, labels, mask)
            
            # Gradient Clipping
            import mlx.utils as mutils
            max_grad_norm = 1.0
            grads_list = mutils.tree_flatten(grads)
            total_norm = mx.sqrt(mx.sum(mx.stack([mx.sum(g**2) for _, g in grads_list])))
            
            if mx.isnan(total_norm):
                print(f"\n⚠️ NaN gradients detected at batch {num_batches}! Skipping update.")
                continue
                
            scale = mx.minimum(1.0, max_grad_norm / (total_norm + 1e-6))
            grads = mutils.tree_map(lambda g: g * scale, grads)

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_val, aux_val)

            loss_item = loss_val.item()
            aux_item = aux_val.item()

            if np.isnan(loss_item):
                print(f"\n⚠️ NaN loss detected at batch {num_batches}!")
                # Optional: break or investigate
                
            total_loss += loss_item
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss_item:.4f}", "aux": f"{aux_item:.4f}"})

        avg_loss = total_loss / num_batches
        
        # Evaluation
        metrics = evaluate_tokens_mlx(model, dev_loader)
        
        print(
            f"Epoch {epoch:2d}/{epochs} (MLX) │ "
            f"Loss: {avg_loss:.4f} │ "
            f"Acc: {metrics['accuracy']:.4f} │ "
            f"P: {metrics['precision']:.4f} │ "
            f"R: {metrics['recall']:.4f} │ "
            f"F1: {metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_counter = 0
            # Save MLX weights
            model.save_weights(str(BEST_SENTENCE_CKPT.with_suffix(".safetensors")))
            # Also save a compatible .pt for metadata if needed, or just a json
            # here we'll save a small pt with metadata
            torch.save({
                "hidden_dim": hidden_dim,
                "d_model": d_model,
                "fine_tune_layers": fine_tune_layers,
                "layer_idx": layer_idx,
                "backend": "mlx",
                "f1": best_f1,
            }, BEST_SENTENCE_CKPT)
            print(f"  → Saved best MLX weights")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    print(f"✓ MLX Training complete. Best F1: {best_f1:.4f}")


@torch.no_grad()
def evaluate_tokens_mlx(model, dataloader) -> dict:
    """Evaluate MLX model."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        emb = mx.array(batch["token_embeddings"].numpy())
        mask = mx.array(batch["token_mask"].numpy())
        labels = batch["token_labels"].numpy()
        mask_cpu = batch["token_mask"].numpy()

        preds, _ = model(emb, mask=mask)
        preds_np = np.array(preds)

        for i in range(preds_np.shape[0]):
            valid = mask_cpu[i]
            p = (preds_np[i][valid] > 0.5).astype(int)
            l = labels[i][valid].astype(int)
            all_preds.extend(p.tolist())
            all_labels.extend(l.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


@torch.no_grad()
def evaluate_tokens(
    model: SpacePredictorMLP,
    dataloader: DataLoader,
    device: torch.device,
    top_k_errors: int = 0,
) -> dict:
    """Evaluate the MLP on token-level data, return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    error_samples = []

    for batch in dataloader:
        emb = batch["token_embeddings"].to(device)
        mask = batch["token_mask"].to(device)
        labels = batch["token_labels"].cpu()
        mask_cpu = batch["token_mask"].cpu()
        offsets = batch.get("token_offsets")
        texts = batch.get("texts", [""] * emb.shape[0])

        preds, _ = model(emb, mask=mask)
        preds = preds.cpu()

        for i in range(preds.shape[0]):
            valid = mask_cpu[i]
            p = (preds[i][valid] > 0.5).int()
            l = labels[i][valid].int()
            
            p_list = p.tolist()
            l_list = l.tolist()
            all_preds.extend(p_list)
            all_labels.extend(l_list)

            # Error tracking
            err_count = (p != l).sum().item()
            if err_count > 0 and top_k_errors > 0:
                # Highlight errors in text if offsets are available
                highlighted_text = ""
                if offsets is not None and torch.any(offsets[i] != 0):
                    sample_offsets = offsets[i][valid].tolist()
                    sample_text = texts[i]
                    last_pos = 0
                    
                    # Sort boundaries to apply correctly
                    for tok_idx, (start, end) in enumerate(sample_offsets):
                        pred_b = p_list[tok_idx]
                        true_b = l_list[tok_idx]
                        
                        # Add text until this token's end
                        highlighted_text += sample_text[last_pos:end]
                        last_pos = end
                        
                        if pred_b == 1 and true_b == 0:
                            highlighted_text += " [FP] " # False Positive
                        elif pred_b == 0 and true_b == 1:
                            highlighted_text += " [FN] " # False Negative
                else:
                    highlighted_text = texts[i]

                error_samples.append({
                    "errors": err_count,
                    "text": highlighted_text,
                    "pred_labels": p_list,
                    "true_labels": l_list
                })

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)

    # Sort and pick top k
    worst_samples = sorted(error_samples, key=lambda x: x["errors"], reverse=True)[:top_k_errors]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "worst_samples": worst_samples
    }
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentence Splitter MLP.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pos_weight", type=float, default=10.0)
    parser.add_argument("--extract", action="store_true", help="Extract and cache embeddings first.")
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "mlx"])
    parser.add_argument("--augment_prob", type=float, default=0.0)
    parser.add_argument("--max_chars", type=int, default=1024)
    parser.add_argument("--stride_chars", type=int, default=512)
    parser.add_argument("--fine_tune_layers", type=int, default=0, help="Number of last layers to fine-tune (requires transformers backend).")
    parser.add_argument("--layer_idx", type=int, default=None, help="Specific layer to extract from (defaults to num_layers - fine_tune_layers).")
    args = parser.parse_args()

    # Determine default layer_idx if fine-tuning
    effective_layer_idx = args.layer_idx
    if effective_layer_idx is None and args.fine_tune_layers > 0:
        # We need the layer count. For Qwen 0.6B it's 28.
        # But let's be generic: we can load config here.
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(MODEL_NAME)
        effective_layer_idx = config.num_hidden_layers - args.fine_tune_layers - 1
        print(f"Auto-selected intermediate layer index: {effective_layer_idx} (Model has {config.num_hidden_layers} layers)")

    if args.extract:
        extract_sentence_embeddings(
            batch_size=8,
            backend=args.backend,
            augment_prob=args.augment_prob,
            max_chars=args.max_chars,
            stride_chars=args.stride_chars,
            layer_idx=effective_layer_idx
        )

    if args.backend == "mlx":
        train_sentence_mlp_mlx(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            pos_weight=args.pos_weight,
            fine_tune_layers=args.fine_tune_layers,
            layer_idx=effective_layer_idx
        )
    else:
        train_sentence_mlp(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            pos_weight=args.pos_weight,
            train_splits=["train"] if args.backend == "transformers" else None, # handle splits
            fine_tune_layers=args.fine_tune_layers,
            layer_idx=effective_layer_idx,
            backend=args.backend
        )
