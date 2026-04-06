import argparse
import torch
import time
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

# Import useful functions and constants from the existing training script
from train_sentence import (
    CachedEmbeddingDataset, 
    cached_collate_fn, 
    evaluate, 
    _seed_worker, 
    set_deterministic_seed,
    _build_balanced_sample_weights,
    _build_torch_generator,
    SENTENCE_CACHE_DIR,
    CHECKPOINT_DIR
)
from torch.utils.data import WeightedRandomSampler
from model import SpacePredictorMLP, FocalLoss
from sentence_embeddings import get_device

def finetune_mlp(
    train_splits: list[str],
    dev_splits: list[str],
    base_ckpt_path: Path,
    output_ckpt_name: str = "finetuned_sentence_mlp.pt",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-5,
    dropout: float = 0.3,
    pos_weight: float = 10.0,
    grad_clip_norm: float = 1.0,
    aux_weight: float = 0.01,
    patience: int = 3,
    balanced_batches: bool = True,
    augment_prob: float = 0.0,
    seed: int = 42
):
    set_deterministic_seed(seed)
    device = get_device()
    
    output_ckpt_path = CHECKPOINT_DIR / output_ckpt_name
    
    print(f"Loading base checkpoint from: {base_ckpt_path}")
    if not base_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {base_ckpt_path}")
        
    checkpoint = torch.load(base_ckpt_path, map_location=device, weights_only=True)
    
    # 1. Reconstruct the model with original parameters from the checkpoint
    hidden_dim = checkpoint.get("hidden_dim", 2048)
    d_model = checkpoint.get("cnn_dim", checkpoint.get("d_model", 256))
    num_experts = checkpoint.get("num_experts", 8)
    top_k = checkpoint.get("top_k", 2)
    
    mlp = SpacePredictorMLP(
        hidden_dim=hidden_dim,
        d_model=d_model,
        dropout=dropout, # Passed from arguments
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)
    
    mlp.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Base model loaded successfully (dropout={dropout}).")

    # 2. Load "hard" datasets using cached embeddings
    print(f"\nLoading Fine-Tuning data... Train: {train_splits}, Dev: {dev_splits}")
    try:
        train_datasets = []
        for s in train_splits:
            train_datasets.append(CachedEmbeddingDataset(SENTENCE_CACHE_DIR / s))
            # Automatically load augmented dataset if it exists and augment_prob > 0
            if augment_prob > 0:
                aug_path = SENTENCE_CACHE_DIR / f"{s}_aug"
                if aug_path.exists():
                    print(f"  → Also loading augmented samples from {aug_path}")
                    train_datasets.append(CachedEmbeddingDataset(aug_path))
                    
        train_ds = ConcatDataset(train_datasets)
        
        # Macro-F1: individual loaders for each dev split
        dev_loaders = {}
        for s in dev_splits:
            ds = CachedEmbeddingDataset(SENTENCE_CACHE_DIR / s)
            dev_loaders[s] = DataLoader(
                ds, 
                batch_size=batch_size, 
                shuffle=False,
                collate_fn=cached_collate_fn, 
                num_workers=0, # num_workers=0 safe for Mac
                worker_init_fn=_seed_worker
            )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Ensure you have extracted embeddings for these splits using 'pdm run python main_sentence.py train --phase extract'!")
        return

    # On macOS, use 0 workers to avoid spawn overhead
    num_workers = 0 if torch.backends.mps.is_available() else 4

    if balanced_batches:
        sample_weights = _build_balanced_sample_weights(train_datasets)
        sampler_gen = _build_torch_generator(seed)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=sampler_gen,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            collate_fn=cached_collate_fn, num_workers=num_workers, worker_init_fn=_seed_worker
        )
    else:
        train_gen = _build_torch_generator(seed)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=cached_collate_fn, num_workers=num_workers, worker_init_fn=_seed_worker,
            generator=train_gen
        )

    # Note: Collective metrics tracking for dev is handled in the loop below
    total_dev_samples = sum(len(loader.dataset) for loader in dev_loaders.values())

    print(f"Train samples: {len(train_ds)} | Dev samples (total): {total_dev_samples}")

    # 3. Setup Loss and Optimizer (Reduced LR)
    criterion = FocalLoss(alpha=pos_weight, gamma=2.0, reduction="none")
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-5) # Light weight decay
    
    best_f1 = 0.0
    patience_counter = 0

    print("\nStarting Fine-Tuning...")
    for epoch in range(1, epochs + 1):
        mlp.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            emb = batch["token_embeddings"].to(device).float()
            labels = batch["token_labels"].to(device)
            mask = batch["token_mask"].to(device)

            preds, moe_aux_loss = mlp(emb, mask=mask)

            loss_all = criterion(preds, labels)
            bce_loss = (loss_all * mask.float()).sum() / max(mask.float().sum(), 1.0)
            loss_total = bce_loss + aux_weight * moe_aux_loss

            optimizer.zero_grad()
            loss_total.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            total_loss += loss_total.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss_total.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        
        # 4. Evaluation using Macro-F1 (arithmetic mean across datasets)
        print(f"\n--- Valutazione Epoca {epoch} ---")
        macro_f1 = 0.0
        
        for dev_name, loader in dev_loaders.items():
            metrics = evaluate(mlp, loader, device)
            macro_f1 += metrics['f1']
            print(f"[{dev_name:12}] Acc: {metrics['accuracy']:.4f} │ P: {metrics['precision']:.4f} │ R: {metrics['recall']:.4f} │ F1: {metrics['f1']:.4f}")
        
        macro_f1 = macro_f1 / len(dev_loaders)
        print(f"MACRO F1 SCORE: {macro_f1:.4f} (Avg across all Dev splits)")

        # 5. Save the new checkpoint based on Macro-F1
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience_counter = 0
            
            # Match the original checkpoint save structure
            checkpoint_data = {
                "model_state_dict": mlp.state_dict(),
                "hidden_dim": hidden_dim,
                "d_model": d_model,
                "cnn_dim": d_model,
                "dropout": mlp.drop.p,
                "num_experts": num_experts,
                "top_k": top_k,
                "f1": best_f1,
            }
            torch.save(checkpoint_data, output_ckpt_path)
            print(f"  → Saved new fine-tuned model to {output_ckpt_path.name} (Macro F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                break

    print(f"\n✓ Fine-Tuning complete. Best MACRO F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the Sentence MLP on hard datasets")
    parser.add_argument("--train-splits", type=str, default="en-gum-train, en-ewt-train, en-partut-train", help="Comma-separated train splits")
    parser.add_argument("--dev-splits", type=str, default="en-gum-dev, en-ewt-dev, en-partut-dev", help="Comma-separated dev splits")
    parser.add_argument("--base-ckpt", type=str, default="best_sentence_mlp.pt", help="Base checkpoint name (in checkpoints/)")
    parser.add_argument("--output-ckpt", type=str, default="finetuned_sentence_mlp.pt", help="Output checkpoint fixed name")
    
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=1.2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.00001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment_prob", type=float, default=0.0)
    parser.add_argument(
        "--balanced-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    
    args = parser.parse_args()
    
    base_ckpt = CHECKPOINT_DIR / args.base_ckpt
    
    train_list = [s.strip() for s in args.train_splits.split(",")]
    dev_list = [s.strip() for s in args.dev_splits.split(",")]
    
    finetune_mlp(
        train_splits=train_list,
        dev_splits=dev_list,
        base_ckpt_path=base_ckpt,
        output_ckpt_name=args.output_ckpt,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
        pos_weight=args.pos_weight,
        grad_clip_norm=args.grad_clip_norm,
        aux_weight=args.aux_weight,
        seed=args.seed,
        augment_prob=args.augment_prob,
        balanced_batches=args.balanced_batches
    )
