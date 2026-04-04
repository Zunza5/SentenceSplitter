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
    SENTENCE_CACHE_DIR,
    CHECKPOINT_DIR
)
from model import SpacePredictorMLP, FocalLoss
from sentence_embeddings import get_device

def finetune_mlp(
    train_splits: list[str],
    dev_splits: list[str],
    base_ckpt_path: Path,
    output_ckpt_name: str = "finetuned_sentence_mlp.pt",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-5,  # Very low learning rate for fine-tuning
    pos_weight: float = 10.0,
    aux_weight: float = 0.01,
    patience: int = 3,
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
        dropout=0.3, # Keep robust dropout to prevent overfitting on specific datasets
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)
    
    mlp.load_state_dict(checkpoint["model_state_dict"])
    print("✓ Base model loaded successfully.")

    # 2. Load "hard" datasets using cached embeddings
    print(f"\nLoading Fine-Tuning data... Train: {train_splits}, Dev: {dev_splits}")
    try:
        train_ds = ConcatDataset([CachedEmbeddingDataset(SENTENCE_CACHE_DIR / s) for s in train_splits])
        dev_ds = ConcatDataset([CachedEmbeddingDataset(SENTENCE_CACHE_DIR / s) for s in dev_splits])
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Ensure you have extracted embeddings for these splits using 'pdm run python main_sentence.py train --phase extract'!")
        return

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=cached_collate_fn, num_workers=0, worker_init_fn=_seed_worker # num_workers=0 safe for Mac
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        collate_fn=cached_collate_fn, num_workers=0, worker_init_fn=_seed_worker
    )

    print(f"Train samples: {len(train_ds)} | Dev samples: {len(dev_ds)}")

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
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_total.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss_total.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        
        # 4. Evaluation on "hard" Dev set
        metrics = evaluate(mlp, dev_loader, device)

        print(
            f"Epoch {epoch:2d}/{epochs} │ Loss: {avg_loss:.4f} │ "
            f"Acc: {metrics['accuracy']:.4f} │ P: {metrics['precision']:.4f} │ "
            f"R: {metrics['recall']:.4f} │ F1: {metrics['f1']:.4f}"
        )

        # 5. Save the new checkpoint
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
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
            print(f"  → Saved new fine-tuned model to {output_ckpt_path.name} (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                break

    print(f"\n✓ Fine-Tuning complete. Best target F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the Sentence MLP on hard datasets")
    parser.add_argument("--train-splits", type=str, required=True, help="Comma-separated train splits (e.g. it-vit-train,it-partut-train)")
    parser.add_argument("--dev-splits", type=str, required=True, help="Comma-separated dev splits (e.g. it-vit-dev,it-partut-dev)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--epochs", type=int, default=5, help="Max epochs")
    
    args = parser.parse_args()
    
    base_ckpt = CHECKPOINT_DIR / "best_sentence_mlp.pt"
    
    train_list = [s.strip() for s in args.train_splits.split(",")]
    dev_list = [s.strip() for s in args.dev_splits.split(",")]
    
    finetune_mlp(
        train_splits=train_list,
        dev_splits=dev_list,
        base_ckpt_path=base_ckpt,
        lr=args.lr,
        epochs=args.epochs
    )
