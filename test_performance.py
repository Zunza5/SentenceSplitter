import torch
import time
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from sentence_embeddings import load_language_model, extract_token_embeddings, get_device
from model import SpacePredictorMLP
from data_sentence import get_sentence_dataloader

def test_performance(
    split="test",
    backend="transformers",
    batch_size=8,
    max_chars=2048,
    stride_chars=2048,
    device=None,
):
    if device is None:
        device = get_device()

    if max_chars >= 4096 and batch_size > 2:
        print(f"Warning: max_chars={max_chars} with batch_size={batch_size} can trigger OOM; forcing batch_size=2.")
        batch_size = 2
    
    # 1. Load Models
    print(f"--- Loading Models (Backend: {backend}) ---")
    llm_model, tokenizer = load_language_model(backend=backend, device=device)
    
    checkpoint_path = Path("checkpoints/best_sentence_mlp.pt")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hidden_dim = checkpoint.get("hidden_dim", 2048)
    d_model = checkpoint.get("d_model", checkpoint.get("cnn_dim", 256))
    dropout = checkpoint.get("dropout", 0.3)
    num_experts = checkpoint.get("num_experts", 8)
    top_k = checkpoint.get("top_k", min(2, num_experts))
    
    mlp = SpacePredictorMLP(
        hidden_dim=hidden_dim,
        d_model=d_model,
        dropout=dropout,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)
    mlp.load_state_dict(checkpoint["model_state_dict"])
    mlp.eval()
    print(f"Loaded MLP from {checkpoint_path} (F1: {checkpoint.get('f1', 'N/A'):.4f})")

    # 2. Load Data
    print(f"--- Loading Test Data: {split} ---")
    dataloader = get_sentence_dataloader(
        split=split,
        batch_size=batch_size,
        tokenizer=tokenizer,
        shuffle=False,
        max_chars=max_chars,
        stride_chars=stride_chars,
        augmentation_mode="original"
    )
    
    # 3. Inference Loop
    print(f"--- Running Inference on {len(dataloader.dataset)} chunks ---")
    all_preds = []
    all_labels = []
    
    total_extract_time = 0.0
    total_expand_time = 0.0
    total_mlp_time = 0.0
    
    num_processed = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["token_labels"].to(device)
            mask = batch["token_mask"].to(device)
            
            # Phase A: LLM Token Embeddings
            t0 = time.time()
            tok_emb = extract_token_embeddings(llm_model, input_ids, attention_mask, backend=backend)
            t1 = time.time()
            total_extract_time += (t1 - t0)
            
            # Phase B: model input stays token-level (no char expansion)
            t2 = time.time()
            token_emb = tok_emb.float()
            t3 = time.time()
            total_expand_time += (t3 - t2)
            
            # Phase C: MLP Forward
            t4 = time.time()
            preds, _ = mlp(token_emb, mask=mask)
            t5 = time.time()
            total_mlp_time += (t5 - t4)
            
            
            # Process results for metrics
            for b in range(preds.shape[0]):
                valid = mask[b].bool()
                p = (preds[b][valid] > 0.5).int().cpu().tolist()
                l = labels[b][valid].int().cpu().tolist()
                all_preds.extend(p)
                all_labels.extend(l)
                num_processed += 1
            
            if (i + 1) % 10 == 0:
                print(f" Batch {i+1}/{len(dataloader)} processed...")

    # 4. Calculate Scores
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)

    total_inference_time = total_extract_time + total_expand_time + total_mlp_time
    avg_time_per_sentence = total_inference_time / num_processed if num_processed > 0 else 0
    
    # 5. Report Results
    print("\n" + "="*40)
    print(f"PERFORMANCE RESULTS - Split: {split}")
    print("="*40)
    print(f"Chunks processed:      {num_processed}")
    print(f"Accuracy:              {accuracy:.4f}")
    print(f"Precision:             {precision:.4f}")
    print(f"Recall:                {recall:.4f}")
    print(f"F1 Score:              {f1:.4f}")
    print("-" * 40)
    print(f"Total Inference Time:  {total_inference_time:.4f} s")
    print(f"  - Extr. Embeddings:  {total_extract_time:.4f} s ({total_extract_time/total_inference_time*100:.1f}%)")
    print(f"  - Expand Embeddings: {total_expand_time:.4f} s ({total_expand_time/total_inference_time*100:.1f}%)")
    print(f"  - MLP Prediction:    {total_mlp_time:.4f} s ({total_mlp_time/total_inference_time*100:.1f}%)")
    print("-" * 40)
    print(f"Avg Time per Chunk:    {avg_time_per_sentence*1000:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test performance of Sentence Splitter")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to test (test, test2, engTest, etc.)")
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "mlx"], help="Model backend")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max-chars", type=int, default=2048, help="Max characters per chunk")
    parser.add_argument("--stride-chars", type=int, default=2048, help="Chunk stride in characters")
    
    args = parser.parse_args()
    
    test_performance(
        split=args.split,
        backend=args.backend,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
        stride_chars=args.stride_chars,
    )
