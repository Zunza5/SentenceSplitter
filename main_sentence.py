"""
Sentence Splitter — CLI entry point.
"""

import argparse
import sys
from pathlib import Path
import torch

from train_sentence import (
    extract_sentence_embeddings, train_sentence_mlp,
    SENTENCE_CACHE_DIR, BEST_SENTENCE_CKPT,
    CachedTokenEmbeddingDataset, cached_token_collate_fn, evaluate_tokens,
)
from inference_sentence import load_sentence_mlp, split_into_sentences
from wordSplitter.embeddings import load_language_model, get_device
from torch.utils.data import DataLoader, ConcatDataset

def cmd_train(args):
    if args.phase in ("extract", "both"):
        # Auto-calculate layer_idx if not provided
        effective_layer_idx = args.layer_idx
        if effective_layer_idx is None and args.fine_tune_layers > 0:
            from transformers import AutoConfig
            from wordSplitter.embeddings import MODEL_NAME
            config = AutoConfig.from_pretrained(MODEL_NAME)
            effective_layer_idx = config.num_hidden_layers - args.fine_tune_layers - 1
            print(f"Auto-selected intermediate layer index: {effective_layer_idx}")

        extract_sentence_embeddings(
            batch_size=args.extract_batch_size, 
            backend=args.backend,
            max_chars=args.max_chars,
            stride_chars=args.stride_chars,
            layer_idx=effective_layer_idx
        )
    if args.phase in ("train", "both"):
        # Re-calculate layer_idx if not provided (needed for path indexing)
        effective_layer_idx = args.layer_idx
        if effective_layer_idx is None and args.fine_tune_layers > 0:
            from transformers import AutoConfig
            from wordSplitter.embeddings import MODEL_NAME
            config = AutoConfig.from_pretrained(MODEL_NAME)
            effective_layer_idx = config.num_hidden_layers - args.fine_tune_layers - 1

        if args.backend == "mlx":
            from train_sentence import train_sentence_mlp_mlx
            train_sentence_mlp_mlx(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                dropout=args.dropout,
                pos_weight=args.pos_weight,
                train_splits=[s.strip() for s in args.train_splits.split(",")],
                dev_splits=[s.strip() for s in args.dev_splits.split(",")],
                aux_weight=args.aux_weight,
                label_smoothing=args.label_smoothing,
                d_model=args.d_model,
                fine_tune_layers=args.fine_tune_layers,
                layer_idx=effective_layer_idx,
            )
        else:
            train_sentence_mlp(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                dropout=args.dropout,
                pos_weight=args.pos_weight,
                train_splits=[s.strip() for s in args.train_splits.split(",")],
                dev_splits=[s.strip() for s in args.dev_splits.split(",")],
                aux_weight=args.aux_weight,
                label_smoothing=args.label_smoothing,
                d_model=args.d_model,
                fine_tune_layers=args.fine_tune_layers,
                layer_idx=effective_layer_idx,
                backend=args.backend,
            )

def cmd_eval(args):
    device = get_device()
    mlp, layer_idx = load_sentence_mlp(device=device)
    
    splits = [s.strip() for s in args.test_splits.split(",")]
    suffix = "" if layer_idx is None else f"_L{layer_idx}"
    test_ds = ConcatDataset([CachedTokenEmbeddingDataset(SENTENCE_CACHE_DIR / f"{s}{suffix}") for s in splits])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=cached_token_collate_fn)
    
    metrics = evaluate_tokens(mlp, test_loader, device, top_k_errors=10)
    print(f"\nSentence Splitting Test Results ({args.test_splits}):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

    if metrics.get("worst_samples"):
        print(f"\nTop {len(metrics['worst_samples'])} samples with most errors:")
        for i, sample in enumerate(metrics["worst_samples"]):
            print(f"\nSample {i+1} ({sample['errors']} errors):")
            p_count = sum(sample["pred_labels"])
            l_count = sum(sample["true_labels"])
            print(f"  Predicted boundaries: {p_count}, Actual boundaries: {l_count}")
            # Show the highlighted text
            print(f"  Text: {sample['text']}")

def cmd_split(args):
    device = get_device()
    llm_model, tokenizer = load_language_model(args.backend, device)
    mlp, layer_idx = load_sentence_mlp(device=device, backend=args.backend)
    
    sentences = split_into_sentences(
        text=args.text,
        mlp=mlp,
        llm_model=llm_model,
        tokenizer=tokenizer,
        device=device,
        backend=args.backend,
        threshold=args.threshold,
        layer_idx=layer_idx
    )
    
    print(f"\nDetected {len(sentences)} sentences:")
    for i, s in enumerate(sentences):
        print(f"  {i+1}. {s}")

def main():
    parser = argparse.ArgumentParser(description="Sentence Splitter CLI")
    subparsers = parser.add_subparsers(dest="command")

    # train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--backend", choices=["transformers", "mlx"], default="transformers")
    train_parser.add_argument("--phase", choices=["extract", "train", "both"], default="both")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--d-model", type=int, default=1024, help="Model hidden dimension")
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--pos-weight", type=float, default=1.2)
    train_parser.add_argument("--aux-weight", type=float, default=0.001, help="MoE load-balancing loss weight")
    train_parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    train_parser.add_argument("--extract-batch-size", type=int, default=8)
    train_parser.add_argument("--max-chars", type=int, default=1024)
    train_parser.add_argument("--stride-chars", type=int, default=512)
    train_parser.add_argument("--train-splits", type=str, default="it-isdt-train,it-vit-train,it-partut-train,it-markit-train,en-ewt-train,en-gum-train,en-partut-train, it-old-train, it-parlamint-train")
    train_parser.add_argument("--dev-splits", type=str, default="it-isdt-dev,it-vit-dev,it-partut-dev,it-markit-dev,en-ewt-dev,en-gum-dev,en-partut-dev")
    train_parser.add_argument("--fine-tune-layers", type=int, default=3, help="Number of last layers to fine-tune")
    train_parser.add_argument("--layer-idx", type=int, default=None, help="The intermediate layer to use as features")

    # eval
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--backend", choices=["transformers", "mlx"], default="transformers")
    eval_parser.add_argument("--batch-size", type=int, default=16)
    eval_parser.add_argument("--test-splits", type=str, default="it-isdt-test,it-postwita-test,it-vit-test,it-twittiro-test,it-partut-test,it-markit-test,en-ewt-test,en-gum-test,en-partut-test,en-pud-test")

    # split
    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--backend", choices=["transformers", "mlx"], default="transformers")
    split_parser.add_argument("text", type=str)
    split_parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "split":
        cmd_split(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
