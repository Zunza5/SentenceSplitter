"""
Sentence Splitter — CLI entry point.
"""

import argparse

from train_sentence import (
    extract_sentence_embeddings,
    train_sentence_mlp,
    SENTENCE_CACHE_DIR,
    CachedEmbeddingDataset,
    cached_collate_fn,
    evaluate,
)
from inference_sentence import load_sentence_mlp, split_into_sentences, load_language_model, get_device
from torch.utils.data import DataLoader, ConcatDataset

def cmd_train(args):
    if args.phase in ("extract", "both"):
        extract_sentence_embeddings(
            batch_size=args.extract_batch_size, 
            backend=args.backend,
            augment_prob=args.augment_prob,
            max_chars=args.max_chars,
            stride_chars=args.stride_chars
        )
    if args.phase in ("train", "both"):
        train_sentence_mlp(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            d_model=args.d_model,
            dropout=args.dropout,
            pos_weight=args.pos_weight,
            train_splits=[s.strip() for s in args.train_splits.split(",")],
            dev_splits=[s.strip() for s in args.dev_splits.split(",")],
            augment_prob=args.augment_prob,
            aux_weight=args.aux_weight,
            balanced_batches=args.balanced_batches,
        )

def cmd_eval(args):
    device = get_device()
    mlp = load_sentence_mlp(device=device)
    
    splits = [s.strip() for s in args.test_splits.split(",")]
    test_ds = ConcatDataset([CachedEmbeddingDataset(SENTENCE_CACHE_DIR / s) for s in splits])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=cached_collate_fn)
    
    metrics = evaluate(mlp, test_loader, device)
    print(f"\nSentence Splitting Test Results ({args.test_splits}):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

def cmd_split(args):
    device = get_device()
    llm_model, tokenizer = load_language_model(args.backend, device)
    mlp = load_sentence_mlp(device=device)
    
    sentences = split_into_sentences(
        text=args.text,
        mlp=mlp,
        llm_model=llm_model,
        tokenizer=tokenizer,
        device=device,
        backend=args.backend,
        threshold=args.threshold
    )
    
    print(f"\nDetected {len(sentences)} sentences:")
    for i, s in enumerate(sentences):
        print(f"  {i+1}. {s}")

def main():
    parser = argparse.ArgumentParser(description="Sentence Splitter CLI")
    parser.add_argument("--backend", choices=["transformers", "mlx"], default="transformers")
    subparsers = parser.add_subparsers(dest="command")

    # train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--phase", choices=["extract", "train", "both"], default="both")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--d-model", type=int, default=256, help="MoE/CNN internal dimension")
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--pos-weight", type=float, default=0.5)
    train_parser.add_argument("--aux-weight", type=float, default=0.01, help="MoE load-balancing loss weight")
    train_parser.add_argument("--augment_prob", type=float, default=0.0)
    train_parser.add_argument("--extract-batch-size", type=int, default=8)
    train_parser.add_argument("--max-chars", type=int, default=512)
    train_parser.add_argument("--stride-chars", type=int, default=256)
    train_parser.add_argument(
        "--balanced-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance source datasets via DataLoader sampling (default: enabled)",
    )
    train_parser.add_argument("--train-splits", type=str, default="it-isdt-train,it-vit-train,it-partut-train,it-markit-train,en-ewt-train,en-gum-train,en-partut-train, it-old-train, it-parlamint-train")
    train_parser.add_argument("--dev-splits", type=str, default="it-isdt-dev,it-vit-dev,it-partut-dev,it-markit-dev,en-ewt-dev,en-gum-dev,en-partut-dev")

    # eval
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--batch-size", type=int, default=16)
    eval_parser.add_argument("--test-splits", type=str, default="it-isdt-test,it-postwita-test,it-vit-test,it-twittiro-test,it-partut-test,it-markit-test,en-ewt-test,en-gum-test,en-partut-test,en-pud-test")

    # split
    split_parser = subparsers.add_parser("split")
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
