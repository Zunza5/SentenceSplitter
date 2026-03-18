"""
Word Splitter — CLI entry point.

Usage:
    python main.py train [--phase extract|train|both] [--epochs N] [--batch-size N]
    python main.py split "textowithoutspaces" [--threshold 0.8] [--no-verify]
"""

import argparse
import sys

import torch


def cmd_train(args):
    """Run the training pipeline."""
    from train import extract_embeddings, train_mlp

    if args.phase in ("extract", "both"):
        extract_embeddings(batch_size=args.extract_batch_size)

    if args.phase in ("train", "both"):
        train_mlp(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            pos_weight=args.pos_weight,
        )


def cmd_split(args):
    """Run inference to split text."""
    from inference import load_mlp, split_text
    from embeddings import load_minerva, get_device

    device = get_device()

    # Load models
    minerva_model, tokenizer = load_minerva(device)
    mlp = load_mlp(device=device)

    text = args.text
    threshold = args.threshold

    if args.no_verify:
        # Fast mode: MLP only, no perplexity verification
        from inference import mlp_predict, select_candidates

        spaceless = text.replace(" ", "")
        spaced = " ".join(list(spaceless))
        probs = mlp_predict(spaceless, mlp, minerva_model, tokenizer, device)

        # Remove spaces where P(remove) > threshold
        result_chars = list(spaceless)
        result = []
        for i, ch in enumerate(result_chars):
            result.append(ch)
            if i < len(probs) - 1 and probs[i] <= threshold:
                result.append(" ")
        result = "".join(result)

        print(f"\n{'─'*60}")
        print(f"Input:  '{text}'")
        print(f"Output: '{result}'")
        print(f"{'─'*60}")
        print(f"(MLP-only mode, no perplexity verification)")
    else:
        # Full iterative pipeline with perplexity verification
        result = split_text(
            text=text,
            mlp=mlp,
            minerva_model=minerva_model,
            tokenizer=tokenizer,
            device=device,
            threshold=threshold,
            max_iterations=args.max_iter,
            verbose=True,
        )

    return result


def cmd_eval(args):
    """Evaluate the MLP on the test set."""
    from train import CachedEmbeddingDataset, cached_collate_fn, evaluate
    from inference import load_mlp
    from embeddings import get_device, CACHE_DIR
    from torch.utils.data import DataLoader

    device = get_device()
    mlp = load_mlp(device=device)

    test_ds = CachedEmbeddingDataset(CACHE_DIR / "test")
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=cached_collate_fn,
    )

    metrics = evaluate(mlp, test_loader, device)
    print(f"\nTest Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Word Splitter — Split spaceless text into words",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                           # Extract embeddings + train MLP
  python main.py train --phase extract           # Only extract embeddings
  python main.py train --phase train --epochs 20 # Only train MLP, 20 epochs
  python main.py split "ciaocomestaivabene"      # Split text (full pipeline)
  python main.py split "ciaocomestaivabene" --no-verify  # MLP-only (fast)
  python main.py eval                            # Evaluate on test set
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ──
    train_parser = subparsers.add_parser("train", help="Train the MLP model")
    train_parser.add_argument(
        "--phase",
        choices=["extract", "train", "both"],
        default="both",
    )
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--pos-weight", type=float, default=2.0)
    train_parser.add_argument("--extract-batch-size", type=int, default=16)

    # ── split ──
    split_parser = subparsers.add_parser("split", help="Split spaceless text")
    split_parser.add_argument("text", type=str, help="Text to split")
    split_parser.add_argument("--threshold", type=float, default=0.8)
    split_parser.add_argument("--max-iter", type=int, default=10)
    split_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip perplexity verification (faster but less accurate)",
    )

    # ── eval ──
    eval_parser = subparsers.add_parser("eval", help="Evaluate on test set")
    eval_parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "split":
        cmd_split(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
