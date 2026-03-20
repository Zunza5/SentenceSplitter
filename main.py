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
        extract_embeddings(batch_size=args.extract_batch_size, backend=args.backend)

    if args.phase in ("train", "both"):
        train_mlp(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            pos_weight=args.pos_weight,
            train_splits=[s.strip() for s in args.train_splits.split(",")],
            dev_splits=[s.strip() for s in args.dev_splits.split(",")],
        )


def cmd_split(args):
    """Run inference to split text."""
    from inference import load_mlp, split_text
    from embeddings import load_language_model, get_device

    device = get_device()

    # Load models
    minerva_model, tokenizer = load_language_model(args.backend, device)
    mlp = load_mlp(device=device)

    text = args.text
    threshold = args.threshold

    if args.no_verify:
        # Fast mode: MLP only, no perplexity verification
        from inference import mlp_predict

        spaceless = text.replace(" ", "")
        spaced = " ".join(list(spaceless))
        probs = mlp_predict(spaceless, mlp, minerva_model, tokenizer, device, backend=args.backend)

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
            backend=args.backend,
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
    from torch.utils.data import DataLoader, ConcatDataset

    device = get_device()
    mlp = load_mlp(device=device)

    splits = [s.strip() for s in args.test_splits.split(",")]
    
    # Pre-load original texts to match with cache if missing
    all_cached_datasets = []
    for s in splits:
        ds = CachedEmbeddingDataset(CACHE_DIR / s)
        # Check if first sample has text. If not, load it from source.
        if ds.samples and "spaceless" not in ds.samples[0]:
            print(f"Loading source text for {s} split...")
            from data import WordSplitDataset
            source_ds = WordSplitDataset(split=s)
            for i, samp in enumerate(ds.samples):
                if i < len(source_ds.samples):
                    samp["spaceless"] = source_ds.samples[i]["spaceless"]
        all_cached_datasets.append(ds)

    test_ds = ConcatDataset(all_cached_datasets)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=cached_collate_fn,
    )

    metrics = evaluate(mlp, test_loader, device, top_k_errors=10)
    print(f"\nTest Results ({args.test_splits}):")
    print(f"  Exact Match (EM): {metrics['exact_match']:.4f}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"  F1:               {metrics['f1']:.4f}")

    if metrics["worst_samples"]:
        print(f"\nTop 10 Worst Performing Sentences:")
        print(f"{'─' * 60}")
        for i, s in enumerate(metrics["worst_samples"]):
            print(f"  {i+1}. Errors: {s['errors']}")
            print(f"     PRED: '{s['pred']}'")
            print(f"     TRUE: '{s['true']}'")
            print(f"{'─' * 60}")


def cmd_eval_beam(args):
    """Evaluate the Beam Search decoding on the test set."""
    from inference import load_mlp, evaluate_beam_search
    from embeddings import load_language_model, get_device

    device = get_device()
    
    # Load models
    minerva_model, tokenizer = load_language_model(args.backend, device)
    mlp = load_mlp(device=device)

    evaluate_beam_search(
        minerva_model=minerva_model,
        tokenizer=tokenizer,
        mlp=mlp,
        device=device,
        backend=args.backend,
        limit=args.limit,
        beam_width=args.beam_width,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        early_exit_threshold=args.early_exit,
        test_splits=[s.strip() for s in args.test_splits.split(",")],
    )


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
  python main.py eval                            # Evaluate MLP on test set
  python main.py eval-beam --limit 5             # Evaluate Beam Search on test set
""",
    )

    parser.add_argument(
        "--backend",
        choices=["transformers", "mlx"],
        default="transformers",
        help="Backend to use for the Language Model (transformers or mlx)."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ──
    train_parser = subparsers.add_parser("train", help="Train the MLP model")
    train_parser.add_argument(
        "--phase",
        choices=["extract", "train", "both"],
        default="both",
    )
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--pos-weight", type=float, default=0.3)
    train_parser.add_argument("--extract-batch-size", type=int, default=16)
    train_parser.add_argument("--train-splits", type=str, default="train", help="Comma-separated list of training splits")
    train_parser.add_argument("--dev-splits", type=str, default="dev", help="Comma-separated list of dev splits")

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
    eval_parser.add_argument("--test-splits", type=str, default="test", help="Comma-separated list of test splits")

    # ── eval-beam ──
    eval_beam_parser = subparsers.add_parser("eval-beam", help="Evaluate Beam Search on test set")
    eval_beam_parser.add_argument("--limit", type=int, default=50, help="Number of sentences to evaluate")
    eval_beam_parser.add_argument("--beam-width", type=int, default=1)
    eval_beam_parser.add_argument("--alpha", type=float, default=0.4)
    eval_beam_parser.add_argument("--beta", type=float, default=3)
    eval_beam_parser.add_argument("--gamma", type=float, default=0.55)
    eval_beam_parser.add_argument("--delta", type=float, default=1.2, help="Weight for dictionary score")
    eval_beam_parser.add_argument("--early-exit", type=float, default=1)
    eval_beam_parser.add_argument("--test-splits", type=str, default="test", help="Comma-separated test splits")

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
    elif args.command == "eval-beam":
        cmd_eval_beam(args)


if __name__ == "__main__":
    main()
