"""
Sentence Splitter — CLI entry point.
"""

import argparse
import torch
from transformers import AutoTokenizer

from train_sentence import (
    extract_sentence_embeddings,
    train_sentence_mlp,
    SENTENCE_CACHE_DIR,
    CachedEmbeddingDataset,
    cached_collate_fn,
    evaluate,
)
from data_sentence import UD_URLS
from inference_sentence import load_sentence_mlp, split_into_sentences, load_language_model, get_device
from sentence_embeddings import MODEL_NAME
from torch.utils.data import DataLoader, ConcatDataset


ALL_TRAIN_SPLITS = ",".join(sorted(s for s in UD_URLS if s.endswith("-train")))
ALL_DEV_SPLITS = ",".join(sorted(s for s in UD_URLS if s.endswith("-dev")))
ALL_TEST_SPLITS = ",".join(sorted(s for s in UD_URLS if s.endswith("-test")))


def _collect_top_errors(model, dataloader, device, threshold=0.5, top_k=5):
    """Collect top-k most confident wrong token predictions (FP/FN)."""
    errors = []
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            emb = batch["token_embeddings"].to(device).float()
            token_mask = batch["token_mask"].to(device)
            labels = batch["token_labels"].cpu()
            mask = batch["token_mask"].cpu()
            texts = batch.get("spaceless", [""] * emb.shape[0])

            outputs = model(emb, mask=token_mask)
            probs = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = probs.cpu()

            for sample_idx in range(probs.shape[0]):
                valid = mask[sample_idx]
                valid_probs = probs[sample_idx][valid].tolist()
                valid_labels = labels[sample_idx][valid].int().tolist()
                sample_text = texts[sample_idx] if sample_idx < len(texts) else ""
                preview = (sample_text[:180] + "...") if len(sample_text) > 180 else sample_text

                for tok_idx, (p, y) in enumerate(zip(valid_probs, valid_labels)):
                    pred = 1 if p > threshold else 0
                    if pred == y:
                        continue

                    # Prioritize highly confident mistakes.
                    confidence = float(p) if pred == 1 else float(1.0 - p)
                    errors.append(
                        {
                            "confidence": confidence,
                            "prob": float(p),
                            "label": int(y),
                            "pred": int(pred),
                            "error_type": "FP" if pred == 1 else "FN",
                            "batch_idx": batch_idx,
                            "sample_idx": sample_idx,
                            "token_idx": tok_idx,
                            "raw_text": sample_text,
                            "text_preview": preview,
                        }
                    )

    errors.sort(key=lambda e: e["confidence"], reverse=True)
    return errors[: max(0, top_k)]


def _visualize_error_span(tokenizer, text: str, valid_token_idx: int, window: int = 40):
    """Return a compact context string with the wrong token span highlighted."""
    if not text:
        return None

    enc = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    input_ids = enc["input_ids"]

    valid = []
    for tid, (start, end) in zip(input_ids, offsets):
        if start == 0 and end == 0:
            continue
        if end <= start:
            continue
        valid.append((tid, start, end))

    if valid_token_idx < 0 or valid_token_idx >= len(valid):
        return None

    tid, start, end = valid[valid_token_idx]
    token_str = tokenizer.convert_ids_to_tokens([tid])[0]

    left = max(0, start - window)
    right = min(len(text), end + window)
    local = text[left:right]

    rel_start = start - left
    rel_end = end - left
    marked = local[:rel_start] + "[[" + local[rel_start:rel_end] + "]]" + local[rel_end:]

    return {
        "token_str": token_str,
        "span": (start, end),
        "context": marked,
    }

def cmd_train(args):
    if args.phase in ("extract", "both"):
        extract_sentence_embeddings(
            batch_size=args.extract_batch_size, 
            backend=args.backend,
            augment_prob=args.augment_prob,
            max_chars=args.max_chars,
            stride_chars=args.stride_chars,
            seed=args.seed,
        )
    if args.phase in ("train", "both"):
        train_sentence_mlp(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            d_model=args.d_model,
            dropout=args.dropout,
            pos_weight=args.pos_weight,
            grad_clip_norm=args.grad_clip_norm,
            train_splits=[s.strip() for s in args.train_splits.split(",")],
            dev_splits=[s.strip() for s in args.dev_splits.split(",")],
            augment_prob=args.augment_prob,
            aux_weight=args.aux_weight,
            balanced_batches=args.balanced_batches,
            seed=args.seed,
        )

def cmd_eval(args):
    device = get_device()
    mlp = load_sentence_mlp(device=device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    splits = [s.strip() for s in args.test_splits.split(",")]
    test_ds = ConcatDataset([CachedEmbeddingDataset(SENTENCE_CACHE_DIR / s) for s in splits])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=cached_collate_fn)
    
    metrics = evaluate(mlp, test_loader, device)
    print(f"\nSentence Splitting Test Results ({args.test_splits}):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

    top_errors = _collect_top_errors(
        model=mlp,
        dataloader=test_loader,
        device=device,
        threshold=args.threshold,
        top_k=args.top_k_errors,
    )

    print(f"\nTop {args.top_k_errors} model errors (most confident wrong predictions):")
    if not top_errors:
        print("  No errors found.")
    else:
        for rank, err in enumerate(top_errors, 1):
            print(
                f"  {rank}. {err['error_type']} | p={err['prob']:.4f} | "
                f"pred={err['pred']} gt={err['label']} | "
                f"batch={err['batch_idx']} sample={err['sample_idx']} tok={err['token_idx']}"
            )
            viz = _visualize_error_span(tokenizer, err.get("raw_text", ""), err["token_idx"])
            if viz is not None:
                s, e = viz["span"]
                print(f"     token: {viz['token_str']} | span=({s},{e})")
                print(f"     where: {viz['context']}")
            if err["text_preview"]:
                print(f"     text: {err['text_preview']}")

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
    train_parser.add_argument("--dropout", type=float, default=0.3)
    train_parser.add_argument("--pos-weight", type=float, default=1.0, help="Positive class weight for imbalanced data")
    train_parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping; set <= 0 to disable",
    )
    train_parser.add_argument("--aux-weight", type=float, default=0.00001, help="MoE load-balancing loss weight")
    train_parser.add_argument("--augment_prob", type=float, default=0.0)
    train_parser.add_argument("--extract-batch-size", type=int, default=8)
    train_parser.add_argument("--max-chars", type=int, default=1024)
    train_parser.add_argument("--stride-chars", type=int, default=512)
    train_parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for reproducibility")
    train_parser.add_argument(
        "--balanced-batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Balance source datasets via DataLoader sampling (default: enabled)",
    )
    train_parser.add_argument("--train-splits", type=str, default='it-isdt-train,it-vit-train,it-partut-train,it-markit-train, it-old-train,it-parlamint-train, it-e3c-train, en-ewt-train,en-gum-train,en-partut-train, en-genia-train, en-lines-train')
    train_parser.add_argument("--dev-splits", type=str, default='it-isdt-dev,it-vit-dev,it-partut-dev,it-markit-dev,en-ewt-dev,en-gum-dev,en-partut-dev')

    # eval
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--batch-size", type=int, default=16)
    eval_parser.add_argument("--test-splits", type=str, default=ALL_TEST_SPLITS)
    eval_parser.add_argument("--threshold", type=float, default=0.5)
    eval_parser.add_argument("--top-k-errors", type=int, default=5)

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
