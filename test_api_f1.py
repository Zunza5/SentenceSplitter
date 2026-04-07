"""
Evaluate the SentenceSplitterAPI, SpaCy, and NLTK on raw text.
All models receive the same reconstructed raw document and their predicted
boundary indices are compared against canonicalized ground truth labels.
"""
import argparse
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import spacy
import nltk
from nltk.tokenize import sent_tokenize

from api_sentence import SentenceSplitterAPI
from data_sentence import get_sentences_for_split, make_sentence_bounds_labels
from compare_spacy import canonicalize_boundary_index, get_spacy_model


def build_evaluation_data(split_name: str):
    """
    Load ground truth sentences, reconstruct raw text, and produce
    canonicalized GT boundary labels (same schema as the API uses).
    """
    sentences = get_sentences_for_split(split_name)
    if not sentences:
        raise ValueError(f"Dataset '{split_name}' is empty.")

    text, labels = make_sentence_bounds_labels(sentences)
    total_len = len(text)

    # Canonicalize GT boundaries to match the API's alignment schema
    raw_gt = [i for i, l in enumerate(labels) if l == 1]
    canon_gt = set(int(canonicalize_boundary_index(text, b, total_len)) for b in raw_gt)

    gt_labels = np.zeros(total_len, dtype=np.int16)
    for b in canon_gt:
        if 0 <= b < total_len:
            gt_labels[b] = 1
    gt_labels[-1] = -1  # Last char has no following boundary

    return text, gt_labels, len(sentences)


def boundaries_from_spacy(text: str, nlp) -> list[int]:
    """Run SpaCy on text and return boundary character indices."""
    doc = nlp(text)
    boundaries = []
    for sent in doc.sents:
        b = sent.end_char
        if b < len(text):
            canon_b = int(canonicalize_boundary_index(text, b, len(text)))
            boundaries.append(canon_b)
    return sorted(set(boundaries))


def boundaries_from_nltk(text: str, language: str = "english") -> list[int]:
    """Run NLTK on text and return boundary character indices."""
    sents = sent_tokenize(text, language=language)
    boundaries = []
    offset = 0
    for i, sent in enumerate(sents[:-1]):  # last sentence has no trailing boundary
        idx = text.find(sent, offset)
        if idx == -1:
            offset += len(sent)
            continue
        b = idx + len(sent)
        if b < len(text):
            canon_b = int(canonicalize_boundary_index(text, b, len(text)))
            boundaries.append(canon_b)
        offset = idx + len(sent)
    return sorted(set(boundaries))


def score(pred_boundaries: list[int], gt_labels: np.ndarray, total_len: int):
    """Compute precision, recall, F1, accuracy from boundary lists vs GT array."""
    pred_labels = np.zeros(total_len, dtype=np.int16)
    for b in pred_boundaries:
        if 0 <= b < total_len:
            pred_labels[b] = 1

    valid = gt_labels >= 0
    gt_v = gt_labels[valid]
    pr_v = pred_labels[valid]

    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_v, pr_v, average="binary", zero_division=0
    )
    accuracy = accuracy_score(gt_v, pr_v)
    pred_count = int((pr_v == 1).sum())
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "pred_count": pred_count}


def print_comparison(results: dict, elapsed: dict, gt_count: int):
    names = list(results.keys())
    print("\n" + "=" * 80)
    print(f"{'COMPARISON ON RAW TEXT':^80}")
    print("=" * 80)
    header = f"{'Metric':<22}" + "".join(f"| {n:<18}" for n in names)
    print(header)
    print("-" * 80)

    rows = [
        ("GT Boundaries", lambda n: str(gt_count)),
        ("Pred Boundaries", lambda n: str(results[n]["pred_count"])),
        ("Accuracy", lambda n: f"{results[n]['accuracy']:.4f}"),
        ("Precision", lambda n: f"{results[n]['precision']:.4f}"),
        ("Recall", lambda n: f"{results[n]['recall']:.4f}"),
        ("F1 Score", lambda n: f"{results[n]['f1']:.4f}"),
        ("Time (s)", lambda n: f"{elapsed[n]:.2f}"),
    ]

    for label, fn in rows:
        row = f"{label:<22}" + "".join(f"| {fn(n):<18}" for n in names)
        print(row)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare API, SpaCy, NLTK on raw text")
    parser.add_argument("--split", type=str, default="en-pud-test")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_sentence_mlp.pt")
    parser.add_argument("--backend", type=str, default="transformers")
    parser.add_argument("--max-chars", type=int, default=1024)
    parser.add_argument("--stride-chars", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    language = "italian" if "it" in args.split else "english"

    # --- Load ground truth ---
    print(f"Loading dataset: {args.split}")
    text, gt_labels, gt_sent_count = build_evaluation_data(args.split)
    total_len = len(text)
    gt_count = int((gt_labels == 1).sum())
    print(f"Document: {total_len} chars | {gt_sent_count} sentences | {gt_count} boundaries")

    # --- Load models ---
    print("\nLoading SpaCy...")
    nlp = get_spacy_model(language)

    print("Loading NLTK...")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print("Loading SentenceSplitterAPI...")
    api = SentenceSplitterAPI(
        checkpoint_path=args.checkpoint,
        backend=args.backend,
        max_chars=args.max_chars,
        stride_chars=args.stride_chars,
        batch_size=args.batch_size,
    )

    # --- SpaCy ---
    print("\nRunning SpaCy...")
    t0 = time.time()
    spacy_boundaries = boundaries_from_spacy(text, nlp)
    spacy_time = time.time() - t0

    # --- NLTK ---
    print("Running NLTK...")
    t0 = time.time()
    nltk_boundaries = boundaries_from_nltk(text, language)
    nltk_time = time.time() - t0

    # --- API ---
    print("Running SentenceSplitterAPI...")
    t0 = time.time()
    api_boundaries = api.get_boundaries(text)
    api_time = time.time() - t0

    # --- Score ---
    results = {
        "SpaCy": score(spacy_boundaries, gt_labels, total_len),
        "NLTK": score(nltk_boundaries, gt_labels, total_len),
        "LLM (API)": score(api_boundaries, gt_labels, total_len),
    }
    elapsed = {"SpaCy": spacy_time, "NLTK": nltk_time, "LLM (API)": api_time}

    print_comparison(results, elapsed, gt_count)


if __name__ == "__main__":
    main()
