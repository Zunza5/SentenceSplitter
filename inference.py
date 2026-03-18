"""
Inference pipeline for the Word Splitter.

Approach: The input spaceless text is expanded to have spaces between every
character. The MLP predicts which spaces to REMOVE. Perplexity verification
confirms removals one at a time (greedy best-first).

Iterative loop:
  1. MLP predicts P(remove space) at each character position
  2. Candidate selection: positions where P(remove) > threshold
  3. Minerva perplexity verification: pick the best removal
  4. Convergence: repeat until no more spaces are removed
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import SpacePredictorMLP
from embeddings import (
    load_minerva,
    extract_token_embeddings,
    expand_to_char_embeddings,
    compute_perplexity,
    get_device,
)
from data import build_char_to_token_map

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def load_mlp(checkpoint_path: Path | None = None, device: torch.device | None = None):
    """Load the trained MLP from a checkpoint."""
    if device is None:
        device = get_device()
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "best_mlp.pt"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    mlp = SpacePredictorMLP(
        hidden_dim=ckpt["hidden_dim"],
        dropout=ckpt.get("dropout", 0.2),
    )
    mlp.load_state_dict(ckpt["model_state_dict"])
    mlp.to(device)
    mlp.eval()
    print(f"Loaded MLP checkpoint (epoch {ckpt['epoch']}, F1={ckpt['f1']:.4f})")
    return mlp


@torch.no_grad()
def mlp_predict(
    text: str,
    mlp: SpacePredictorMLP,
    minerva_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> list[float]:
    """
    Run MLP inference on a spaced-out text string.

    The text is first expanded to "c i a o ..." format, then tokenized
    and fed through Minerva + MLP.

    Returns:
        List of P(remove space) probabilities, one per character.
    """
    # Expand to spaced format
    spaced = " ".join(list(text.replace(" ", "")))

    # Tokenize and get char→token map
    input_ids, char_to_token = build_char_to_token_map(spaced, tokenizer)

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids_t)
    char_to_token_t = torch.tensor([char_to_token], dtype=torch.long, device=device)

    # Extract token embeddings from Minerva
    tok_emb = extract_token_embeddings(minerva_model, input_ids_t, attention_mask)

    # Expand to character level
    char_emb = expand_to_char_embeddings(tok_emb, char_to_token_t)

    # MLP prediction
    probs = mlp(char_emb)  # (1, num_chars)
    return probs.squeeze(0).cpu().tolist()


def select_candidates(
    probs: list[float],
    current_text: str,
    threshold: float = 0.5,
) -> list[int]:
    """
    Select candidate space positions to REMOVE.

    Maps MLP character-level predictions back to space positions
    in the current text.

    Args:
        probs: per-character P(remove space) from the MLP
        current_text: the current text with some spaces remaining
        threshold: minimum probability to consider

    Returns:
        List of indices into current_text where spaces could be removed.
    """
    candidates = []

    # Map original char indices to space positions in current_text
    # current_text has spaces; we need to find which spaces correspond
    # to which original character boundaries
    char_idx = 0
    for i, ch in enumerate(current_text):
        if ch == " ":
            # This space sits between original char_idx-1 and char_idx
            # The MLP probability at char_idx-1 tells us if this space should be removed
            if char_idx > 0 and char_idx - 1 < len(probs):
                if probs[char_idx - 1] > threshold:
                    candidates.append(i)
        else:
            char_idx += 1

    return candidates


def verify_with_perplexity(
    text: str,
    candidates: list[int],
    minerva_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    already_confirmed: set[int],
    device: torch.device,
) -> list[int]:
    """
    Verify candidate space REMOVALS using perplexity.

    Evaluates all candidates and picks only the single best one —
    the removal that yields the largest perplexity drop.

    Args:
        text: current text with spaces
        candidates: indices of spaces that could be removed
        minerva_model: Minerva model for perplexity computation
        tokenizer: Minerva tokenizer
        already_confirmed: set of positions already removed (skip)
        device: computation device

    Returns:
        List with at most one confirmed space index to remove.
    """
    pp_baseline = compute_perplexity(minerva_model, tokenizer, text, device)

    best_pos = None
    best_pp = pp_baseline

    pp = []

    for pos in candidates:
        if pos in already_confirmed:
            continue

        # Build text with this space removed
        text_without_space = text[:pos] + text[pos + 1:]
        pp_without = compute_perplexity(minerva_model, tokenizer, text_without_space, device)

        if pp_without < best_pp *1.2:
            pp.append(pos)
            best_pp = pp_without
            best_pos = pos

    # Accept only if the best removal significantly reduces perplexity
    if best_pos is not None and best_pp < pp_baseline *1.2:
        return pp

    return []


def remove_space_at(text: str, pos: int) -> str:
    """Remove the space at position `pos` in the text."""
    return text[:pos] + text[pos + 1:]


def split_text(
    text: str,
    mlp: SpacePredictorMLP,
    minerva_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device | None = None,
    threshold: float = 0.5,
    max_iterations: int = 50,
    verbose: bool = True,
) -> str:
    """
    Full iterative word splitting pipeline.

    Starting from text with spaces between every character,
    iteratively remove spaces that the MLP + perplexity agree on.

    Args:
        text: input spaceless text
        mlp: trained MLP model
        minerva_model: Minerva model
        tokenizer: Minerva tokenizer
        device: computation device
        threshold: P(remove) threshold for candidate selection
        max_iterations: maximum number of iterations
        verbose: print progress information

    Returns:
        Text with recovered word boundaries.
    """
    if device is None:
        device = get_device()

    # Remove any existing spaces and create fully-spaced version
    spaceless = text.replace(" ", "")
    current_text = " ".join(list(spaceless))
    removed_positions: set[int] = set()

    if verbose:
        print(f"Input: '{spaceless}'")
        print(f"Spaced: '{current_text}'")
        print(f"Threshold: {threshold}")
        print(f"{'─' * 60}")

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\nIteration {iteration}:")

        # Step 1: MLP prediction on the spaceless text
        probs = mlp_predict(spaceless, mlp, minerva_model, tokenizer, device)

        # Step 2: Candidate selection (spaces in current_text to remove)
        candidates = select_candidates(probs, current_text, threshold)
        new_candidates = [c for c in candidates if c not in removed_positions]

        if verbose:
            print(f"  MLP candidates: {len(candidates)} total, {len(new_candidates)} new")

        if not new_candidates:
            if verbose:
                print("  → No new candidates. Converged!")
            break

        # Step 3: Perplexity verification — pick best single removal
        confirmed = verify_with_perplexity(
            text=current_text,
            candidates=new_candidates,
            minerva_model=minerva_model,
            tokenizer=tokenizer,
            already_confirmed=removed_positions,
            device=device,
        )

        if verbose:
            print(f"  Perplexity confirmed: {len(confirmed)} / {len(new_candidates)}")

        if not confirmed:
            if verbose:
                print("  → No removals confirmed. Converged!")
            break

        # Apply the confirmed removal
        for pos in confirmed:
            current_text = remove_space_at(current_text, pos - len(removed_positions))

            # Track removed positions (adjust for shifted indices)
            removed_positions.add(pos)

        if verbose:
            print(f"  Current: '{current_text}'")

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Result: '{current_text}'")

    return current_text
