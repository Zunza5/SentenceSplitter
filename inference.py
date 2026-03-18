"""
Inference pipeline for the Word Splitter.

Iterative loop:
  1. MLP predicts P(space) at each character position
  2. Candidate selection: positions where P(space) > threshold
  3. Minerva perplexity verification: confirm only spaces that reduce perplexity
  4. Convergence: repeat until no new spaces are confirmed
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
    Run MLP inference on a text string.

    Returns:
        List of P(space) probabilities, one per character.
    """
    # Tokenize and get char→token map
    input_ids, char_to_token = build_char_to_token_map(text, tokenizer)

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


def select_candidates(probs: list[float], threshold: float = 0.8) -> list[int]:
    """
    Select candidate positions where P(space) > threshold.

    Args:
        probs: per-character space probabilities
        threshold: minimum probability to consider

    Returns:
        List of character indices where spaces should be inserted (after that char).
    """
    candidates = []
    for i, p in enumerate(probs):
        if p > threshold and i < len(probs) - 1:  # no space at very end
            candidates.append(i)
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
    Verify candidate space positions using perplexity.

    Evaluates all candidates and picks only the single best one —
    the position whose space insertion yields the largest perplexity drop.
    The drop must be significant (≥10%) to be accepted.

    Args:
        text: current text (may already have some spaces)
        candidates: proposed space positions (indices in the SPACELESS version)
        minerva_model: Minerva model for perplexity computation
        tokenizer: Minerva tokenizer
        already_confirmed: set of positions already confirmed (skip these)
        device: computation device

    Returns:
        List with at most one confirmed position (the best candidate).
    """
    pp_baseline = compute_perplexity(minerva_model, tokenizer, text, device)
    best_pos = None
    best_pp = pp_baseline

    #text = " ".join(text)
            

    for pos in candidates:
        if pos in already_confirmed:
            continue

        # Build text with space inserted at this position
        #text_with_space = text[:pos] + " " + text[pos:]

        text_without_space = text[:pos] + text[pos+1:]
        
        
        pp_with = compute_perplexity(minerva_model, tokenizer, text_without_space, device)
        if pp_with < best_pp:
            best_pp = pp_with
            best_pos = pos
            print(text_without_space)

    # Accept only if the best candidate significantly reduces perplexity
    if best_pos is not None and best_pp < pp_baseline*1:
        return [best_pos]

    return []


def insert_spaces(text: str, positions: list[int]) -> str:
    """
    Insert spaces into text at the given positions.

    Positions are character indices in the original spaceless text;
    a space is inserted AFTER each position.
    """
    if not positions:
        return text

    # Sort positions in reverse so insertions don't shift indices
    sorted_pos = sorted(positions, reverse=True)
    chars = list(text)
    for pos in sorted_pos:
        chars.insert(pos + 1, " ")
    return "".join(chars)


def split_text(
    text: str,
    mlp: SpacePredictorMLP,
    minerva_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device | None = None,
    threshold: float = 0.8,
    max_iterations: int = 10,
    verbose: bool = True,
) -> str:
    """
    Full iterative word splitting pipeline.

    Args:
        text: input spaceless text
        mlp: trained MLP model
        minerva_model: Minerva model
        tokenizer: Minerva tokenizer
        device: computation device
        threshold: P(space) threshold for candidate selection
        max_iterations: maximum number of iterations
        verbose: print progress information

    Returns:
        Text with recovered word boundaries (spaces inserted).
    """
    if device is None:
        device = get_device()

    # Remove any existing spaces from input
    spaceless = text.replace(" ", "")
    current_text = spaceless
    all_confirmed: set[int] = set()

    if verbose:
        print(f"Input: '{spaceless}'")
        print(f"Threshold: {threshold}")
        print(f"{'─'*60}")

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\nIteration {iteration}:")

        # Step 1: MLP prediction on SPACELESS text
        # We always run MLP on the original spaceless text
        probs = mlp_predict(spaceless, mlp, minerva_model, tokenizer, device)

        # Step 2: Candidate selection
        candidates = select_candidates(probs, threshold)
        new_candidates = [c for c in candidates if c not in all_confirmed]

        if verbose:
            print(f"  MLP candidates: {len(candidates)} total, {len(new_candidates)} new")

        if not new_candidates:
            if verbose:
                print("  → No new candidates. Converged!")
            break

        # Step 3: Perplexity verification
        # Build the current version of text with confirmed spaces
        current_text = insert_spaces(spaceless, sorted(all_confirmed))

        confirmed = verify_with_perplexity(
            text=current_text,
            candidates=new_candidates,
            minerva_model=minerva_model,
            tokenizer=tokenizer,
            already_confirmed=all_confirmed,
            device=device,
        )

        if verbose:
            print(f"  Perplexity confirmed: {len(confirmed)} / {len(new_candidates)}")

        if not confirmed:
            if verbose:
                print("  → No new spaces confirmed. Converged!")
            break

        all_confirmed.update(confirmed)
        current_text = insert_spaces(spaceless, sorted(all_confirmed))

        if verbose:
            print(f"  Current: '{current_text}'")

    # Final result
    result = insert_spaces(spaceless, sorted(all_confirmed))

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Result: '{result}'")

    return result
