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
import itertools
import math
from dataclasses import dataclass


import torch

try:
    from typing import Any
except ImportError:
    Any = object

import nltk

# Initialize Italian dictionary using WordNet
_valid_words = set()
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import wordnet
    _valid_words = set(wordnet.all_lemma_names(lang='ita'))
except Exception as e:
    print(f"Warning: Could not initialize NLTK Italian dictionary: {e}")

from model import SpacePredictorMLP
from embeddings import (
    load_language_model,
    extract_token_embeddings,
    expand_to_char_embeddings,
    compute_perplexity,
    compute_perplexity_batch,
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
    minerva_model: Any,
    tokenizer: Any,
    device: torch.device,
    backend: str = "transformers"
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
    tok_emb = extract_token_embeddings(minerva_model, input_ids_t, attention_mask, backend=backend)

    # Expand to character level
    char_emb = expand_to_char_embeddings(tok_emb, char_to_token_t)

    # MLP prediction
    probs = mlp(char_emb)  # (1, num_chars)
    return probs.squeeze(0).cpu().tolist()


def build_text_from_mask(spaceless: str, space_present: list[bool]) -> str:
    """Build the string from the spaceless text and boolean space mask."""
    chars = []
    for i, ch in enumerate(spaceless):
        chars.append(ch)
        if i < len(space_present) and space_present[i]:
            chars.append(" ")
    return "".join(chars)

@dataclass
class BeamState:
    space_present: list[bool]
    mlp_log_prob: float
    fitness: float


def calculate_structural_penalty(text: str) -> float:
    """
    Calculate the structural penalty Omega(S).
    Penalizes:
      - Words longer than 15 characters
      - Orphan single-letter consonants
    """
    words = text.split(" ")
    penalty = 0.0
    valid_singles = {"a", "e", "i", "o", "u", "y", "d"}
    for w in words:
        if len(w) > 15:
            penalty += (len(w) - 15) * 1.0
        if len(w) == 1 and w.lower() not in valid_singles:
            if w.isalpha():
                penalty += 5.0
    return penalty


def calculate_dictionary_score(text: str) -> float:
    """
    Calculate the dictionary score for a text.
    Rewards words that are present in the NLTK Italian WordNet dictionary.
    """
    if not _valid_words:
        return 0.0

    words = text.split(" ")
    score = 0.0
    for w in words:
        if len(w) > 1 and w.lower() in _valid_words:
            # Reward +1.0 for each valid word found in the dictionary
            score += 1.0
    return score


def beam_search_decode(
    spaceless: str,
    probs: list[float],
    minerva_model: Any,
    tokenizer: Any,
    device: torch.device,
    beam_width: int = 20,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.1,
    delta: float = 0.5,
    early_exit_threshold: float = 0.999,
    backend: str = "transformers",
    verbose: bool = True
) -> str:
    """
    Left-to-Right Beam Search using Hybrid Fitness:
    Global Coherence (LM) + Local Evidence (MLP) - Structural Penalty.
    """
    # Initial state
    initial_state = BeamState(space_present=[], mlp_log_prob=0.0, fitness=0.0)
    beam = [initial_state]

    n_decisions = len(spaceless) - 1
    if n_decisions <= 0:
        return spaceless

    for i in range(n_decisions):
        p_remove = probs[i]
        p_keep = 1.0 - p_remove

        next_beam = []
        for state in beam:
            branches = []
            if p_remove > early_exit_threshold:
                # Early exit: MLP is highly confident space should be removed
                branches.append((False, p_remove))
            elif p_keep > early_exit_threshold:
                # Early exit: MLP is highly confident space should be kept
                branches.append((True, p_keep))
            else:
                branches.append((False, p_remove))
                branches.append((True, p_keep))

            for keep_space, prob in branches:
                new_space_map = state.space_present + [keep_space]
                # Avoid log(0)
                safe_prob = max(prob, 1e-10)
                new_mlp_log_prob = state.mlp_log_prob + math.log(safe_prob)
                
                next_beam.append(BeamState(
                    space_present=new_space_map,
                    mlp_log_prob=new_mlp_log_prob,
                    fitness=0.0
                ))
                
        # To evaluate PPL, we build the prefix string for the current step
        # The prefix includes characters up to index i+1
        prefix_chars = spaceless[:i+2]
        
        state_texts = []
        for state in next_beam:
            state_texts.append(build_text_from_mask(prefix_chars, state.space_present))
            
        # Batch compute Perplexity
        ppls = compute_perplexity_batch(minerva_model, tokenizer, state_texts, device, backend)
        
        for idx, state in enumerate(next_beam):
            ppl = ppls[idx]
            text = state_texts[idx]
            
            num_decisions = i + 1
            
            # Global Coherence: -ln(PPL) is already an intensive average log-prob per token.
            # Ranges typically from -5.0 to -1.0
            coherence_avg = -math.log(max(ppl, 1.0001))
            
            # Local Evidence: state.mlp_log_prob is a cumulative sum. 
            # We divide by num_decisions to get the intensive average log-prob per decision.
            # Ranges typically from -1.0 to 0.0
            evidence_avg = state.mlp_log_prob / num_decisions
            
            # Structural Penalty: total penalty so far.
            # We divide by num_decisions to keep its relative scale constant vs the other averages.
            penalty_avg = calculate_structural_penalty(text) / num_decisions
            
            # Dictionary Score: total dictionary reward so far.
            dict_avg = calculate_dictionary_score(text) / num_decisions
            
            # Hybrid Fitness calculation (all components are now intensive averages)
            # This ensures \alpha, \beta, \gamma have the same meaning at step 10 and step 100.
            state.fitness = alpha * coherence_avg + beta * evidence_avg - gamma * penalty_avg + delta * dict_avg
            
        # Sort by fitness descending
        next_beam.sort(key=lambda s: s.fitness, reverse=True)
        # Prune to beam width
        beam = next_beam[:beam_width]

        if verbose and (i + 1) % 10 == 0:
            best_text = build_text_from_mask(prefix_chars, beam[0].space_present)
            print(f"  Step {i+1:02d}/{n_decisions} | Best prefix: '{best_text}' | Fitness: {beam[0].fitness:.2f}")

    best_final_map = beam[0].space_present
    return build_text_from_mask(spaceless, best_final_map)


def split_text(
    text: str,
    mlp: SpacePredictorMLP,
    minerva_model: Any,
    tokenizer: Any,
    device: torch.device | None = None,
    backend: str = "transformers",
    threshold: float = 0.5,
    max_iterations: int = 50,
    verbose: bool = True,
) -> str:
    """
    Main entry point for splitting text using Beam Search decoding.
    (Note: `max_iterations` and `threshold` are ignored in this beam search approach, 
    but kept for API compatibility. In real usage, tweak beam_width and early_exit_threshold).
    """
    if device is None:
        device = get_device()

    spaceless = text.replace(" ", "")

    if verbose:
        print(f"Input: '{spaceless}'")
        print(f"{'─' * 60}")
        print("Running Beam Search Decoding...")

    # Step 1: Pre-compute MLP probabilities for all space positions
    probs = mlp_predict(spaceless, mlp, minerva_model, tokenizer, device, backend=backend)

    # Step 2: Decode with Beam Search
    current_text = beam_search_decode(
        spaceless=spaceless,
        probs=probs,
        minerva_model=minerva_model,
        tokenizer=tokenizer,
        device=device,
        beam_width=1,
        alpha=0.4,
        beta=2,
        gamma=0.55,
        delta=1.2,
        early_exit_threshold=1,
        backend=backend,
        verbose=verbose
    )

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Result: '{current_text}'")

    return current_text


def evaluate_beam_search(
    minerva_model: Any,
    tokenizer: Any,
    mlp: SpacePredictorMLP,
    device: torch.device,
    backend: str = "transformers",
    limit: int = 50,
    beam_width: int = 1,
    alpha: float = 1,
    beta: float = 3,
    gamma: float = 0,
    delta: float = 0,
    early_exit_threshold: float = 1,
    test_splits: list[str] = None,
):
    """
    Evaluate Beam Search decoding on the test dataset.
    Computes Exact Match and character boundary F1 score.
    """
    from data import WordSplitDataset
    from tqdm import tqdm
    import time

    print(f"Loading test dataset for Beam Search Evaluation (splits={test_splits}, limit={limit})...")
    # Load test dataset sentences
    sentences = []
    for split in test_splits:
        dataset = WordSplitDataset(split=split, tokenizer=tokenizer)
        sentences.extend(dataset.sentences)
    sentences = sentences[:limit]
    
    tp, fp, fn = 0, 0, 0
    exact_matches = 0
    total_possible_spaces = 0
    
    start_time = time.time()
    for words in tqdm(sentences, desc="Evaluating", unit="sent"):
        text_true = " ".join(words)
        spaceless = "".join(words)
        total_possible_spaces += max(0, len(spaceless) - 1)
        
        # Predict spaces
        probs = mlp_predict(spaceless, mlp, minerva_model, tokenizer, device, backend=backend)
        text_pred = beam_search_decode(
            spaceless=spaceless,
            probs=probs,
            minerva_model=minerva_model,
            tokenizer=tokenizer,
            device=device,
            beam_width=beam_width,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            early_exit_threshold=early_exit_threshold,
            backend=backend,
            verbose=False
        )
        
        if text_true == text_pred:
            exact_matches += 1
            
        # Helper to compute space positions
        def get_space_indices(spaced_str: str) -> set[int]:
            indices = set()
            char_idx = 0
            for ch in spaced_str:
                if ch == ' ':
                    indices.add(char_idx)
                else:
                    char_idx += 1
            return indices
            
        true_spaces = get_space_indices(text_true)
        pred_spaces = get_space_indices(text_pred)
        
        for pos in pred_spaces:
            if pos in true_spaces:
                tp += 1
            else:
                fp += 1
        for pos in true_spaces:
            if pos not in pred_spaces:
                fn += 1
                
    end_time = time.time()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (total_possible_spaces - fp - fn) / total_possible_spaces if total_possible_spaces > 0 else 0.0
    em = exact_matches / len(sentences) if len(sentences) > 0 else 0.0
    
    print("\n" + "="*50)
    print(f"Beam Search Evaluation Results (limit={limit})")
    print("="*50)
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    print(f"Sentences Evaluated: {len(sentences)}")
    print(f"Exact Match (EM): {em:.4f} ({exact_matches}/{len(sentences)})")
    print(f"Space Boundary Accuracy:  {accuracy:.4f}")
    print(f"Space Boundary Precision: {precision:.4f}")
    print(f"Space Boundary Recall:    {recall:.4f}")
    print(f"Space Boundary F1 Score:  {f1:.4f}")
    print("="*50)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "exact_match": em}
