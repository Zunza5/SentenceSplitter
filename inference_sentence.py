"""
Inference pipeline for Sentence Splitter.
"""

from pathlib import Path
import torch
from wordSplitter.model import SpacePredictorMLP
from wordSplitter.embeddings import load_language_model, get_device
from wordSplitter.inference import load_mlp, mlp_predict

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BEST_SENTENCE_CKPT = CHECKPOINT_DIR / "best_sentence_mlp.pt"


def load_sentence_mlp(checkpoint_path: Path | None = None, device: torch.device | None = None):
    """Load the trained sentence MLP."""
    if checkpoint_path is None:
        checkpoint_path = BEST_SENTENCE_CKPT
    return load_mlp(checkpoint_path, device)


def split_into_sentences(
    text: str,
    mlp: SpacePredictorMLP,
    llm_model: any,
    tokenizer: any,
    device: torch.device,
    backend: str = "transformers",
    threshold: float = 0.5,
) -> list[str]:
    """
    Split a continuous text into a list of sentences using the MLP.
    """
    # mlp_predict works on text and returns P(boundary) per character.
    # Note: inference.py's mlp_predict uses spaced = " ".join(list(text.replace(" ", "")))
    # which is what our Word Splitter expected (every char separated by space).
    # For Sentence Splitting, our dataset used the ACTUAL text (with original spaces).
    
    # We need a version of mlp_predict that doesn't spaced-out the text if we want 
    # to preserve original word-internal spaces.
    
    from data_sentence import build_sentence_char_to_token_map
    from wordSplitter.embeddings import extract_token_embeddings, expand_to_char_embeddings
    
    # Custom prediction for sentence splitting (preserves spaces)
    input_ids, char_to_token = build_sentence_char_to_token_map(text, tokenizer)
    
    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids_t)
    char_to_token_t = torch.tensor([char_to_token], dtype=torch.long, device=device)
    
    tok_emb = extract_token_embeddings(llm_model, input_ids_t, attention_mask, backend=backend)
    char_emb = expand_to_char_embeddings(tok_emb, char_to_token_t)
    
    probs = mlp(char_emb).squeeze(0).cpu().tolist()
    
    sentences = []
    current_sent = ""
    
    for i, char in enumerate(text):
        current_sent += char
        # If probability of boundary is high, split
        if i < len(probs) and probs[i] > threshold:
            sentences.append(current_sent.strip())
            current_sent = ""
            
    if current_sent.strip():
        sentences.append(current_sent.strip())
        
    return sentences
