"""
Inference pipeline for Sentence Splitter (token-level).
"""

from pathlib import Path
from typing import Optional
import torch
from wordSplitter.model import SpacePredictorMLP, FineTuneSentenceSplitter
from wordSplitter.embeddings import load_language_model, extract_token_embeddings, get_device, INSTRUCT_PROMPT, MODEL_NAME

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BEST_SENTENCE_CKPT = CHECKPOINT_DIR / "best_sentence_mlp.pt"
BEST_SENTENCE_MLX_CKPT = CHECKPOINT_DIR / "best_sentence_mlp.safetensors"


def load_sentence_mlp(checkpoint_path: Path | None = None, device: torch.device | None = None, backend: str = "transformers"):
    """Load the trained sentence MLP (and transformer layers if fine-tuned)."""
    if checkpoint_path is None:
        checkpoint_path = BEST_SENTENCE_CKPT
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hidden_dim = checkpoint.get("hidden_dim", 2560)
    d_model = checkpoint.get("d_model", 512)
    dropout = checkpoint.get("dropout", 0.2)
    fine_tune_layers = checkpoint.get("fine_tune_layers", 0)
    backend_used = checkpoint.get("backend", "transformers")
    
    if backend_used == "mlx" or backend == "mlx":
        from wordSplitter.model_mlx import SpacePredictorMLP as SpacePredictorMLP_MLX
        from wordSplitter.model_mlx import FineTuneSentenceSplitterMLX, apply_lora_to_module
        mlp_only = SpacePredictorMLP_MLX(hidden_dim=hidden_dim, d_model=d_model, dropout=dropout)
    else:
        mlp_only = SpacePredictorMLP(hidden_dim=hidden_dim, d_model=d_model, dropout=dropout)
    
    if fine_tune_layers > 0:
        print(f"Loading fine-tuned model ({backend_used}) with {fine_tune_layers} transformer layers...")
        llm_full, _ = load_language_model(backend_used if backend == "transformers" else backend, device)
        
        if backend_used == "mlx" or backend == "mlx":
            from wordSplitter.model_mlx import FineTuneSentenceSplitterMLX
            trunk = getattr(llm_full, "model", None)
            if hasattr(llm_full, "language_model"):
                trunk = getattr(llm_full.language_model, "model", trunk)
            all_layers = trunk.layers
            fine_tune_start = len(all_layers) - fine_tune_layers
            target_layers = [all_layers[i] for i in range(fine_tune_start, len(all_layers))]
            
            # Apply LoRA before loading weights
            for layer in target_layers:
                apply_lora_to_module(layer, r=16, lora_alpha=32)
                
            model = FineTuneSentenceSplitterMLX(target_layers, mlp_only)
        else:
            if hasattr(llm_full, "model") and hasattr(llm_full.model, "layers"):
                all_layers = llm_full.model.layers
                fine_tune_start = len(all_layers) - fine_tune_layers
                import torch.nn as nn
                target_layers = nn.ModuleList([all_layers[i] for i in range(fine_tune_start, len(all_layers))])
                model = FineTuneSentenceSplitter(target_layers, mlp_only).to(device)
            else:
                raise ValueError("Could not find transformer layers in LLM for fine-tuning reconstruction.")
    else:
        model = mlp_only
        if backend_used != "mlx" and backend != "mlx":
            model = model.to(device)

    if backend_used == "mlx" or backend == "mlx":
        # MLX weights are in .safetensors
        mlx_path = checkpoint_path.with_suffix(".safetensors")
        if mlx_path.exists():
            model.load_weights(str(mlx_path))
        else:
            print(f"Warning: MLX weights not found at {mlx_path}")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    return model, checkpoint.get("layer_idx")


@torch.no_grad()
def split_into_sentences(
    text: str,
    mlp: SpacePredictorMLP,
    llm_model: any,
    tokenizer: any,
    device: torch.device,
    backend: str = "transformers",
    threshold: float = 0.5,
    max_chars: int = 1024,
    stride_chars: int = 512,
    layer_idx: Optional[int] = None,
) -> list[str]:
    """
    Split a continuous text into a list of sentences using the MLP.
    
    Works at token-level: the MLP predicts P(sentence_boundary) per token,
    then offset_mapping is used to find the character position of each boundary.
    """
    # Accumulate boundary probabilities per character (for sliding window averaging)
    probs_sum = [0.0] * len(text)
    probs_count = [0] * len(text)
    
    prompt_char_len = len(INSTRUCT_PROMPT)
    
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text = text[start:end]
        
        # Prepend instruction prompt
        prompted_text = f"{INSTRUCT_PROMPT}{chunk_text}"
        
        # Tokenize with offset mapping
        encoding = tokenizer(
            prompted_text,
            return_tensors="pt",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        offsets = encoding["offset_mapping"].squeeze(0).tolist()
        
        # Extract token embeddings
        tok_emb = extract_token_embeddings(llm_model, input_ids, attention_mask, backend=backend, layer_idx=layer_idx)

        # Identify valid text token indices (exclude BOS and Prompt)
        text_token_indices = []
        for i, (ts, te) in enumerate(offsets):
            if ts == 0 and te == 0: continue # BOS/EOS/Special
            if te <= prompt_char_len: continue # Prompt tokens
            text_token_indices.append(i)
            
        if not text_token_indices:
            continue

        # Extract only the text token embeddings (matching training-time format)
        valid_tok_emb = tok_emb[:, text_token_indices, :]
        
        # Predict
        if backend == "mlx":
            import mlx.core as mx
            valid_tok_mx = mx.array(valid_tok_emb.cpu().numpy())
            # For MLX transformer layers, we might need a mask
            # but currently split_into_sentences doesn't build a proper 4D mask.
            # We'll pass None or a simple 2D mask if the model supports it.
            text_token_probs, _ = mlp(valid_tok_mx)
            text_token_probs = [float(p) for p in text_token_probs.reshape(-1)]
        else:
            # Note: We pass the filtered sequence so positional embeddings and context match training.
            text_token_probs, _ = mlp(valid_tok_emb)
            text_token_probs = text_token_probs.squeeze(0).cpu().tolist()
        
        # Create a mapping of probabilities for all tokens (default 0 for non-text tokens)
        token_probs_raw = [0.0] * len(offsets)
        for idx_in_valid, orig_idx in enumerate(text_token_indices):
            token_probs_raw[orig_idx] = text_token_probs[idx_in_valid]
            
        # Map token probabilities back to character positions
        for tok_idx in text_token_indices:
            tok_start, tok_end = offsets[tok_idx]
            
            # Map back to original chunk relative position
            chunk_last_char_pos = (tok_end - 1) - prompt_char_len
            if chunk_last_char_pos < 0:
                continue
                
            abs_pos = start + chunk_last_char_pos
            if abs_pos < len(text):
                probs_sum[abs_pos] += token_probs_raw[tok_idx]
                probs_count[abs_pos] += 1
                
        if end == len(text):
            break
            
        start += stride_chars

    # Average probabilities from overlapping windows
    probs = [s / max(1, c) for s, c in zip(probs_sum, probs_count)]
    
    # Split text at boundary positions
    sentences = []
    current_sent = ""
    
    for i, char in enumerate(text):
        current_sent += char
        if i < len(probs) and probs[i] > threshold:
            sentences.append(current_sent.strip())
            current_sent = ""
            
    if current_sent.strip():
        sentences.append(current_sent.strip())
        
    return sentences
