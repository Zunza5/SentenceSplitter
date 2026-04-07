import torch
import numpy as np
from pathlib import Path

# Imports from codebase
from sentence_embeddings import load_language_model, extract_token_embeddings, get_device
from model import SpacePredictorMLP
from data_sentence import build_sentence_char_to_token_map

# Using jitter-resistant canonicalization for precision
from compare_spacy import canonicalize_boundary_index, PUNCT_CHARS

class SentenceSplitterAPI:
    def __init__(self, checkpoint_path="checkpoints/best_sentence_mlp.pt", backend="transformers", threshold=0.5):
        """
        Initializes the API by loading models into memory ONCE.
        This avoids the "cold start" (10-15 seconds of loading) on every call.
        """
        self.device = get_device()
        self.backend = backend
        self.threshold = threshold
        
        print(f"🔄 Initializing SentenceSplitterAPI on {self.device} (Backend: {backend})...")
        
        # 1. Load Qwen (LLM) and Tokenizer
        self.llm_model, self.tokenizer = load_language_model(backend=self.backend, device=self.device)
        
        # 2. Load MoE Network (MLP)
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        hidden_dim = checkpoint.get("hidden_dim", 2048)
        d_model = checkpoint.get("d_model", checkpoint.get("cnn_dim", 256))
        num_experts = checkpoint.get("num_experts", 8)
        top_k = checkpoint.get("top_k", min(2, num_experts))
        
        self.mlp = SpacePredictorMLP(
            hidden_dim=hidden_dim,
            d_model=d_model,
            dropout=0.0, # Pure inference
            num_experts=num_experts,
            top_k=top_k,
        ).to(self.device)
        
        self.mlp.load_state_dict(checkpoint["model_state_dict"])
        self.mlp.eval()
        
        print("✅ Models loaded and ready for inference.")

    def split_text(self, text: str) -> list[str]:
        """
        Takes a block of text, calculates boundaries, and returns the array of sentences.
        Optimized for texts up to ~2000-3000 characters at a time.
        """
        if not text or not text.strip():
            return []

        # 1. Tokenization and Mapping (Same logic as compare_spacy.py)
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        
        _, char_to_token = build_sentence_char_to_token_map(text, self.tokenizer)
        c2t_np = np.array(char_to_token)
        valid_char_len = len(text)
        
        # 2. Extraction and Inference
        with torch.no_grad():
            tok_emb = extract_token_embeddings(self.llm_model, input_ids, attention_mask, backend=self.backend)
            tok_emb = tok_emb.float()
            
            preds_out, _ = self.mlp(tok_emb)
            preds_prob = torch.sigmoid(preds_out).squeeze(0).cpu().numpy()
            
        # 3. Probability projection on characters and boundary search
        boundaries = []
        unique_tokens = np.unique(c2t_np)
        
        for tok_idx in unique_tokens:
            if tok_idx <= 0 or tok_idx >= len(preds_prob):
                continue # Skip special tokens (e.g., CLS)
                
            prob = preds_prob[tok_idx]
            if prob > self.threshold:
                # Find all characters mapped to this candidate token
                char_indices = np.where(c2t_np == tok_idx)[0]
                best_idx = char_indices[0]
                
                # Search for a space to cut cleanly
                for c_idx in char_indices:
                    if text[c_idx].isspace():
                        best_idx = c_idx
                        break
                        
                # Refine with jitter-resistant logic
                best_idx = int(canonicalize_boundary_index(text, best_idx, valid_char_len))
                boundaries.append(best_idx)
                
        # Remove duplicates and sort chronologically
        boundaries = sorted(list(set(boundaries)))
        
        # 4. Slice the original text using the calculated boundaries
        sentences = []
        start = 0
        for b in boundaries:
            if b > start:
                sentence = text[start:b].strip()
                if sentence:
                    sentences.append(sentence)
            start = b
            
        # Add the last remaining piece of text
        if start < len(text):
            final_sentence = text[start:].strip()
            if final_sentence:
                sentences.append(final_sentence)
                
        return sentences

    def split_document(self, document_text: str) -> list[str]:
        """
        Safety function for the hackathon.
        Splits text by paragraphs (newlines) first to prevent OOM on massive documents.
        """
        paragraphs = document_text.split('\n')
        all_sentences = []
        
        for p in paragraphs:
            if p.strip():
                # Process individual paragraph
                sents = self.split_text(p)
                all_sentences.extend(sents)
                
        return all_sentences
