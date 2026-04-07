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
    def __init__(self, checkpoint_path="checkpoints/best_sentence_mlp.pt", backend="transformers", threshold=0.5, max_chars=1024, stride_chars=512, batch_size=8):
        """
        Initializes the API by loading models into memory ONCE.
        This version supports Sliding Window aggregation with batching for maximum throughput.
        """
        self.device = get_device()
        self.backend = backend
        self.threshold = threshold
        self.max_chars = max_chars
        self.stride_chars = stride_chars
        self.batch_size = batch_size
        
        print(f"🔄 Initializing SentenceSplitterAPI on {self.device} (Backend: {backend}, Window: {max_chars}/{stride_chars}, Batch: {batch_size})...")
        
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

    def get_boundaries(self, text: str) -> list[int]:
        """
        Core method: returns the character indices where sentence boundaries were detected.
        Uses Sliding Window with Binary Voting aggregation.
        """
        if not text or not text.strip():
            return []

        total_len = len(text)
        full_preds_sum = np.zeros(total_len, dtype=np.float32)
        full_preds_count = np.zeros(total_len, dtype=np.int32)
        
        # 1. Prepare overlapping chunks
        chunks = []
        start = 0
        while start < total_len:
            end = min(start + self.max_chars, total_len)
            chunks.append((start, end, text[start:end]))
            if end == total_len: break
            start += self.stride_chars

        # 2. Process chunks in batches
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_texts = [c[2] for c in batch_chunks]
                
            enc = self.tokenizer(batch_texts, return_tensors="pt", add_special_tokens=True, padding=True)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            
            with torch.no_grad():
                tok_emb = extract_token_embeddings(self.llm_model, input_ids, attention_mask, backend=self.backend)
                preds_out, _ = self.mlp(tok_emb.float())
                preds_prob = torch.sigmoid(preds_out).cpu().numpy()
            
            for b_idx, (start_idx, end_idx, chunk_text) in enumerate(batch_chunks):
                _, char_to_token = build_sentence_char_to_token_map(chunk_text, self.tokenizer)
                c2t_np = np.array(char_to_token)
                p_chunk_sparse = np.zeros(len(chunk_text), dtype=np.float32)
                unique_tokens = np.unique(c2t_np)
                chunk_preds = preds_prob[b_idx]
                
                for tok_idx in unique_tokens:
                    if tok_idx <= 0 or tok_idx >= len(chunk_preds): continue
                    
                    if chunk_preds[tok_idx] > self.threshold:
                        char_indices = np.where(c2t_np == tok_idx)[0]
                        best_idx = char_indices[0]
                        for c_idx in char_indices:
                            if chunk_text[c_idx].isspace():
                                best_idx = c_idx
                                break
                        
                        best_idx = int(canonicalize_boundary_index(chunk_text, best_idx, len(chunk_text)))
                        p_chunk_sparse[best_idx] = 1.0

                full_preds_sum[start_idx:end_idx] += p_chunk_sparse
                full_preds_count[start_idx:end_idx] += 1

        # 3. Final Consensus (Binary Agreement)
        boundaries = []
        for j in range(total_len):
            if full_preds_count[j] > 0:
                avg_p = full_preds_sum[j] / full_preds_count[j]
                if avg_p >= 0.5:
                    boundaries.append(j)

        return sorted(set(boundaries))

    def split_document(self, text: str) -> list[str]:
        """
        Splits a document into sentences using the Sliding Window pipeline.
        Delegates boundary detection to get_boundaries().
        """
        if not text or not text.strip():
            return []

        boundaries = self.get_boundaries(text)
        
        sentences = []
        last_break = 0
        for b in boundaries:
            if b > last_break:
                piece = text[last_break:b].strip()
                if piece: sentences.append(piece)
            last_break = b
            
        final_piece = text[last_break:].strip()
        if final_piece: sentences.append(final_piece)
        
        return sentences

    def split_text(self, text: str) -> list[str]:
        """Alias for split_document."""
        return self.split_document(text)
