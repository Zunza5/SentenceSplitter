import torch
import time
import argparse
import spacy
import nltk
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from wordSplitter.embeddings import load_language_model, extract_token_embeddings, get_device
from wordSplitter.model import SpacePredictorMLP
from data_sentence import get_sentence_dataloader

def get_spacy_model(language):
    model_name = "it_core_news_lg" if language == "italian" else "en_core_web_lg"
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"SpaCy model {model_name} not found. Downloading...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)

def evaluate_model(dataloader, llm_model, tokenizer, mlp, device, backend="mlx"):
    """Evaluate the LLM-based model at token level."""
    print(f"\n--- Running LLM Inference on {len(dataloader.dataset)} chunks ---")
    
    all_preds = []
    all_labels = []
    total_time = 0.0
    num_processed = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["token_labels"]
            mask = batch["token_mask"]
            
            start_batch = time.time()
            
            tok_emb = extract_token_embeddings(llm_model, input_ids, attention_mask, backend=backend)
            tok_emb = tok_emb.float()
            
            # Fix Prompt Shift: filter each sample in batch to exclude prompt/padding
            batch_probs = []
            for b in range(tok_emb.shape[0]):
                valid_mask = mask[b]
                if not valid_mask.any():
                    batch_probs.append(torch.zeros(0, device=device))
                    continue
                
                # Extract only text tokens for this sample
                sample_text_emb = tok_emb[b, valid_mask, :].unsqueeze(0) # (1, num_text_tokens, hidden_dim)
                
                # Predict
                probs, _ = mlp(sample_text_emb)
                batch_probs.append(probs.squeeze(0))
            
            end_batch = time.time()
            total_time += (end_batch - start_batch)
            
            # Collect predictions
            for b in range(len(batch_probs)):
                p = (batch_probs[b].cpu() > 0.5).int().tolist()
                valid_mask = mask[b]
                l = labels[b, valid_mask].int().tolist()
                
                all_preds.extend(p)
                all_labels.extend(l)
                num_processed += 1
                
            if (i + 1) % 10 == 0:
                print(f" LLM: Batch {i+1}/{len(dataloader)} processed...")

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_time": total_time,
        "num_processed": num_processed
    }

def _build_char_to_token_map(text, tokenizer):
    """Build a mapping from character position to token index using offset_mapping."""
    encoding = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"].squeeze(0).tolist()
    
    # Map: for each character position, which token covers it?
    char_to_tok = [-1] * len(text)
    for tok_idx, (start, end) in enumerate(offsets):
        if start == end:
            continue  # skip special tokens
        for c in range(start, min(end, len(text))):
            char_to_tok[c] = tok_idx
    
    # Build a set of "valid" token indices (those that are not special tokens)
    # This matches the token_mask logic in data_sentence.py
    valid_tok_indices = []
    for tok_idx, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            continue
        valid_tok_indices.append(tok_idx)
    
    # Build mapping from absolute token index to position in the "valid" array.
    # This is needed because the labels array only contains valid tokens (mask=True).
    tok_to_valid_idx = {}
    for valid_pos, tok_idx in enumerate(valid_tok_indices):
        tok_to_valid_idx[tok_idx] = valid_pos
    
    return char_to_tok, tok_to_valid_idx


def _char_boundaries_to_token_preds(boundary_chars, text, tokenizer, num_valid_tokens):
    """Convert char-level boundary positions to token-level predictions using tokenizer offsets."""
    char_to_tok, tok_to_valid = _build_char_to_token_map(text, tokenizer)
    
    p = [0] * num_valid_tokens
    for bc in boundary_chars:
        # Try the exact char position first, then nearby chars
        tok_idx = -1
        for offset in [0, -1, 1, -2, 2]:
            pos = bc + offset
            if 0 <= pos < len(text) and char_to_tok[pos] != -1:
                tok_idx = char_to_tok[pos]
                break
        
        if tok_idx != -1 and tok_idx in tok_to_valid:
            valid_idx = tok_to_valid[tok_idx]
            if 0 <= valid_idx < num_valid_tokens:
                p[valid_idx] = 1
    
    return p


def evaluate_spacy(dataloader, nlp_model, tokenizer=None):
    """Evaluate SpaCy at token-level using proper offset_mapping alignment."""
    print(f"\n--- Running SpaCy Inference on {len(dataloader.dataset)} chunks ---")
    
    all_preds = []
    all_labels = []
    total_time = 0.0
    num_processed = 0
    
    for i, batch in enumerate(dataloader):
        texts = batch["text"]
        token_labels = batch["token_labels"]
        token_mask = batch["token_mask"]
        
        start_batch = time.time()
        
        for b in range(len(texts)):
            text = texts[b]
            valid = token_mask[b]
            l = token_labels[b][valid].int().tolist()
            
            # Run SpaCy on the original text
            doc = nlp_model(text)
            
            # Collect SpaCy boundary char positions
            spacy_boundary_chars = set()
            sents = list(doc.sents)
            for sent_idx, sent in enumerate(sents[:-1]):
                bc = sent.end_char
                if bc < len(text) and text[bc] == " ":
                    spacy_boundary_chars.add(bc)
                elif bc > 0:
                    spacy_boundary_chars.add(bc - 1)
            
            # Map to token-level using proper offset_mapping
            num_tokens = len(l)
            if tokenizer is not None:
                p = _char_boundaries_to_token_preds(spacy_boundary_chars, text, tokenizer, num_tokens)
            else:
                # Fallback (should not happen)
                p = [0] * num_tokens
            
            all_preds.extend(p)
            all_labels.extend(l)
            num_processed += 1
            
        end_batch = time.time()
        total_time += (end_batch - start_batch)
            
        if (i + 1) % 10 == 0:
            print(f" SpaCy: Batch {i+1}/{len(dataloader)} processed...")

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_time": total_time,
        "num_processed": num_processed
    }

def evaluate_nltk(dataloader, language="italian", tokenizer=None):
    """Evaluate NLTK at token-level using proper offset_mapping alignment."""
    print(f"\n--- Running NLTK Inference on {len(dataloader.dataset)} chunks ---")
    
    all_preds = []
    all_labels = []
    total_time = 0.0
    num_processed = 0
    
    for i, batch in enumerate(dataloader):
        texts = batch["text"]
        token_labels = batch["token_labels"]
        token_mask = batch["token_mask"]
        
        start_batch = time.time()
        
        for b in range(len(texts)):
            text = texts[b]
            valid = token_mask[b]
            l = token_labels[b][valid].int().tolist()
            
            # Run NLTK
            sentences = nltk.sent_tokenize(text, language=language)
            
            # Build char-level boundary positions
            nltk_boundary_chars = set()
            current_pos = 0
            for sent_idx, sent_text in enumerate(sentences[:-1]):
                idx = text.find(sent_text, current_pos)
                if idx != -1:
                    boundary_idx = idx + len(sent_text)
                    if boundary_idx < len(text):
                        if text[boundary_idx] == " ":
                            nltk_boundary_chars.add(boundary_idx)
                        elif boundary_idx > 0:
                            nltk_boundary_chars.add(boundary_idx - 1)
                    current_pos = boundary_idx
            
            # Map to token-level using proper offset_mapping
            num_tokens = len(l)
            if tokenizer is not None:
                p = _char_boundaries_to_token_preds(nltk_boundary_chars, text, tokenizer, num_tokens)
            else:
                p = [0] * num_tokens
            
            all_preds.extend(p)
            all_labels.extend(l)
            num_processed += 1
            
        end_batch = time.time()
        total_time += (end_batch - start_batch)
            
        if (i + 1) % 10 == 0:
            print(f" NLTK: Batch {i+1}/{len(dataloader)} processed...")

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_time": total_time,
        "num_processed": num_processed
    }

def print_comparison(spacy_res, nltk_res, llm_res):
    print("\n" + "="*80)
    print("                              PERFORMANCE COMPARISON                              ")
    print("="*80)
    print(f"{'Metric':<20} | {'SpaCy':<15} | {'NLTK':<15} | {'LLM (MLX)':<15}")
    print("-" * 80)
    print(f"{'Chunks Processed':<20} | {spacy_res['num_processed']:<15} | {nltk_res['num_processed']:<15} | {llm_res['num_processed']:<15}")
    print(f"{'Accuracy':<20} | {spacy_res['accuracy']:<15.4f} | {nltk_res['accuracy']:<15.4f} | {llm_res['accuracy']:<15.4f}")
    print(f"{'Precision':<20} | {spacy_res['precision']:<15.4f} | {nltk_res['precision']:<15.4f} | {llm_res['precision']:<15.4f}")
    print(f"{'Recall':<20} | {spacy_res['recall']:<15.4f} | {nltk_res['recall']:<15.4f} | {llm_res['recall']:<15.4f}")
    print(f"{'F1 Score':<20} | {spacy_res['f1']:<15.4f} | {nltk_res['f1']:<15.4f} | {llm_res['f1']:<15.4f}")
    print("-" * 80)
    
    spacy_time = spacy_res['total_time']
    nltk_time = nltk_res['total_time']
    min_time = llm_res['total_time']
    spacy_avg = (spacy_time / spacy_res['num_processed']) * 1000
    nltk_avg = (nltk_time / nltk_res['num_processed']) * 1000
    llm_avg = (min_time / llm_res['num_processed']) * 1000
    
    print(f"{'Total Time (s)':<20} | {spacy_time:<15.4f} | {nltk_time:<15.4f} | {min_time:<15.4f}")
    print(f"{'Avg Time/Chunk (ms)':<20} | {spacy_avg:<15.2f} | {nltk_avg:<15.2f} | {llm_avg:<15.2f}")
    print("="*80)

def plot_combined_results(results):
    SPLIT_NAMES = {
        "it-isdt-test": "ISDT (IT)",
        "it-postwita-test": "PoSTWITA (IT)",
        "it-vit-test": "VIT (IT)",
        "it-twittiro-test": "TWITTIRO (IT)",
        "it-partut-test": "ParTUT (IT)",
        "it-markit-test": "MarkIT (IT)",
        "en-ewt-test": "EWT (EN)",
        "en-gum-test": "GUM (EN)",
        "en-pud-test": "PUD (EN)",
        "en-partut-test": "ParTUT (EN)"
    }
    splits = [SPLIT_NAMES.get(r[0], r[0]) for r in results]
    spacy_f1 = [r[1]['f1'] for r in results]
    nltk_f1 = [r[2]['f1'] for r in results]
    llm_f1 = [r[3]['f1'] for r in results]
    
    x = np.arange(len(splits))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, spacy_f1, width, label='SpaCy', color='#1f77b4')
    rects2 = ax.bar(x, nltk_f1, width, label='NLTK', color='#2ca02c')
    rects3 = ax.bar(x + width, llm_f1, width, label='LLM (MLX)', color='#ff7f0e')
    
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison by Test Split')
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=45, ha='right')
    ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    plt.savefig("f1_comparison.png", dpi=300)
    print("\nSaved F1 comparison graph to f1_comparison.png")

def plot_time_comparison(results):
    SPLIT_NAMES = {
        "it-isdt-test": "ISDT (IT)",
        "it-postwita-test": "PoSTWITA (IT)",
        "it-vit-test": "VIT (IT)",
        "it-twittiro-test": "TWITTIRO (IT)",
        "it-partut-test": "ParTUT (IT)",
        "it-markit-test": "MarkIT (IT)",
        "en-ewt-test": "EWT (EN)",
        "en-gum-test": "GUM (EN)",
        "en-pud-test": "PUD (EN)",
        "en-partut-test": "ParTUT (EN)"
    }
    splits = [SPLIT_NAMES.get(r[0], r[0]) for r in results]
    
    spacy_time = [(r[1]['total_time'] / max(r[1]['num_processed'], 1)) * 1000 for r in results]
    nltk_time = [(r[2]['total_time'] / max(r[2]['num_processed'], 1)) * 1000 for r in results]
    llm_time = [(r[3]['total_time'] / max(r[3]['num_processed'], 1)) * 1000 for r in results]
    
    x = np.arange(len(splits))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, spacy_time, width, label='SpaCy', color='#1f77b4')
    rects2 = ax.bar(x, nltk_time, width, label='NLTK', color='#2ca02c')
    rects3 = ax.bar(x + width, llm_time, width, label='LLM (MLX)', color='#ff7f0e')
    
    ax.set_ylabel('Avg Time/Chunk (ms)')
    ax.set_title('Inference Speed Comparison by Test Split')
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=45, ha='right')
    ax.legend(loc='upper right')
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    plt.savefig("time_comparison.png", dpi=300)
    print("\nSaved time comparison graph to time_comparison.png")

def main():
    parser = argparse.ArgumentParser(description="Compare SpaCy and LLM Performance")
    parser.add_argument("--test-splits", type=str, default="it-isdt-test,it-postwita-test,it-vit-test,it-twittiro-test,it-partut-test,it-markit-test,en-ewt-test,en-gum-test,en-pud-test,en-partut-test", help="Comma-separated test splits")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for LLM inference")
    parser.add_argument("--max-chars", type=int, default=1024)
    parser.add_argument("--stride-chars", type=int, default=512, help="Set to < max-chars to enable overlapping window averaging")
    args = parser.parse_args()
    
    device = get_device()
    backend = "mlx"
    
    print("Loading LLM model and tokenizer...")
    llm_model, tokenizer = load_language_model(backend=backend, device=device)
    
    print("\nInitializing NLTK...")
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("NLTK tokenizer data not found. Downloading 'punkt' and 'punkt_tab'...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    
    print("\nLoading LLM MLP...")
    checkpoint_path = Path("checkpoints/best_sentence_mlp.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hidden_dim = checkpoint.get("hidden_dim", 2560)
    d_model = checkpoint.get("d_model", 512)
    dropout = checkpoint.get("dropout", 0.2)
    mlp = SpacePredictorMLP(hidden_dim=hidden_dim, d_model=d_model, dropout=dropout).to(device)
    mlp.load_state_dict(checkpoint["model_state_dict"])
    mlp.eval()
    
    test_splits = [s.strip() for s in args.test_splits.split(",")]
    all_results = []
    loaded_spacy = {}
    
    for split in test_splits:
        language = "italian" if "it" in split else "english"
        
        print(f"\n{'='*60}")
        print(f" evaluating split: {split} (Language: {language})")
        print(f"{'='*60}")
        try:
            if language not in loaded_spacy:
                print(f"Loading SpaCy model for {language}...")
                loaded_spacy[language] = get_spacy_model(language)
            nlp = loaded_spacy[language]
            dataloader = get_sentence_dataloader(
                split=split,
                batch_size=args.batch_size,
                tokenizer=tokenizer,
                max_chars=args.max_chars, 
                stride_chars=args.stride_chars,
                augment_prob=0.0,
                augmentation_mode="original"
            )
            
            spacy_results = evaluate_spacy(dataloader, nlp, tokenizer=tokenizer)
            nltk_results = evaluate_nltk(dataloader, language, tokenizer=tokenizer)
            llm_results = evaluate_model(dataloader, llm_model, tokenizer, mlp, device, backend=backend)
            
            print_comparison(spacy_results, nltk_results, llm_results)
            all_results.append((split, spacy_results, nltk_results, llm_results))
        except Exception as e:
            print(f"Failed to evaluate split {split}: {e}")
            
    if all_results:
        plot_combined_results(all_results)
        plot_time_comparison(all_results)

if __name__ == "__main__":
    main()
