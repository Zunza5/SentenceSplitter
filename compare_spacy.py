import torch
import gc
import time
import argparse
import spacy
import nltk
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from sentence_embeddings import load_language_model, extract_token_embeddings, get_device
from model import SpacePredictorMLP
from data_sentence import get_sentence_dataloader, UD_URLS


ALL_TEST_SPLITS = ",".join(sorted(s for s in UD_URLS if s.endswith("-test")))
PUNCT_CHARS = set(".!?;:)]}")


def _valid_token_offsets(tokenizer, text: str):
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    tok_offsets = enc["offset_mapping"].squeeze(0).tolist()
    return [
        (start, end)
        for start, end in tok_offsets
        if not (start == 0 and end == 0) and end > start and end <= len(text)
    ]


def _resolve_char_target_idx(local_text: str, local_start: int, local_end: int, valid_char_len: int):
    """Map token span to a character index compatible with sentence-boundary labels."""
    if valid_char_len <= 0:
        return None

    # With current labeling, sentence boundaries are on the separating space char.
    # If tokenization merges leading space with the following token, use that space.
    span_end = min(local_end, valid_char_len)
    if 0 <= local_start < span_end:
        for idx in range(local_start, span_end):
            if local_text[idx].isspace():
                return idx

    prev_idx = local_end - 1
    end_in_bounds = 0 <= local_end < valid_char_len
    prev_in_bounds = 0 <= prev_idx < valid_char_len

    # Prefer the boundary space position when available (dataset label semantics).
    if end_in_bounds and local_text[local_end].isspace():
        return local_end

    # Punctuation-attached boundaries can be one char before an exclusive end.
    if prev_in_bounds and local_text[prev_idx] in PUNCT_CHARS:
        return prev_idx

    if end_in_bounds:
        return local_end
    if prev_in_bounds:
        return prev_idx
    return None

def get_spacy_model(language):
    model_name = "it_core_news_lg" if language == "italian" else "en_core_web_lg"
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"SpaCy model {model_name} not found. Downloading...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)

def evaluate_model(dataloader, llm_model, tokenizer, mlp, device, backend="mlx", threshold=0.5):
    print(f"\n--- Running LLM Inference on {len(dataloader.dataset)} chunks ---")
    
    # Pre-determine total length to initialize buffers
    total_text_len = 0
    for sample in dataloader.dataset:
        total_text_len = max(total_text_len, sample["char_offset"] + len(sample["spaceless"]))
    
    # Use NumPy arrays for massive memory efficiency over Python lists
    full_probs_max = np.zeros(total_text_len, dtype=np.float32)
    full_labels = np.zeros(total_text_len, dtype=np.int16)
    label_filled = np.zeros(total_text_len, dtype=bool)
    
    total_time = 0.0
    time_llm = 0.0
    time_sync = 0.0
    time_post = 0.0
    num_processed = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_mask = batch["token_mask"].to(device)
            labels = batch["char_labels"]
            mask = batch["char_mask"]
            offsets = batch["char_offset"]
            texts = batch["spaceless"]
            
            start_batch = time.time()
            
            tok_emb = extract_token_embeddings(llm_model, input_ids, attention_mask, backend=backend)
            tok_emb = tok_emb.float()
            preds, _ = mlp(tok_emb, mask=token_mask)
            
            end_llm = time.time()
            time_llm += (end_llm - start_batch)
            
            # Explicitly delete to free memory earlier
            del tok_emb
            
            # Move to CPU ONCE to avoid O(BatchSize) sync overheads on MPS
            preds_cpu = preds.cpu()
            mask_cpu = mask.cpu()
            labels_cpu = labels.cpu()
            token_mask_cpu = token_mask.cpu()
            
            end_sync = time.time()
            time_sync += (end_sync - end_llm)
            
            for b in range(preds.shape[0]):
                offset = int(offsets[b])
                valid_char_len = int(mask_cpu[b].sum().item())
                l_list = labels_cpu[b][:valid_char_len].int().tolist()

                local_text = texts[b]
                valid_tok = token_mask_cpu[b].bool()
                p_tok = preds_cpu[b][valid_tok].tolist()
                tok_offsets = _valid_token_offsets(tokenizer, local_text)
                valid_tok_len = min(len(p_tok), len(tok_offsets))

                # Project token probabilities to characters
                p_char = [0.0] * valid_char_len
                for j in range(valid_tok_len):
                    local_start, local_end = tok_offsets[j]
                    target_idx = _resolve_char_target_idx(local_text, local_start, local_end, valid_char_len)
                    if target_idx is not None:
                        p_char[target_idx] = max(p_char[target_idx], float(p_tok[j]))

                # Vectorize the NumPy array updates to fix the O(N) scalar slowdown
                end_idx = min(offset + valid_char_len, total_text_len)
                actual_len = end_idx - offset
                
                if actual_len > 0:
                    p_char_arr = np.array(p_char[:actual_len], dtype=np.float32)
                    full_probs_max[offset:end_idx] = np.maximum(
                        full_probs_max[offset:end_idx], p_char_arr
                    )
                    
                    slice_filled = label_filled[offset:end_idx]
                    unfilled_mask = ~slice_filled
                    
                    if np.any(unfilled_mask):
                        l_arr = np.array(l_list[:actual_len], dtype=np.int16)
                        full_labels[offset:end_idx][unfilled_mask] = l_arr[unfilled_mask]
                        label_filled[offset:end_idx] = True
                
                num_processed += 1
            
            end_post = time.time()
            time_post += (end_post - end_sync)
            total_time += (end_post - start_batch)
                
            if (i + 1) % 10 == 0:
                print(f" LLM: Batch {i+1}/{len(dataloader)} processed | LLM: {time_llm:.2f}s | Sync: {time_sync:.2f}s | Post: {time_post:.2f}s")

    # Max pooling across overlapping windows preserves sparse boundary peaks.
    final_preds = []
    final_labels = []
    for j in range(total_text_len):
        if label_filled[j] and full_labels[j] >= 0:
            max_p = full_probs_max[j]
            final_preds.append(1 if max_p > threshold else 0)
            final_labels.append(full_labels[j])

    precision, recall, f1, _ = precision_recall_fscore_support(
        final_labels, final_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(final_labels, final_preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_time": total_time,
        "num_processed": num_processed
    }

def evaluate_spacy(dataloader, nlp_model):
    print(f"\n--- Running SpaCy Inference on {len(dataloader.dataset)} chunks ---")
    
    total_text_len = 0
    for sample in dataloader.dataset:
        total_text_len = max(total_text_len, sample["char_offset"] + len(sample["spaceless"]))
        
    # Use NumPy arrays for large text buffers
    full_preds_sum = np.zeros(total_text_len, dtype=np.float32)
    full_preds_count = np.zeros(total_text_len, dtype=np.int32)
    full_labels = np.zeros(total_text_len, dtype=np.int16)
    label_filled = np.zeros(total_text_len, dtype=bool)
    
    total_time = 0.0
    num_processed = 0
    
    for i, batch in enumerate(dataloader):
        labels = batch["char_labels"]
        mask = batch["char_mask"]
        texts = batch["spaceless"]
        offsets = batch["char_offset"]
        
        start_batch = time.time()
        
        for b in range(len(texts)):
            text = texts[b]
            offset = int(offsets[b])
            valid_len = int(mask[b].sum().item())
            l_list = labels[b][:valid_len].int().tolist()
            
            # Run SpaCy
            doc = nlp_model(text)
            p = [0] * valid_len
            
            sents_list = list(doc.sents)
            num_sents = len(sents_list)
            for sent_idx, sent in enumerate(sents_list):
                if sent_idx < num_sents - 1:
                    boundary_idx = sent.end_char
                    if boundary_idx < valid_len:
                        if text[boundary_idx] == " ":
                            p[boundary_idx] = 1
                        elif boundary_idx > 0 and text[boundary_idx - 1] == " ":
                            p[boundary_idx - 1] = 1
                        else:
                            p[boundary_idx] = 1 
                            
            # Accumulate char-level predictions directly on text positions.
            for j in range(valid_len):
                idx = offset + j
                if idx < total_text_len:
                    full_preds_sum[idx] += p[j]
                    full_preds_count[idx] += 1
                    if not label_filled[idx]:
                        full_labels[idx] = l_list[j]
                        label_filled[idx] = True
            
            num_processed += 1
            
        end_batch = time.time()
        total_time += (end_batch - start_batch)
            
        if (i + 1) % 10 == 0:
            print(f" SpaCy: Batch {i+1}/{len(dataloader)} processed...")

    final_preds = []
    final_labels = []
    for j in range(total_text_len):
        if label_filled[j] and full_labels[j] >= 0:
            avg_p = full_preds_sum[j] / max(1, full_preds_count[j])
            final_preds.append(1 if avg_p >= 0.5 else 0)
            final_labels.append(full_labels[j])

    precision, recall, f1, _ = precision_recall_fscore_support(
        final_labels, final_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(final_labels, final_preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_time": total_time,
        "num_processed": num_processed
    }

def evaluate_nltk(dataloader, language="italian"):
    print(f"\n--- Running NLTK Inference on {len(dataloader.dataset)} chunks ---")
    
    total_text_len = 0
    for sample in dataloader.dataset:
        total_text_len = max(total_text_len, sample["char_offset"] + len(sample["spaceless"]))
        
    # Use NumPy arrays for large text buffers
    full_preds_sum = np.zeros(total_text_len, dtype=np.float32)
    full_preds_count = np.zeros(total_text_len, dtype=np.int32)
    full_labels = np.zeros(total_text_len, dtype=np.int16)
    label_filled = np.zeros(total_text_len, dtype=bool)
    
    total_time = 0.0
    num_processed = 0

    try:
        punkt_tokenizer = nltk.data.load(f"tokenizers/punkt/{language}.pickle")
    except LookupError:
        punkt_tokenizer = None
    
    for i, batch in enumerate(dataloader):
        labels = batch["char_labels"]
        mask = batch["char_mask"]
        texts = batch["spaceless"]
        offsets = batch["char_offset"]
        
        start_batch = time.time()
        
        for b in range(len(texts)):
            text = texts[b]
            offset = int(offsets[b])
            valid_len = int(mask[b].sum().item())
            l_list = labels[b][:valid_len].int().tolist()
            
            # Run NLTK
            p = [0] * valid_len

            if punkt_tokenizer is not None:
                sent_spans = list(punkt_tokenizer.span_tokenize(text))
                for _, sent_end in sent_spans[:-1]:
                    boundary_idx = sent_end
                    if boundary_idx < valid_len:
                        if text[boundary_idx] == " ":
                            p[boundary_idx] = 1
                        elif boundary_idx > 0 and text[boundary_idx - 1] == " ":
                            p[boundary_idx - 1] = 1
                        else:
                            p[boundary_idx] = 1
            else:
                sentences = nltk.sent_tokenize(text, language=language)
                current_pos = 0
                for sent_idx, sent_text in enumerate(sentences):
                    if sent_idx < len(sentences) - 1:
                        idx = text.find(sent_text, current_pos)
                        if idx != -1:
                            boundary_idx = idx + len(sent_text)
                            if boundary_idx < valid_len:
                                if text[boundary_idx] == " ":
                                    p[boundary_idx] = 1
                                elif boundary_idx > 0 and text[boundary_idx - 1] == " ":
                                    p[boundary_idx - 1] = 1
                                else:
                                    p[boundary_idx] = 1
                            current_pos = boundary_idx

            # Accumulate char-level predictions directly on text positions.
            for j in range(valid_len):
                idx = offset + j
                if idx < total_text_len:
                    full_preds_sum[idx] += p[j]
                    full_preds_count[idx] += 1
                    if not label_filled[idx]:
                        full_labels[idx] = l_list[j]
                        label_filled[idx] = True
            
            num_processed += 1
            
        end_batch = time.time()
        total_time += (end_batch - start_batch)
            
        if (i + 1) % 10 == 0:
            print(f" NLTK: Batch {i+1}/{len(dataloader)} processed...")

    final_preds = []
    final_labels = []
    for j in range(total_text_len):
        if label_filled[j] and full_labels[j] >= 0:
            avg_p = full_preds_sum[j] / max(1, full_preds_count[j])
            final_preds.append(1 if avg_p >= 0.5 else 0)
            final_labels.append(full_labels[j])

    precision, recall, f1, _ = precision_recall_fscore_support(
        final_labels, final_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(final_labels, final_preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_time": total_time,
        "num_processed": num_processed
    }

def print_comparison(spacy_res, nltk_res, llm_res, llm_label="LLM"):
    print("\n" + "="*80)
    print("                              PERFORMANCE COMPARISON                              ")
    print("="*80)
    print(f"{'Metric':<20} | {'SpaCy':<15} | {'NLTK':<15} | {llm_label:<15}")
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
    parser.add_argument("--test-splits", type=str, default=ALL_TEST_SPLITS, help="Comma-separated test splits")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for LLM inference")
    parser.add_argument("--max-chars", type=int, default=1024, help="Max characters per chunk for LLM inference")
    parser.add_argument("--stride-chars", type=int, default=512, help="Set to < max-chars to enable overlapping window averaging")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for LLM token boundaries")
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "mlx"], help="Embedding backend used by the LLM")
    args = parser.parse_args()
    
    device = get_device()
    backend = args.backend
    
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
    hidden_dim = checkpoint.get("hidden_dim", 2048)
    d_model = checkpoint.get("d_model", checkpoint.get("cnn_dim", 256))
    num_experts = checkpoint.get("num_experts", 8)
    top_k = checkpoint.get("top_k", min(2, num_experts))
    mlp = SpacePredictorMLP(
        hidden_dim=hidden_dim,
        d_model=d_model,
        dropout=0.0,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)
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
            # Only keep the current language model for SpaCy to save memory
            if language not in loaded_spacy:
                # Clear other SpaCy models first
                for old_lang in list(loaded_spacy.keys()):
                    if old_lang != language:
                        print(f"Unloading SpaCy model for {old_lang}...")
                        del loaded_spacy[old_lang]
                
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
            
            spacy_results = evaluate_spacy(dataloader, nlp)
            nltk_results = evaluate_nltk(dataloader, language)
            llm_results = evaluate_model(
                dataloader,
                llm_model,
                tokenizer,
                mlp,
                device,
                backend=backend,
                threshold=args.threshold,
            )
            
            print_comparison(spacy_results, nltk_results, llm_results, llm_label=f"LLM ({backend})")
            all_results.append((split, spacy_results, nltk_results, llm_results))

            # Manual cleanup to free memory between datasets
            del dataloader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Failed to evaluate split {split}: {e}")
            
    if all_results:
        plot_combined_results(all_results)
        plot_time_comparison(all_results)

if __name__ == "__main__":
    main()
