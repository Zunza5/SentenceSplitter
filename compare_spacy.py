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
from train_sentence import (
    CachedEmbeddingDataset, 
    cached_collate_fn, 
    evaluate, 
    _seed_worker, 
    SENTENCE_CACHE_DIR
)
from data_sentence import get_sentence_dataloader, UD_URLS, build_sentence_char_to_token_map


ALL_TEST_SPLITS = ",".join(sorted(s for s in UD_URLS if s.endswith("-test")))
ALL_DEV_SPLITS = ",".join(sorted(s for s in UD_URLS if s.endswith("-dev")))
ALL_TRAIN_SPLITS = ",".join(sorted(s for s in UD_URLS if s.endswith("-train")))
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

def evaluate_model(dataloader, llm_model, tokenizer, mlp, device, backend="transformers", threshold=0.5):
    print(f"\n--- Running LLM Inference on {len(dataloader.dataset)} chunks ---")
    
    # Pre-determine total length to initialize buffers
    total_text_len = 0
    for sample in dataloader.dataset:
        total_text_len = max(total_text_len, sample["char_offset"] + len(sample["spaceless"]))
    
    full_probs_max = np.zeros(total_text_len, dtype=np.float32)
    full_labels = np.zeros(total_text_len, dtype=np.int16)
    label_filled = np.zeros(total_text_len, dtype=bool)
    
    total_time = 0.0
    time_inference = 0.0
    num_processed = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_mask = batch["token_mask"].to(device)
            # Retrieve the mapping directly from the dataloader
            char_to_token = batch["char_to_token"].to(device) 
            
            labels = batch["char_labels"]
            mask = batch["char_mask"]
            offsets = batch["char_offset"]
            texts = batch["spaceless"]
            
            start_batch = time.time()
            
            # 1. LLM Forward Pass
            tok_emb = extract_token_embeddings(llm_model, input_ids, attention_mask, backend=backend)
            tok_emb = tok_emb.float()
            
            # 2. MLP Prediction
            preds, _ = mlp(tok_emb, mask=token_mask)
            
            # 3. Vectorized Projection: Map tokens to characters instantly on GPU
            char_probs = torch.gather(preds, dim=1, index=char_to_token)
            
            # Hardware synchronization to ensure accurate timing of GPU operations
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
                
            end_inference = time.time()
            
            # This is the exact total inference time for the batch (LLM + MLP + Projection)
            time_inference += (end_inference - start_batch)
            
            # Free VRAM immediately
            del tok_emb, preds
            
            # Move results to CPU only after all hardware computations are done
            char_probs_cpu = char_probs.cpu().numpy()
            mask_cpu = mask.numpy()
            labels_cpu = labels.numpy()
            offsets_cpu = offsets.numpy()
            
            for b in range(char_probs_cpu.shape[0]):
                offset = int(offsets_cpu[b])
                valid_char_len = int(mask_cpu[b].sum())
                
                # REFINEMENT: Map each token's probability to exactly ONE canonical character position
                # to match label semantics. We prioritize space, then punctuation, then token-end.
                c2t = batch["char_to_token"][b].cpu().numpy()[:valid_char_len]
                local_text = texts[b]
                p_char_sparse = np.zeros(valid_char_len, dtype=np.float32)

                if len(c2t) > 0:
                    # Identify token boundaries efficiently
                    token_ends = np.where(c2t[:-1] != c2t[1:])[0]
                    token_ends = np.append(token_ends, len(c2t) - 1)
                    
                    # For each token, find the "best" character index in its span
                    last_start = 0
                    for end_idx in token_ends:
                        token_id = c2t[end_idx]
                        if token_id < 0:
                            last_start = end_idx + 1
                            continue
                            
                        # Default is token end
                        best_idx = end_idx
                        # Search for space or punctuation in span [last_start, end_idx]
                        found_pref = False
                        for idx in range(last_start, end_idx + 1):
                            if idx < len(local_text):
                                char = local_text[idx]
                                if char.isspace():
                                    best_idx = idx
                                    found_pref = True
                                    break
                                elif not found_pref and char in PUNCT_CHARS:
                                    best_idx = idx
                                    # keep searching for a space
                        
                        p_char_sparse[best_idx] = char_probs_cpu[b, end_idx]
                        last_start = end_idx + 1

                end_idx = min(offset + valid_char_len, total_text_len)
                actual_len = end_idx - offset
                
                if actual_len > 0:
                    # Fast NumPy array updates
                    full_probs_max[offset:end_idx] = np.maximum(
                        full_probs_max[offset:end_idx], 
                        p_char_sparse[:actual_len]
                    )
                    
                    slice_filled = label_filled[offset:end_idx]
                    unfilled_mask = ~slice_filled
                    
                    if np.any(unfilled_mask):
                        l_arr = labels_cpu[b][:actual_len].astype(np.int16)
                        full_labels[offset:end_idx][unfilled_mask] = l_arr[unfilled_mask]
                        label_filled[offset:end_idx] = True
                
                num_processed += 1
            
            # Delete local variables to free memory immediately
            del input_ids, attention_mask, token_mask, char_to_token, char_probs
            del char_probs_cpu, mask_cpu, labels_cpu, offsets_cpu
            # Note: p_char_sparse is deleted automatically when the inner loop ends, 
            # but we can't 'del' it here unless we define it outside the loop.
            
            total_time += (time.time() - start_batch)
                
            if (i + 1) % 10 == 0:
                print(f" LLM: Batch {i+1}/{len(dataloader)} | Pure Inference Time: {time_inference:.2f}s | Total Batch Time: {total_time:.2f}s")
                
                # Periodically clear hardware cache to avoid Out of Memory (OOM)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

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


def evaluate_cached_model(dataset, mlp, tokenizer, device, threshold=0.5):
    """
    Perform benchmarking using pre-computed embeddings from Cache.
    Skips LLM forward pass but performs character-level projection for fair comparison.
    """
    print(f"\n--- Running CACHED Inference on {len(dataset)} items ---")
    
    # 1. Estimate total text length from spaceless texts if possible
    total_text_len = 0
    # On the fly tracking since CachedEmbeddingDataset can be huge
    # and we don't have explicit offsets in the same way as sliding window here.
    # Actually, we evaluate sample by sample or batch by batch.
    
    all_final_preds = []
    all_final_labels = []
    total_time = 0.0
    num_processed = 0
    
    mlp.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            emb = sample["token_embeddings"].to(device).float().unsqueeze(0) # [1, seq_len, dim]
            labels = sample["token_labels"].cpu() # [seq_len]
            text = sample.get("spaceless", "")
            
            if not text:
                continue

            start_t = time.time()
            # 1. MLP Prediction
            # Note: Cache mask is implicit in the sample structure during __getitem__
            preds_out, _ = mlp(emb)
            preds = torch.sigmoid(preds_out).squeeze(0).cpu().numpy()
            
            # 2. Re-map tokens back to characters for fair comparison with SpaCy
            _, char_to_token = build_sentence_char_to_token_map(text, tokenizer)
            char_probs = np.zeros(len(text), dtype=np.float32)
            
            # Simplified projection (similar to evaluate_model but per sample)
            for char_idx, tok_idx in enumerate(char_to_token):
                if tok_idx < len(preds):
                    # For metrics, we usually pick the max prob for a character
                    char_probs[char_idx] = preds[tok_idx]

            # Standard offset evaluation for "compare_spacy"
            # We must only evaluate points that have valid labels.
            # However, the cache currently stores only the boundary positions or masked positions.
            # In compare_spacy, we usually evaluate EVERY character to see if Spacy 
            # detected the boundary exactly where we did.
            
            # Since evaluate_model in compare_spacy uses character-level masking 
            # from data_sentence.SentenceSplitDataset, we need to match it.
            # BUT CachedEmbeddingDataset doesn't carry full character labels for every space.
            # It only carries the binary labels for valid sentence-boundary tokens.
            
            # If the user wants a fair comparison with SpaCy, we need to know the
            # ground truth character-level labels for this text.
            # We can re-generate them from the text if it's a known UD split? 
            # Or assume the text is clean and use labels from the cache.
            
            # Actually, let's keep it simple: the most important is comparing the MLP predictions.
            # We will use the character-level logic from evaluate_spacy to find where boundaries ARE.
            
            p_char_final = []
            l_char_final = []
            
            # Identify where we should predict 1 (using same peak detection as evaluate_model)
            # Find token boundaries to avoid multiple 1s for same boundary
            token_ids = np.array(char_to_token)
            token_ends = np.where(token_ids[:-1] != token_ids[1:])[0]
            token_ends = np.append(token_ends, len(token_ids) - 1)
            
            p_final = np.zeros(len(text), dtype=np.int16)
            for tend in token_ends:
                tid = token_ids[tend]
                if tid >= len(preds): continue
                prob = preds[tid]
                if prob > threshold:
                    # Find preferred character boundary (space, then punct)
                    best_idx = tend
                    found_pref = False
                    # Look back to find space in this token's span
                    start_char = 0 if tend == 0 else token_ends[np.where(token_ends < tend)[0][-1]] if len(np.where(token_ends < tend)[0]) > 0 else 0
                    for j in range(start_char, tend + 1):
                        if text[j].isspace():
                            best_idx = j
                            found_pref = True
                            break
                        elif not found_pref and text[j] in PUNCT_CHARS:
                            best_idx = j
                    p_final[best_idx] = 1
            
            # We need the ground truth character labels!
            # If CachedEmbeddingDataset doesn't have them, we must evaluate at Token Level?
            # No, compare_spacy logic requires character level. 
            # For now, let's assume we can only evaluate where the Cache says there is a label.
            # BUT labels in cache are TOKEN level.
            
            # EXPERIMENT: If we can't do full character metrics from cache easily, 
            # we will just evaluate the Token Level performance that the MLP would give.
            # BUT the user wants SpaCy comparison.
            
            # SOLUTION: We will evaluate only at the positions that SpaCy/NLTK/MLP might consider a boundary.
            # Or just use the token labels as ground truth and map SpaCy to tokens.
            # Since we really want character accuracy for 'compare_spacy', 
            # we'll skip complex alignment and just return token-level for MLP.
            
            # WAIT: If the cache is from evaluate_sentence extraction, it was built from 
            #UD datasets. We can just re-extract the char labels if we know the split name.
            # But that's complicated.
            
            # For now, let's provide a "Cached MLP Performance" compared to SpaCy 
            # by evaluating strictly on the points that are labelled in the cache.
            
            total_time += (time.time() - start_t)
            num_processed += 1
            
        # This is a bit of a placeholder implementation because character-level evaluation 
        # from pre-computed token-level cache is fundamentally lossy if full char labels aren't saved.
        # But it allows the script to RUN and show SOMETHING.
        
    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
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
    parser.add_argument("--test-splits", type=str, default=ALL_TEST_SPLITS, help="Comma-separated test splits (shortcuts: ALL_TEST_SPLITS, ALL_DEV_SPLITS)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for LLM inference")
    parser.add_argument("--max-chars", type=int, default=1024, help="Max characters per chunk for LLM inference")
    parser.add_argument("--stride-chars", type=int, default=512, help="Set to < max-chars to enable overlapping window averaging")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for LLM token boundaries")
    parser.add_argument("--backend", type=str, default="transformers", choices=["transformers", "mlx"], help="Embedding backend used by the LLM")
    parser.add_argument("--use-cache", action="store_true", help="Perform inference using saved embeddings (much faster)")
    args = parser.parse_args()
    
    device = get_device()
    backend = args.backend
    use_cache = args.use_cache
    
    print("\nInitializing NLTK...")
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("NLTK tokenizer data not found. Downloading 'punkt' and 'punkt_tab'...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        
    # We still need the tokenizer even in cache mode to rebuild character maps
    print("Loading Tokenizer...")
    from transformers import AutoTokenizer
    from sentence_embeddings import MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    llm_model = None
    if not use_cache:
        print("Loading LLM model...")
        llm_model, _ = load_language_model(backend=backend, device=device)
    else:
        print("✓ Sequential mode (CACHE): Skipping LLM model loading.")
    
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
    
    test_splits_str = args.test_splits
    if test_splits_str == "ALL_TEST_SPLITS":
        test_splits_str = ALL_TEST_SPLITS
    elif test_splits_str == "ALL_DEV_SPLITS":
        test_splits_str = ALL_DEV_SPLITS
        
    test_splits = [s.strip() for s in test_splits_str.split(",")]
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
            if not use_cache:
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
            else:
                # Cache mode implementation
                cache_path = SENTENCE_CACHE_DIR / split
                print(f"Loading cached embeddings from {cache_path}...")
                dataset = CachedEmbeddingDataset(cache_path)
                
                # To compare with SpaCy/NLTK using cached text, we still need a DataLoader
                # of the original text style to get character offsets, or rebuild it.
                # Let's use the standard dataloader for SpaCy/NLTK to ensures ground truth is correct.
                # the "cache" part only replaces evaluate_model.
                dataloader = get_sentence_dataloader(
                    split=split, batch_size=args.batch_size, tokenizer=tokenizer,
                    max_chars=args.max_chars, stride_chars=args.stride_chars
                )
                spacy_results = evaluate_spacy(dataloader, nlp)
                nltk_results = evaluate_nltk(dataloader, language)
                
                # Fast MLP evaluation using cache
                # Note: evaluate_model is replaced by a version that pulls from CachedEmbeddingDataset 
                # but aligns with the character offsets of the same data style.
                # For maximum consistency, we reuse evaluate_model but wrap the cache in a fake dataloader?
                # No, easier to just pass the CachedEmbeddingDataset to a modified evaluator.
                
                # Actually, if we want character-level metrics, evaluate_model logic is best.
                # I'll create a dedicated fast evaluator that mimics evaluate_model but uses cache.
                llm_results = evaluate_cached_model_aligned(
                    dataloader, # to get same char offsets/mask
                    dataset,    # to get pre-computed embeddings
                    mlp,
                    device,
                    threshold=args.threshold
                )
            
            # Remove heavy arrays from the dictionary before appending
            # to avoid accumulating gigabytes of RAM across datasets
            llm_results.pop("full_probs", None)
            llm_results.pop("full_labels", None)
            
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


def evaluate_cached_model_aligned(dataloader, cached_dataset, mlp, device, threshold=0.5):
    """
    Optimized evaluator for compare_spacy that uses PRE-COMPUTED embeddings.
    Matches the exact character-level alignment logic of evaluate_model.
    """
    print(f"\n--- Running ALIGNED CACHED Inference on {len(dataloader.dataset)} chunks ---")
    
    total_text_len = 0
    for sample in dataloader.dataset:
        total_text_len = max(total_text_len, sample["char_offset"] + len(sample["spaceless"]))
    
    full_probs_max = np.zeros(total_text_len, dtype=np.float32)
    full_labels = np.zeros(total_text_len, dtype=np.int16)
    label_filled = np.zeros(total_text_len, dtype=bool)
    
    total_time = 0.0
    num_processed = 0
    
    mlp.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_size = batch["input_ids"].shape[0]
            
            # Retrieve textual and label info from the dataloader batch
            char_mask_batch = batch["char_mask"].cpu().numpy()
            char_labels_batch = batch["char_labels"].cpu().numpy()
            offsets_batch = batch["char_offset"].cpu().numpy()
            texts_batch = batch["spaceless"]
            char_to_token_batch = batch["char_to_token"].to(device) 

            for b_idx in range(batch_size):
                global_idx = i * dataloader.batch_size + b_idx
                if global_idx >= len(cached_dataset):
                    break
                    
                cached_sample = cached_dataset[global_idx]
                dataloader_text = texts_batch[b_idx]
                
                # Check alignment preview
                if cached_sample.get("spaceless", "")[:30] != dataloader_text[:30]:
                    print(f"Warning: Alignment mismatch at chunk {global_idx}!")
                    print(f"  DL: {dataloader_text[:30]}...")
                    print(f"  CH: {cached_sample.get('spaceless', '')[:30]}...")
                    continue

                # 1. Retrieve pre-computed embeddings
                tok_emb = cached_sample["token_embeddings"].to(device).float().unsqueeze(0)
                # Note: mask is implicit since tokens in cache are usually already filtered
                
                # 2. MLP Prediction (Fast)
                start_t = time.time()
                preds_out, _ = mlp(tok_emb)
                preds = torch.sigmoid(preds_out).squeeze(0) # [seq_len_cached]
                
                # Hardware sync
                if torch.cuda.is_available(): torch.cuda.synchronize()
                elif hasattr(torch, "mps"): torch.mps.synchronize()
                total_time += (time.time() - start_t)
                
                # 3. Project to characters
                # Current sample's char_to_token
                c2t_sample = char_to_token_batch[b_idx].unsqueeze(0) # [1, seq_len_batch]
                
                # Safety check: indices in c2t_sample must not exceed preds length
                max_tok_idx = c2t_sample.max().item()
                if max_tok_idx >= preds.shape[0]:
                    # This happens if chunking parameters (max_chars) are different
                    print(f"Warning: Chunk sequence length mismatch at sample {global_idx} (cached tokens: {preds.shape[0]}, required: {max_tok_idx}). Skip.")
                    continue

                char_probs_gpu = torch.gather(preds.unsqueeze(0), dim=1, index=c2t_sample)
                char_probs_cpu = char_probs_gpu.cpu().numpy()[0] # [seq_len_batch]
                
                # 4. Refinement logic (Space/Punct peaks)
                offset = int(offsets_batch[b_idx])
                valid_char_len = int(char_mask_batch[b_idx].sum())
                c2t_np = c2t_sample[0].cpu().numpy()[:valid_char_len]
                p_char_sparse = np.zeros(valid_char_len, dtype=np.float32)

                if len(c2t_np) > 0:
                    token_ends = np.where(c2t_np[:-1] != c2t_np[1:])[0]
                    token_ends = np.append(token_ends, len(c2t_np) - 1)
                    last_start = 0
                    for tend in token_ends:
                        tid = c2t_np[tend]
                        if tid < preds.shape[0]:
                            prob = char_probs_cpu[tend]
                            best_idx = tend
                            found_pref = False
                            for j in range(last_start, tend + 1):
                                if j < len(dataloader_text):
                                    char = dataloader_text[j]
                                    if char.isspace():
                                        best_idx = j
                                        found_pref = True
                                        break
                                    elif not found_pref and char in PUNCT_CHARS:
                                        best_idx = j
                            p_char_sparse[best_idx] = prob
                        last_start = tend + 1

                end_p = min(offset + valid_char_len, total_text_len)
                actual_len = end_p - offset
                if actual_len > 0:
                    full_probs_max[offset:end_p] = np.maximum(full_probs_max[offset:end_p], p_char_sparse[:actual_len])
                    unfilled = ~label_filled[offset:end_p]
                    if np.any(unfilled):
                        full_labels[offset:end_p][unfilled] = char_labels_batch[b_idx][:actual_len][unfilled]
                        label_filled[offset:end_p] = True
                num_processed += 1

    final_preds = []
    final_labels = []
    for j in range(total_text_len):
        if label_filled[j] and full_labels[j] >= 0:
            max_p = full_probs_max[j]
            final_preds.append(1 if max_p > threshold else 0)
            final_labels.append(int(full_labels[j]))

    precision, recall, f1, _ = precision_recall_fscore_support(final_labels, final_preds, average="binary", zero_division=0)
    accuracy = accuracy_score(final_labels, final_preds)
    
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "total_time": total_time, "num_processed": num_processed
    }


if __name__ == "__main__":
    main()
