import torch
import time
import argparse
import spacy
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from embeddings import load_language_model, extract_token_embeddings, expand_to_char_embeddings, get_device
from model import SpacePredictorMLP
from data_sentence import get_sentence_dataloader

def evaluate_minerva(dataloader, llm_model, tokenizer, mlp, device, backend="mlx"):
    print(f"\n--- Running Minerva Inference on {len(dataloader.dataset)} chunks ---")
    all_preds = []
    all_labels = []
    
    total_time = 0.0
    num_processed = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            char_to_token = batch["char_to_token"].to(device)
            labels = batch["char_labels"].to(device)
            mask = batch["char_mask"].to(device)
            
            start_batch = time.time()
            
            tok_emb = extract_token_embeddings(llm_model, input_ids, attention_mask, backend=backend)
            char_emb = expand_to_char_embeddings(tok_emb, char_to_token)
            char_emb = char_emb.float()
            preds = mlp(char_emb)
            
            end_batch = time.time()
            total_time += (end_batch - start_batch)
            
            for b in range(preds.shape[0]):
                valid = mask[b]
                p = (preds[b][valid] > 0.4).int().cpu().tolist()
                l = labels[b][valid].int().cpu().tolist()
                all_preds.extend(p)
                all_labels.extend(l)
                num_processed += 1
                
            if (i + 1) % 10 == 0:
                print(f" Minerva: Batch {i+1}/{len(dataloader)} processed...")

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

def evaluate_spacy(dataloader, nlp_model):
    print(f"\n--- Running SpaCy Inference on {len(dataloader.dataset)} chunks ---")
    all_preds = []
    all_labels = []
    
    total_time = 0.0
    num_processed = 0
    
    for i, batch in enumerate(dataloader):
        labels = batch["char_labels"]
        mask = batch["char_mask"]
        texts = batch["spaceless"] # This is the original text with spaces
        
        start_batch = time.time()
        
        for b in range(len(texts)):
            text = texts[b]
            valid_len = int(mask[b].sum().item())
            
            # Ground truth
            l = labels[b][:valid_len].int().tolist()
            
            # Run SpaCy
            doc = nlp_model(text)
            
            # Extract boundary positions (character indices)
            # sent.end_char is the index of the character *after* the sentence.
            # In our dataset, the label=1 is on the SPACE separating sentences.
            # SpaCy usually includes the trailing space or punctuation.
            # We map SpaCy output to a binary list of length `valid_len`.
            
            p = [0] * valid_len
            
            for sent_idx, sent in enumerate(doc.sents):
                if sent_idx < len(list(doc.sents)) - 1:
                    # In our model, we label the space between sentences.
                    # SpaCy's sent.end_char gives the position after the sentence text.
                    boundary_idx = sent.end_char
                    
                    # Sometimes SpaCy includes the space in the sentence, sometimes it leaves it out.
                    # We check around end_char to find the space and mark it as 1.
                    if boundary_idx < valid_len:
                        if text[boundary_idx] == " ":
                            p[boundary_idx] = 1
                        elif boundary_idx > 0 and text[boundary_idx - 1] == " ":
                            p[boundary_idx - 1] = 1
                        else:
                            p[boundary_idx] = 1 # Fallback, mark the exact split point
                            
            end_batch = time.time()
            total_time += (end_batch - start_batch)
            start_batch = time.time() # Reset for next item calculation
            
            all_preds.extend(p)
            all_labels.extend(l)
            num_processed += 1
            
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

def print_comparison(spacy_res, minerva_res):
    print("\n" + "="*60)
    print("                 PERFORMANCE COMPARISON                 ")
    print("="*60)
    print(f"{'Metric':<20} | {'SpaCy (it_core_news_lg)':<20} | {'Minerva (MLX)':<15}")
    print("-" * 60)
    print(f"{'Chunks Processed':<20} | {spacy_res['num_processed']:<20} | {minerva_res['num_processed']:<15}")
    print(f"{'Accuracy':<20} | {spacy_res['accuracy']:<20.4f} | {minerva_res['accuracy']:<15.4f}")
    print(f"{'Precision':<20} | {spacy_res['precision']:<20.4f} | {minerva_res['precision']:<15.4f}")
    print(f"{'Recall':<20} | {spacy_res['recall']:<20.4f} | {minerva_res['recall']:<15.4f}")
    print(f"{'F1 Score':<20} | {spacy_res['f1']:<20.4f} | {minerva_res['f1']:<15.4f}")
    print("-" * 60)
    
    spacy_time = spacy_res['total_time']
    min_time = minerva_res['total_time']
    spacy_avg = (spacy_time / spacy_res['num_processed']) * 1000
    minerva_avg = (min_time / minerva_res['num_processed']) * 1000
    
    print(f"{'Total Time (s)':<20} | {spacy_time:<20.4f} | {min_time:<15.4f}")
    print(f"{'Avg Time/Chunk (ms)':<20} | {spacy_avg:<20.2f} | {minerva_avg:<15.2f}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Compare SpaCy and Minerva Performance")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to test")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for Minerva inference")
    args = parser.parse_args()
    
    device = get_device()
    backend = "mlx"
    
    # 1. Load Data
    print(f"Loading Test Data: {args.split} ...")
    llm_model, tokenizer = load_language_model(backend=backend, device=device)
    
    dataloader = get_sentence_dataloader(
        split=args.split,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        shuffle=False,
        max_chars=4096, 
        chunk_size=2,  
        augmentation_mode="original"
    )

    # 2. Run SpaCy
    print("\nLoading SpaCy model (it_core_news_lg)...")
    try:
        nlp = spacy.load("it_core_news_lg")
    except OSError:
        print("SpaCy Italian model not found. Downloading...")
        spacy.cli.download("it_core_news_lg")
        nlp = spacy.load("it_core_news_lg")
        
    spacy_results = evaluate_spacy(dataloader, nlp)
    
    # 3. Run Minerva
    print("\nLoading Minerva MLP...")
    checkpoint_path = Path("checkpoints/best_sentence_mlp.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    mlp = SpacePredictorMLP(hidden_dim=checkpoint.get("hidden_dim", 2048), dropout=0.0).to(device)
    mlp.load_state_dict(checkpoint["model_state_dict"])
    mlp.eval()
    
    minerva_results = evaluate_minerva(dataloader, llm_model, tokenizer, mlp, device, backend=backend)
    
    # 4. Print Comparison
    print_comparison(spacy_results, minerva_results)

if __name__ == "__main__":
    main()
