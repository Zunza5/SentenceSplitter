import time
from api_sentence import SentenceSplitterAPI

def main():
    # 1. START THE ENGINE (Do this immediately at startup)
    print("Preparing system...")
    start_init = time.time()
    
    # Use backend="mlx" on Apple Silicon (Mac M-series) for maximum speed
    # Ensure the finetuned checkpoint exists if using this specific path
    ckpt_path = "checkpoints/finetuned_sentence_mlp.pt"
    # Fallback to best_sentence_mlp.pt if finetuned one doesn't exist
    import os
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/best_sentence_mlp.pt"
        
    api = SentenceSplitterAPI(checkpoint_path=ckpt_path, backend="transformers")
    print(f"Engine ready in {time.time() - start_init:.2f} seconds.\n")

    # 2. READ THE SECRET TEXT (e.g., from a file)
    # with open("secret_test_set.txt", "r", encoding="utf-8") as f:
    #     secret_text = f.read()
    
    # Quick live test:
    secret_text = """Il paziente (p.z.) di 45 aa. giunge in P.S. riferendo dolore toracico. All'ingresso PA 130/80 mmHg.
    Si consiglia D.Lgs. n. 81/2008 per sicurezza."""
    
    # 3. PROCESS (Use split_document which is OOM-resistant)
    print("Starting inference...")
    start_inf = time.time()
    
    sentences = api.split_document(secret_text)
    
    print(f"Inference completed in {time.time() - start_inf:.2f} seconds.")
    
    # 4. SAVE OR PRINT RESULTS
    print(f"\nFound {len(sentences)} sentences:")
    for i, s in enumerate(sentences, 1):
        print(f"{i}. {s}")

if __name__ == "__main__":
    main()
