import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import torch
import time

# Import pipeline functions
from inference_sentence import load_sentence_mlp, split_into_sentences, load_language_model, get_device

class SentenceSplitterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentence Splitter")
        self.root.geometry("800x600")
        
        # UI Setup
        self.setup_ui()
        
        # Model Variables
        self.device = None
        self.llm_model = None
        self.tokenizer = None
        self.mlp = None
        self.models_loaded = False
        
        # Start initial load in background
        self.status_var.set("Loading models (MLX)... please wait.")
        self.split_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.load_models_thread, daemon=True).start()

    def setup_ui(self):
        # Input Frame
        input_frame = tk.Frame(self.root)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(input_frame, text="Input Text (Paste long text here):").pack(anchor=tk.W)
        self.input_text = scrolledtext.ScrolledText(input_frame, height=10, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True)

        # Controls Frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.split_btn = tk.Button(control_frame, text="Split Sentences", command=self.on_split_click, height=2, bg="#4CAF50", fg="black")
        self.split_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing...")
        self.status_label = tk.Label(control_frame, textvariable=self.status_var, fg="blue")
        self.status_label.pack(side=tk.LEFT, padx=15)

        # Output Frame
        output_frame = tk.Frame(self.root)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(output_frame, text="Output (Sentences split by newlines):").pack(anchor=tk.W)
        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def load_models_thread(self):
        try:
            self.device = get_device()
            # Hardcoding mlx since it's the fastest on Apple Silicon
            backend = "mlx"
            self.llm_model, self.tokenizer = load_language_model(backend=backend, device=self.device)
            self.mlp = load_sentence_mlp(device=self.device)
            self.mlp.eval()
            self.models_loaded = True
            
            # Update UI on main thread
            self.root.after(0, self.on_models_loaded)
        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showerror("Error", f"Failed to load models:\n{str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Model load failed."))

    def on_models_loaded(self):
        self.status_var.set("Models loaded. Ready.")
        self.split_btn.config(state=tk.NORMAL)

    def on_split_click(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to split.")
            return

        # Disable button and update status
        self.split_btn.config(state=tk.DISABLED)
        self.status_var.set("Running inference... (this may take a few seconds)")
        self.output_text.delete("1.0", tk.END)

        # Run inference in a separate thread to avoid freezing UI
        threading.Thread(target=self.run_inference_thread, args=(text,), daemon=True).start()

    def run_inference_thread(self, text):
        try:
            start_time = time.time()
            
            # Predict
            with torch.no_grad():
                sentences = split_into_sentences(
                    text=text,
                    mlp=self.mlp,
                    llm_model=self.llm_model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    backend="mlx",
                    threshold=0.5
                )
                
            elapsed = time.time() - start_time
            
            # Update UI
            self.root.after(0, self.on_inference_complete, sentences, elapsed)
            
        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showerror("Error", f"Inference failed:\n{str(e)}"))
            self.root.after(0, self.on_inference_error)

    def on_inference_complete(self, sentences, elapsed):
        self.output_text.delete("1.0", tk.END)
        for idx, sentence in enumerate(sentences):
            self.output_text.insert(tk.END, f"{idx + 1}. {sentence}\n\n")
            
        self.status_var.set(f"Done! Found {len(sentences)} sentences in {elapsed:.2f}s")
        self.split_btn.config(state=tk.NORMAL)

    def on_inference_error(self):
        self.status_var.set("Error during inference.")
        self.split_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = SentenceSplitterGUI(root)
    root.mainloop()
