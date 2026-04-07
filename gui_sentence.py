import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import time
import os

# New API Import
from api_sentence import SentenceSplitterAPI

class SentenceSplitterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentence Splitter PRO")
        self.root.geometry("950x750")
        self.root.configure(bg="#f8f9fa")
        
        # API Instance
        self.api = None
        self.backend = "transformers" # Default to transformers for stability
        
        # UI Setup
        self.setup_ui()
        
        # Start initial load in background
        self.status_var.set(f"Initializing Engine ({self.backend})...")
        self.split_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.load_api_thread, daemon=True).start()

    def setup_ui(self):
        # Header Styling
        header = tk.Frame(self.root, bg="#1e293b", height=70)
        header.pack(fill=tk.X)
        tk.Label(header, text="SENTENCE SPLITTER PRO", fg="#f1f5f9", bg="#1e293b", 
                 font=("Inter", 18, "bold")).pack(pady=20)

        # Main Container
        main_frame = tk.Frame(self.root, bg="#f8f9fa")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # Input Section
        tk.Label(main_frame, text="INPUT TEXT", bg="#f8f9fa", fg="#475569", 
                 font=("Inter", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.input_text = scrolledtext.ScrolledText(main_frame, height=12, font=("JetBrains Mono", 11),
                                                 relief=tk.FLAT, borderwidth=1, highlightthickness=1,
                                                 highlightbackground="#cbd5e1", highlightcolor="#3b82f6")
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        # Action Bar
        action_bar = tk.Frame(main_frame, bg="#f8f9fa")
        action_bar.pack(fill=tk.X, pady=(0, 20))
        
        # Custom Button Style using a Label for better aesthetic control if needed, 
        # but sticking to standard button for now with better colors
        self.split_btn = tk.Button(action_bar, text="SEGMENT DOCUMENT", command=self.on_split_click,
                                  font=("Inter", 10, "bold"), bg="#2563eb", fg="white", 
                                  activebackground="#1d4ed8", activeforeground="white",
                                  relief=tk.FLAT, padx=25, pady=12, cursor="hand2")
        self.split_btn.pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(action_bar, textvariable=self.status_var, fg="#64748b", bg="#f8f9fa",
                                   font=("Inter", 9, "italic"))
        self.status_label.pack(side=tk.LEFT, padx=25)

        # Output Section
        tk.Label(main_frame, text="SEGMENTED sentences", bg="#f8f9fa", fg="#475569",
                 font=("Inter", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.output_text = scrolledtext.ScrolledText(main_frame, height=12, font=("JetBrains Mono", 11),
                                                  relief=tk.FLAT, borderwidth=1, highlightthickness=1,
                                                  highlightbackground="#cbd5e1", bg="#ffffff")
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def load_api_thread(self):
        try:
            # Check for best checkpoint
            ckpt = "checkpoints/best_sentence_mlp.pt"
            if not os.path.exists(ckpt):
                # Try finding any checkpoint
                import glob
                ckpts = glob.glob("checkpoints/*.pt")
                if ckpts:
                    ckpt = ckpts[0]
                else:
                    raise FileNotFoundError("No model checkpoint found in checkpoints/")
            
            # Instantiate the API (loads LLM and MLP once)
            self.api = SentenceSplitterAPI(
                checkpoint_path=ckpt,
                backend=self.backend,
                batch_size=8
            )
            
            self.root.after(0, self.on_ready)
        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Engine Error", f"Failed to initialize API:\n{msg}"))
            self.root.after(0, lambda: self.status_var.set("Engine Failure"))

    def on_ready(self):
        self.status_var.set(f"Engine Ready • {str(self.api.device).upper()} • Stride: {self.api.stride_chars}")
        self.split_btn.config(state=tk.NORMAL)

    def on_split_click(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input Required", "Please paste the text you wish to segment.")
            return

        self.split_btn.config(state=tk.DISABLED)
        self.status_var.set("Analyzing structure... (Sliding Window + Batch 8)")
        self.output_text.delete("1.0", tk.END)

        threading.Thread(target=self.run_inference, args=(text,), daemon=True).start()

    def run_inference(self, text):
        try:
            start_t = time.time()
            # USE THE API: Automatically handles the sliding window complexity
            sentences = self.api.split_document(text)
            elapsed = time.time() - start_t
            
            self.root.after(0, self.on_complete, sentences, elapsed)
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: messagebox.showerror("Inference Error", err))
            self.root.after(0, lambda: self.status_var.set("Inference Error"))
            self.root.after(0, lambda: self.split_btn.config(state=tk.NORMAL))

    def on_complete(self, sentences, elapsed):
        self.output_text.delete("1.0", tk.END)
        for i, s in enumerate(sentences, 1):
            self.output_text.insert(tk.END, f"[{i:03d}] {s}\n\n")
            
        self.status_var.set(f"Completed! Found {len(sentences)} units in {elapsed:.2f}s")
        self.split_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    # Set app icon or styling if desired
    try:
        # Standard DPI awareness for Windows if needed
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
        
    app = SentenceSplitterGUI(root)
    root.mainloop()
