# Sentence Splitter (Token-Level MoE Architecture)

This repository implements a high-performance, token-level sentence boundary detection system. It leverages Large Language Model (LLM) embeddings combined with a Mixture of Experts (MoE) classifier and a Multi-Scale Convolutional Neural Network (CNN) to achieve state-of-the-art accuracy across diverse linguistic domains, from formal UD treebanks to noisy social media text (Twitter/Reddit).

---

## Neural Architecture

The system transitions from heavy LLM representations to lightweight, specialized boundary detectors.

### 1. Feature Extraction (LLM Backend)
- **Engine**: Supports `transformers` (PyTorch) or `mlx` (Apple Silicon optimized).
- **Backbone**: Default is `Qwen3.5-0.8B` or `Qwen2.5-0.5B`.
- **Strategy**: Extracts hidden states from the last layer. These embeddings encapsulate deep semantic and syntactic context, making the model robust to informal grammar and unconventional punctuation.

### 2. SpacePredictorMLP (MoE + CNN)
The classifier is designed to be efficient yet expressive:
- **Expert Routing**: Uses a Mixture of Experts (MoE) architecture where multiple `ExpertBlock` networks (MLPs) specialize in different linguistic patterns.
- **Dense Sequence Routing**: Optimized for Apple Silicon (MPS), the router processes the entire sequence in a single dense pass, avoiding memory-expensive dynamic masking.
- **Multi-Scale CNN**: Before the routing, a `MultiScaleConv1d` block captures context at three scales:
    - **Local (3x3)**: Standard sentence endings (periods, exclamation marks).
    - **Medium (5x5)**: Abbreviations and honorifics (e.g., Mr., St.).
    - **Wide (7x7, Dilation 2)**: Large-scale separators like quotes, brackets, and newlines.

---

## File-by-File Technical Reference

### Core Execution & API
- **`main_sentence.py`**: The central entry point. Orchestrates the three main workflows: `train`, `eval`, and `split` (inference). It uses a subparser-based CLI for task-specific configuration.
- **`inference_sentence.py`**: A clean API for programmatically using the model. Contains `split_into_sentences()` which handles LLM-to-MLP mapping and character-level text slicing.
- **`gui_sentence.py`**: A Tkinter-based graphical interface for interactive real-time testing of the model splitting performance.

### Training & Fine-Tuning
- **`train_sentence.py`**: Implements the main training pipeline. 
    - **Phase 1 (Extraction)**: Runs the LLM to save embeddings to disk.
    - **Phase 2 (Training)**: Loads cached embeddings via `CachedEmbeddingDataset` to train the MLP instantly (skipping the LLM forward pass).
    - **Memory Mapping**: Uses `mmap=True` to load consolidated datasets without exhausting RAM.
- **`finetune_sentence.py`**: A specialized script for cross-domain adaptation. 
    - **Macro-F1 Strategy**: Evaluates the model separately on multiple hard dev sets (e.g., medical, tweets, legal) and averages the F1 scores to prevent overfitting to the largest dataset.
    - **Balanced Batches**: Implements weighted sampling to ensure rare datasets are seen as often as large ones during training.

### Model Definitions
- **`model.py`**: Definitions for `SpacePredictorMLP`, `MoELayer`, `ExpertBlock`, and `MultiScaleConv1d`. Also contains the **Focal Loss** implementation, which addresses the extreme class imbalance (boundaries vs. non-boundaries).
- **`sentence_embeddings.py`**: Abstraction layer for LLM backends. Handles model loading, quantization (bfloat16/4-bit), and the `extract_token_embeddings` logic.

### Data & Evaluation
- **`data_sentence.py`**: The data pipeline.
    - **Universal Dependencies (UD)**: Loaders for CoNLLU and SentSplit formats.
    - **Augmentations**: Includes Twitter-style abbreviations or mentions and Boundary Diversity (varying punctuation) to improve model robustness.
    - **Labeling**: Logic for aligning character-level boundaries to tokenizer tokens.
- **`compare_spacy.py`**: A comprehensive benchmarking suite.
    - **Baseline Comparison**: Compares the model against **SpaCy** (`it_core_news_lg`) and **NLTK** (Punkt).
    - **Cached Benchmarking**: Supports a `--use-cache` mode to evaluate the MLP performance across thousands of sentences in seconds.
- **`consolidate_data.py`**: Maintenance utility. Merges individual batch-level cache files into a single `consolidated.pt` to optimize I/O and training speed.

---

## Installation & Setup

1. **Requirements**: Python 3.10+, PyTorch, Transformers.
2. **Setup**:
   ```bash
   pdm install  # Recommended (using pyproject.toml)
   # Or using pip:
   pip install torch transformers datasets scikit-learn
   ```
3. **Data**: Place your UD treebanks in `sent_split_data/` or provide URLs in `data_sentence.py`.

---

## Training Workflow

### Step 1: Offline Embedding Extraction
Extract and cache embeddings to avoid expensive LLM forward passes during training:
```bash
python main_sentence.py train --phase extract --backend transformers --max-chars 1024
```

### Step 2: Training the MLP
Train the lightweight classifier on the cached embeddings:
```bash
python main_sentence.py train --phase train --epochs 50 --lr 1e-4 --pos-weight 0.5
```

### Step 3: Domain Fine-Tuning (Optional)
Improve performance on difficult datasets using Macro-F1 balancing:
```bash
python finetune_sentence.py --train-splits it-postwita-train,it-twittiro-train --epochs 10
```

---

## Benchmarking

Compare performance against standard libraries:
```bash
python compare_spacy.py --test-splits ALL_TEST_SPLITS --use-cache
```

### Benchmark results (MPS - Apple Silicon)
| Model          | Accuracy | Precision | Recall | F1 Score |
|----------------|----------|-----------|--------|----------|
| SpaCy (LG)     | 0.9997   | 0.9721    | 0.9947 | 0.9833   |
| NLTK (Punkt)   | 0.9988   | 0.9585    | 0.9055 | 0.9313   |
| **Minerva (MoE)** | **0.9999** | **0.9946** | **0.9911** | **0.9929** |

---

## Advanced CLI Configuration

### `main_sentence.py train` Options:
- `--aux-weight`: (Default: `1e-5`) Controls the MoE balancing loss. Increase if one expert is dominating.
- `--pos-weight`: (Default: `0.5`) Compensation for class imbalance. Increase if the model has low recall (misses boundaries).
- `--balanced-batches`: Automatically balances training samples across multiple input datasets.

### `compare_spacy.py` Options:
- `--use-cache`: Extremely fast evaluation using `sentence_embedding_cache`. Required for testing on large splits like `it-vit` or `en-ewt`.
- `--threshold`: (Default: `0.5`) The probability at which a token is considered a sentence boundary.
