# Sentence Splitter

This repository provides a lightweight, character-level sentence boundary prediction model. It uses Large Language Model (LLM) embeddings combined with a Multi-Scale Convolutional Neural Network (CNN) to accurately identify sentence splits.

## Methodology

The sentence splitting method operates at the character level to precisely locate sentence boundaries (e.g., spaces and punctuation). The architecture consists of the following pipeline:

1. **Embedding Extraction**: The input text is processed by a base LLM (such as Qwen) to extract token embeddings.
2. **Character Expansion**: The token-level embeddings are expanded to character-level embeddings using a mapping algorithm.
3. **SpacePredictorMLP**: The core model is a lightweight Multi-Layer Perceptron (MLP) combined with a `MultiScaleConv1d` block. Self-attention was deliberately removed to prevent overfitting on small datasets.
   - **Multi-Scale CNN**: Captures context at three different scales using parallel branches:
     - *Local context* (kernel size 3) for standard periods and spaces.
     - *Medium context* (kernel size 5) for abbreviations.
     - *Wide context* (kernel size 7, dilation 2) for quotes and brackets.
4. **Classification & Focal Loss**: The model predicts a boundary probability for each character (0 or 1) using a Sigmoid activation. During training, it utilizes `FocalLoss` to handle the severe class imbalance, since sentence boundaries are much rarer than standard characters.

## Training

Training is divided into a two-phase pipeline to optimize resource usage and speed:

1. **Phase 1: Offline Embedding Extraction**
   To avoid recomputing heavy LLM embeddings during each epoch, the embeddings for sentence chunks are extracted offline and cached to the disk. You can optionally apply data augmentation during this phase.
   
2. **Phase 2: MLP Training**
   Once embeddings are cached, the lightweight `SpacePredictorMLP` is trained on these representations. The training pipeline uses an Adam optimizer, ReduceLROnPlateau scheduling, and early stopping based on the F1 score evaluated on a validation (dev) split.

## Evaluation and Inference (Splits)

You can run inference to split continuous text into sentences or evaluate the model against a specific dataset split (e.g., `test`, `dev`, `train`).

- **Inference**: The `split_into_sentences` function maps the continuous text to embeddings, passes them through the MLP, and slices the text wherever the model predicts a boundary probability higher than a defined threshold (default `0.5`).
- **Performance Evaluation**: You can test the accuracy on a specific dataset split using the `test_performance.py` script. It computes Accuracy, Precision, Recall, and F1 Score, along with the average inference time per chunk.
  
  ```bash
  python test_performance.py --split test --backend transformers --batch-size 32
This guide provides detailed instructions on how to use the Sentence Splitter CLI via `main_sentence.py`.

---

## Command-Line Interface Guide

The `main_sentence.py` script serves as the central entry point for all operations, including training, evaluation, and live inference.

### 1. Training (`train`)
The training command handles both the extraction of LLM embeddings and the optimization of the MLP classifier.

**Basic Syntax:**
```bash
python main_sentence.py train [OPTIONS]
```

**Key Arguments:**
* **`--phase`**: Selects the workflow stage. Options are `extract` (only save embeddings), `train` (only train the MLP on cached data), or `both` (default).
* **`--backend`**: Chooses the LLM engine: `transformers` or `mlx`.
* **`--augment-prob`**: Sets the probability (0.0 to 1.0) of generating informal "Twitter-style" data during extraction.
* **`--epochs`**: Maximum training epochs (default: `50`).
* **`--lr`**: Learning rate for the Adam optimizer (default: `1e-4`).
* **`--pos-weight`**: Weight for the positive class in Focal Loss to address imbalance (default: `0.8`).
* **`--train-splits` / `--dev-splits`**: Comma-separated lists of UD splits to use (e.g., `"train,train2,engTrain"`).

**Example - Full Training with Augmentation:**
```bash
python main_sentence.py --backend mlx train --phase both --augment-prob 0.4 --epochs 30
```

---

### 2. Evaluation (`eval`)
Once a model is trained, use the `eval` command to measure its performance on specific test datasets.

**Basic Syntax:**
```bash
python main_sentence.py eval [OPTIONS]
```

**Key Arguments:**
* **`--test-splits`**: Comma-separated list of cached splits to evaluate (default: `"test,test2,test3,test4,test5,engTest"`).
* **`--batch-size`**: Number of samples processed per batch (default: `16`).

**Example:**
```bash
python main_sentence.py eval --test-splits "test,engTest" --batch-size 32
```

---

### 3. Sentence Splitting (`split`)
This command allows you to perform live inference on a custom string of text.

**Basic Syntax:**
```bash
python main_sentence.py split "YOUR_TEXT_HERE" [OPTIONS]
```

**Key Arguments:**
* **`text`**: The continuous string you want to split into sentences.
* **`--threshold`**: The probability confidence required to trigger a split (default: `0.5`). Lowering this makes the model more "aggressive" at splitting.

**Example:**
```bash
python main_sentence.py split "This is the first sentence. And this is the second? Yes!" --threshold 0.6
```

---

Here is the detailed documentation for the Command-Line Interface (CLI) and benchmark scripts in English:

---

## 2. Benchmark Scripts

These specialized scripts provide deeper insights into the model's accuracy and computational efficiency.

### Performance Tester (`test_performance.py`)
Evaluates the model and provides detailed timing for each phase (LLM extraction, expansion, and MLP prediction).

**Syntax:**
```bash
python test_performance.py --split [SPLIT] --backend [BACKEND] --batch-size [SIZE]
```

**Arguments:**
* **`--split`**: The dataset split to test (e.g., `test`, `engTest`). Default is `test`.
* **`--backend`**: Use `transformers` or `mlx` engine. Default is `transformers`.
* **`--batch-size`**: Number of chunks to process per batch. Default is `32`.

### SpaCy Comparison (`compare_spacy.py`)
Benchmarks the Minerva MLP model against the standard SpaCy Italian model (`it_core_news_lg`).

**Syntax:**
```bash
python compare_spacy.py --split [SPLIT] --batch-size [SIZE]
```

**Arguments:**
* **`--split`**: The dataset split for comparison. Default is `test`.
* **`--batch-size`**: Batch size specifically for the Minerva inference part. Default is `32`.

### Metrics Reported
Both benchmark scripts output the following metrics:
* **Accuracy, Precision, Recall, and F1 Score**: Standard quality metrics for boundary detection.
* **Total Inference Time**: The total time taken to process the chosen split.
* **Avg Time per Chunk**: The average processing speed in milliseconds (ms) per text segment.omparison with SpaCy's `it_core_news_lg` model to benchmark accuracy and speed.

