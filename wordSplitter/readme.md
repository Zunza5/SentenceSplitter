# WordSplitter Module

This module provides specialized character-level word segmentation, designed to identify word boundaries in continuous or noisy text streams. It is particularly effective for processing social media content where standard tokenization often fails.

## Directory Structure

- **`main.py`**: The primary entry point. Orchestrates training, testing, and inference.
- **`model.py`**: Contains the neural network architecture (CNN + MLP).
- **`train.py`**: Logic for the training loop, including Focal Loss implementation.
- **`inference.py`**: Dedicated script for running predictions on single strings or batches.
- **`data.py`**: Dataset classes and preprocessing utilities.
- **`embeddings.py`**: Character-level embedding logic.

## CLI Documentation

The module is designed to be controlled via command-line arguments. The main interface is provided by `wordSplitter/main.py`.

### Global Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | `str` | `train` | Operation mode: `train`, `test`, or `inference`. |
| `--model_path` | `str` | `checkpoints/` | Directory to save or load model weights. |

### 1. Training Mode (`--mode train`)
Used to train the segmentation model from scratch or resume training.

```bash
python wordSplitter/main.py --mode train [OPTIONS]
```

**Training Options:**
- `--epochs`: Number of training iterations (Default: `20`).
- `--batch_size`: Number of samples per gradient update (Default: `64`).
- `--lr`: Learning rate for the optimizer (Default: `0.001`).
- `--val_split`: Percentage of data used for validation (Default: `0.1`).
- `--alpha`: Alpha parameter for Focal Loss to handle class imbalance.
- `--gamma`: Gamma parameter for Focal Loss to focus on hard examples.

### 2. Inference Mode (`--mode inference`)
Used to run the model on specific text input.

```bash
python wordSplitter/main.py --mode inference --text "input_text_here"
```

**Inference Options:**
- `--text`: The string you want to segment (e.g., "thisisatest").
- `--threshold`: Probability threshold for considering a character a split point (Default: `0.5`).

### 3. Evaluation Mode (`--mode test`)
Runs the model against a test dataset to generate performance metrics (F1 Score, Precision, Recall).

```bash
python wordSplitter/main.py --mode test --data_path "path/to/test_data.csv"
```

---

## Usage Examples

### Training with custom parameters:
```bash
python wordSplitter/main.py --mode train --epochs 50 --batch_size 128 --lr 0.0005
```

### Running a quick inference test:
```bash
python wordSplitter/main.py --mode inference --text "mondaymotivation"
# Output: [ "monday", "motivation" ]
```

### Batch inference using the dedicated script:
If you need to bypass the main orchestrator for specific inference tasks:
```bash
python wordSplitter/inference.py --text "segmentedtext" --model_path "checkpoints/best_model.pt"
```
