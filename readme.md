# Sentence Splitter

This repository provides a lightweight, character-level sentence boundary prediction model. It uses Large Language Model (LLM) embeddings combined with a Multi-Scale Convolutional Neural Network (CNN) to accurately identify sentence splits.

## Methodology

The sentence splitting method operates at the character level to precisely locate sentence boundaries (e.g., spaces and punctuation). The architecture consists of the following pipeline:

1. **Embedding Extraction**: The input text is processed by a base LLM (such as Minerva or Qwen) to extract token embeddings.
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