"""
Data pipeline for Sentence Splitter.

Loads Italian UD treebanks, creates character-level labels for sentence boundaries,
and builds PyTorch datasets for training.

Approach: The input text is continuous (spaces between words are preserved).
The MLP predicts which spaces/characters are SENTENCE BOUNDARIES
(label=1 means "sentence ends after this char").
"""

import os
import io
import urllib.request
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import conllu
import random
import re
from embeddings import MODEL_NAME

# Reusing UD_URLS and CACHE_DIR from data.py
from wordSplitter.data import UD_URLS, CACHE_DIR, download_ud_file, parse_conllu


def chunk_sentences(sentences: list[list[str]], chunk_size: int = 5) -> list[list[list[str]]]:
    """Group sentences into chunks of `chunk_size`."""
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunks.append(sentences[i : i + chunk_size])
    return chunks


def augment_twitter_style(chunk: list[list[str]]) -> list[list[str]]:
    """
    Transform a chunk of sentences into informal / Twitter-style text.
    - Randomly lowercase
    - Randomly replace Italian words with abbreviations
    - Randomly remove punctuation
    - Add hashtags or mentions
    """
    abbreviations = {
        "per": "x",
        "comunque": "cmq",
        "non": "nn",
        "perchè": "xkè",
        "perché": "xkè",
        "sei": "6",
        "uno": "1",
        "che": "k",
        "cosa": "ks",
        "quando": "quando", # usually already short
        "bene": "bn",
        "domani": "dmn",
        "amore": "amr",
        "niente": "nt",
    }
    
    new_chunk = []
    for sentence in chunk:
        new_sentence = []
        for word in sentence:
            w_lower = word.lower()
            
            # 1. Randomly remove punctuation
            if w_lower in ".,;:!?()" and random.random() < 0.5:
                continue
            
            # 2. Randomly use abbreviations
            if w_lower in abbreviations and random.random() < 0.4:
                word = abbreviations[w_lower]
            
            # 3. Random lowercase everything
            if random.random() < 0.8:
                word = word.lower()
                
            new_sentence.append(word)
            
        # 4. Add random mentions or hashtags
        if random.random() < 0.2:
            new_sentence.append("#" + random.choice(["ita", "mood", "web", "news", "italy"]))
        if random.random() < 0.1:
            new_sentence.append("@user")
            
        if new_sentence:
            new_chunk.append(new_sentence)
            
    return new_chunk


def make_sentence_bounds_labels(chunk: list[list[str]]) -> tuple[str, list[int]]:
    """
    Given a list of sentences (where each sentence is a list of words), create:
      - text: the concatenated string with spaces between words, keeping 
              sentences combined naturally.
      - labels: binary list of length len(text),
                labels[i] = 1 if the space/boundary after char i separates two sentences
                labels[i] = 0 otherwise
                
    The very last character has no semantic following boundary, so label=-1
    """
    text = ""
    labels = []
    
    for sent_idx, words in enumerate(chunk):
        sent_text = " ".join(words)
        text += sent_text
        labels.extend([0] * len(sent_text))
        
        # After a sentence, unless it's the last one, there is a space separating
        # it from the next sentence. This space is the sentence boundary.
        if sent_idx < len(chunk) - 1:
            text += " "
            # The character before this space (the last char of the sentence)
            # could be considered the boundary, OR the space itself.
            # Usually, the space character itself is predicting the boundary.
            # We'll label the SPACE character's position as 1.
            # So the last character of sent_text was '0' above.
            labels.append(1)
            
    # The last character of the entire string gets -1 (ignore)
    labels[-1] = -1
    return text, labels


def build_sentence_char_to_token_map(
    text: str,
    tokenizer: AutoTokenizer,
) -> tuple[list[int], list[int]]:
    """
    Tokenize the text and build a mapping from character position → token index.
    """
    encoding = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"].squeeze(0).tolist()
    offsets = encoding["offset_mapping"].squeeze(0).tolist()

    char_to_token = [0] * len(text)
    
    for tok_idx, (start, end) in enumerate(offsets):
        for char_pos in range(start, end):
            if char_pos < len(text):
                char_to_token[char_pos] = tok_idx

    return input_ids, char_to_token


class SentenceSplitDataset(Dataset):
    """
    PyTorch Dataset for the sentence splitting task.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer: Optional[AutoTokenizer] = None,
        max_chars: int = 2048,
        chunk_size: int = 5,
        augment_prob: float = 0.0,
        augmentation_mode: str = "original", # "original", "augmented", "both"
    ):
        assert split in UD_URLS, f"Invalid split: {split}"

        # Load and parse UD data using data.py's functions
        path = download_ud_file(split)
        self.sentences = parse_conllu(path)

        if split == "engTrain":
            self.sentences = self.sentences[:2773]
        elif split == "engDev":
            self.sentences = self.sentences[:185]
        elif split == "engTest":
            self.sentences = self.sentences[:173]

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        else:
            self.tokenizer = tokenizer

        self.samples = []
        chunks = chunk_sentences(self.sentences, chunk_size=chunk_size)
        
        print(f"Dataset '{split}': processing {len(chunks)} chunks (augment_prob={augment_prob})...")
        
        for chunk in chunks:
            # 1. Original
            if augmentation_mode in ("original", "both"):
                self._add_sample(chunk, max_chars)
            
            # 2. Augmented
            if augmentation_mode in ("augmented", "both"):
                # If mode is "augmented", we use augment_prob to decide per chunk
                # OR if it's "augmented" we might want to force it?
                # Let's stick to the prob.
                if augment_prob > 0 and random.random() < augment_prob:
                    aug_chunk = augment_twitter_style(chunk)
                    if aug_chunk:
                        self._add_sample(aug_chunk, max_chars)

    def _add_sample(self, chunk: list[list[str]], max_chars: int):
        text, labels = make_sentence_bounds_labels(chunk)
        if 0 < len(text) <= max_chars:
            input_ids, char_to_token = build_sentence_char_to_token_map(
                text, self.tokenizer
            )
            self.samples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "char_labels": torch.tensor(labels, dtype=torch.float32),
                    "char_to_token": torch.tensor(char_to_token, dtype=torch.long),
                    "spaceless": text,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_sentence_fn(batch: list[dict]) -> dict:
    """
    Custom collation function for sentence splitting.
    """
    input_ids = pad_sequence(
        [s["input_ids"] for s in batch], batch_first=True, padding_value=0
    )
    char_labels = pad_sequence(
        [s["char_labels"] for s in batch], batch_first=True, padding_value=-1.0
    )
    char_to_token = pad_sequence(
        [s["char_to_token"] for s in batch], batch_first=True, padding_value=0
    )
    attention_mask = pad_sequence(
        [torch.ones_like(s["input_ids"]) for s in batch],
        batch_first=True,
        padding_value=0,
    )
    char_mask = char_labels >= 0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "char_labels": char_labels,
        "char_to_token": char_to_token,
        "char_mask": char_mask,
        "spaceless": [s["spaceless"] for s in batch],
    }


def get_sentence_dataloader(
    split: str = "train",
    batch_size: int = 16,
    tokenizer: Optional[AutoTokenizer] = None,
    max_chars: int = 2048,
    chunk_size: int = 5,
    shuffle: Optional[bool] = None,
    augment_prob: float = 0.0,
    augmentation_mode: str = "original",
) -> DataLoader:
    dataset = SentenceSplitDataset(
        split=split, 
        tokenizer=tokenizer, 
        max_chars=max_chars, 
        chunk_size=chunk_size,
        augment_prob=augment_prob,
        augmentation_mode=augmentation_mode
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sentence_fn,
        num_workers=0,
    )
