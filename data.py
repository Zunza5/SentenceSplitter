"""
Data pipeline for the Word Splitter.

Loads Italian UD treebanks, creates character-level labels,
and builds PyTorch datasets for training.
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

# ── UD Treebank URLs ──────────────────────────────────────────────────────────
UD_URLS = {
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu",
    "dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-dev.conllu",
    "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-test.conllu",
}

CACHE_DIR = Path(__file__).parent / "data_cache"

MODEL_NAME = "sapienzanlp/Minerva-1B-base-v1.0"


def download_ud_file(split: str) -> Path:
    """Download a UD conllu file if not already cached."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"it_isdt-ud-{split}.conllu"
    if not path.exists():
        print(f"Downloading {split} split...")
        urllib.request.urlretrieve(UD_URLS[split], path)
        print(f"  → saved to {path}")
    return path


def parse_conllu(path: Path) -> list[list[str]]:
    """Parse a conllu file and return list of sentences (each a list of word-forms)."""
    with open(path, "r", encoding="utf-8") as f:
        data = conllu.parse(f.read())
    sentences = []
    for sent in data:
        # Skip multi-word tokens (id is a range like "1-2")
        words = [tok["form"] for tok in sent if isinstance(tok["id"], int)]
        sentences.append(words)
    return sentences


def make_char_labels(words: list[str]) -> tuple[str, list[int]]:
    """
    Given a list of words, create:
      - spaceless: the concatenated string without spaces
      - labels: binary list of length len(spaceless),
                labels[i] = 1 if a space should be inserted AFTER character i

    Example:
        words = ["ciao", "come", "stai"]
        spaceless = "ciaocomesstai"
        labels    = [0,0,0,1, 0,0,0,1, 0,0,0,0]
                             ^space    ^space    ^end (no space)
    """
    spaceless = "".join(words)
    labels = [0] * len(spaceless)

    pos = 0
    for word in words[:-1]:  # no space after the last word
        pos += len(word)
        labels[pos - 1] = 1  # space AFTER last char of this word

    return spaceless, labels


def build_char_to_token_map(
    spaceless: str,
    tokenizer: AutoTokenizer,
) -> tuple[list[int], list[int]]:
    """
    Tokenize the spaceless string and build a mapping from
    character position → token index.

    Returns:
        input_ids: token IDs from the tokenizer
        char_to_token: list of length len(spaceless),
                       char_to_token[i] = index into input_ids
    """
    encoding = tokenizer(
        spaceless,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"].squeeze(0).tolist()
    offsets = encoding["offset_mapping"].squeeze(0).tolist()

    # Build char → token mapping
    char_to_token = [0] * len(spaceless)
    for tok_idx, (start, end) in enumerate(offsets):
        for char_idx in range(start, end):
            if char_idx < len(spaceless):
                char_to_token[char_idx] = tok_idx

    return input_ids, char_to_token


class WordSplitDataset(Dataset):
    """
    PyTorch Dataset for the word splitting task.

    Each item contains:
        - input_ids: tokenized spaceless text (LongTensor)
        - char_labels: binary labels per character (FloatTensor)
        - char_to_token: mapping from char pos to token idx (LongTensor)
        - spaceless: the original spaceless string
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer: Optional[AutoTokenizer] = None,
        max_chars: int = 512,
    ):
        assert split in ("train", "dev", "test"), f"Invalid split: {split}"

        # Load and parse UD data
        path = download_ud_file(split)
        self.sentences = parse_conllu(path)

        # Tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        else:
            self.tokenizer = tokenizer

        # Pre-process all samples
        self.samples = []
        for words in self.sentences:
            spaceless, labels = make_char_labels(words)
            if len(spaceless) == 0 or len(spaceless) > max_chars:
                continue
            input_ids, char_to_token = build_char_to_token_map(
                spaceless, self.tokenizer
            )
            self.samples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "char_labels": torch.tensor(labels, dtype=torch.float32),
                    "char_to_token": torch.tensor(char_to_token, dtype=torch.long),
                    "spaceless": spaceless,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collation function that pads sequences to the max length in the batch.
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
    char_mask = char_labels >= 0  # True for real positions, False for padding

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "char_labels": char_labels,
        "char_to_token": char_to_token,
        "char_mask": char_mask,
    }


def get_dataloader(
    split: str = "train",
    batch_size: int = 32,
    tokenizer: Optional[AutoTokenizer] = None,
    max_chars: int = 512,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Convenience function to create a DataLoader."""
    dataset = WordSplitDataset(split=split, tokenizer=tokenizer, max_chars=max_chars)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
    )
