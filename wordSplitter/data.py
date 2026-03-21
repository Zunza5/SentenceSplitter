"""
Data pipeline for the Word Splitter.

Loads Italian UD treebanks, creates character-level labels,
and builds PyTorch datasets for training.

Approach: The input text has spaces between every character.
The MLP predicts which spaces to REMOVE (label=1 means "remove space after this char").
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
    "test2": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-PoSTWITA/refs/heads/master/it_postwita-ud-test.conllu",
    "train2": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-PoSTWITA/refs/heads/master/it_postwita-ud-train.conllu",
    "dev2": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-PoSTWITA/refs/heads/master/it_postwita-ud-dev.conllu",
    "engTrain": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu",
    "engDev": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu",
    "engTest": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu",
    "test3": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-VIT/refs/heads/master/it_vit-ud-test.conllu",
    "train3": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-VIT/refs/heads/master/it_vit-ud-train.conllu",
    "dev3": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-VIT/refs/heads/master/it_vit-ud-dev.conllu",
    "test4": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-TWITTIRO/refs/heads/master/it_twittiro-ud-test.conllu",
    "train4": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-TWITTIRO/refs/heads/master/it_twittiro-ud-train.conllu",
    "dev4": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-TWITTIRO/refs/heads/master/it_twittiro-ud-dev.conllu",
    "train5": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ParTUT/refs/heads/master/it_partut-ud-train.conllu",
    "test5": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ParTUT/refs/heads/master/it_partut-ud-test.conllu",
    "dev5": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ParTUT/refs/heads/master/it_partut-ud-dev.conllu",
}

CACHE_DIR = Path(__file__).parent.parent / "data_cache"

from wordSplitter.embeddings import MODEL_NAME

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


def make_char_labels(words: list[str]) -> tuple[str, str, list[int]]:
    """
    Given a list of words, create:
      - spaceless: the concatenated string without spaces
      - spaced: every character separated by a space ("c i a o c o m e ...")
      - labels: binary list of length len(spaceless),
                labels[i] = 1 if the space after char i should be REMOVED
                             (i.e. this char and the next are in the same word)
                labels[i] = 0 if the space after char i is a real word boundary
                             (keep the space)

    The last character has no space after it, so it gets label -1 (ignore).

    Example:
        words = ["ciao", "come", "stai"]
        spaceless = "ciaocomesstai"
        spaced    = "c i a o c o m e s t a i"
        labels    = [1,1,1,0, 1,1,1,0, 1,1,1,-1]
                         ^keep       ^keep      ^end (ignore)
    """
    spaceless = "".join(words)
    spaced = " ".join(list(spaceless))

    # Default: all spaces should be removed (same word → 1)
    labels = [1] * len(spaceless)

    # Mark real word boundaries as 0 (keep the space)
    pos = 0
    for word in words[:-1]:
        pos += len(word)
        labels[pos - 1] = 0  # space AFTER last char of this word is a real boundary

    # Last character: no space after it, ignore
    labels[-1] = -1

    return spaceless, spaced, labels


def build_char_to_token_map(
    spaced: str,
    tokenizer: AutoTokenizer,
) -> tuple[list[int], list[int]]:
    """
    Tokenize the SPACED string and build a mapping from
    character position → token index.

    The spaced string has the format "c i a o ..." where each original
    character is separated by spaces. We map each original character
    to the token that covers it.

    Returns:
        input_ids: token IDs from the tokenizer
        char_to_token: list of length num_original_chars,
                       char_to_token[i] = index into input_ids
    """
    encoding = tokenizer(
        spaced,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"].squeeze(0).tolist()
    offsets = encoding["offset_mapping"].squeeze(0).tolist()

    # In the spaced string, original char i is at position i*2
    # (e.g., "c i a o" → c=0, i=2, a=4, o=6)
    num_chars = (len(spaced) + 1) // 2  # number of original characters
    char_to_token = [0] * num_chars

    for tok_idx, (start, end) in enumerate(offsets):
        for spaced_pos in range(start, end):
            # Only map even positions (the actual characters, not the spaces)
            if spaced_pos % 2 == 0:
                char_idx = spaced_pos // 2
                if char_idx < num_chars:
                    char_to_token[char_idx] = tok_idx

    return input_ids, char_to_token


class WordSplitDataset(Dataset):
    """
    PyTorch Dataset for the word splitting task.

    Each item contains:
        - input_ids: tokenized spaced text (LongTensor)
        - char_labels: binary labels per character (FloatTensor)
        - char_to_token: mapping from char pos to token idx (LongTensor)
        - spaceless: the original spaceless string
        - spaced: the spaced-out string
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer: Optional[AutoTokenizer] = None,
        max_chars: int = 512,
    ):
        assert split in ("train", "dev", "test", "test2", "train2", "dev2", "engTrain", "engDev", "engTest", "test3", "train3", "dev3"), f"Invalid split: {split}"

        # Load and parse UD data
        path = download_ud_file(split)
        self.sentences = parse_conllu(path)

        if split == "engTrain":
            self.sentences = self.sentences[:2773]
        elif split == "engDev":
            self.sentences = self.sentences[:185]
        elif split == "engTest":
            self.sentences = self.sentences[:173]
        


        # Tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        else:
            self.tokenizer = tokenizer

        # Pre-process all samples
        self.samples = []
        for words in self.sentences:
            spaceless, spaced, labels = make_char_labels(words)
            if len(spaceless) == 0 or len(spaceless) > max_chars:
                continue
            input_ids, char_to_token = build_char_to_token_map(
                spaced, self.tokenizer
            )
            self.samples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "char_labels": torch.tensor(labels, dtype=torch.float32),
                    "char_to_token": torch.tensor(char_to_token, dtype=torch.long),
                    "spaceless": spaceless,
                    "spaced": spaced,
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
        "spaceless": [s["spaceless"] for s in batch],
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
