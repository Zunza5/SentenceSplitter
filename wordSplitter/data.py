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
# ── UD Treebank URLs ──────────────────────────────────────────────────────────
UD_URLS = {
    # Italian
    "it-isdt-train": str(Path(__file__).parent.parent / "sent_split_data" / "UD_Italian-ISDT" / "it_isdt-ud-train.sent_split"),
    "it-isdt-dev": str(Path(__file__).parent.parent / "sent_split_data" / "UD_Italian-ISDT" / "it_isdt-ud-dev.sent_split"),
    "it-isdt-test": str(Path(__file__).parent.parent / "sent_split_data" / "UD_Italian-ISDT" / "it_isdt-ud-test.sent_split"),
    
    "it-vit-train": str(Path(__file__).parent.parent / "sent_split_data" / "UD_Italian-VIT" / "it_vit-ud-train.sent_split"),
    "it-vit-dev": str(Path(__file__).parent.parent / "sent_split_data" / "UD_Italian-VIT" / "it_vit-ud-dev.sent_split"),
    "it-vit-test": str(Path(__file__).parent.parent / "sent_split_data" / "UD_Italian-VIT" / "it_vit-ud-test.sent_split"),
    
    "it-partut-train": str(Path(__file__).parent.parent / "sent_split_data" / "UD_Italian-ParTUT" / "it_partut-ud-train.sent_split"),
    "it-partut-dev": str(Path(__file__).parent.parent / "sent_split_data" / "UD_Italian-ParTUT" / "it_partut-ud-dev.sent_split"),
    "it-partut-test": str(Path(__file__).parent.parent / "sent_split_data" / "UD_Italian-ParTUT" / "it_partut-ud-test.sent_split"),
    
    "it-markit-train": str(Path(__file__).parent.parent / "sent_split_data" / "UD_Italian-MarkIT" / "it_markit-ud-train.sent_split"),
    "it-markit-dev": str(Path(__file__).parent.parent / "sent_split_data" / "UD_Italian-MarkIT" / "it_markit-ud-dev.sent_split"),
    "it-markit-test": str(Path(__file__).parent.parent / "sent_split_data" / "UD_Italian-MarkIT" / "it_markit-ud-test.sent_split"),

    "it-postwita-train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-PoSTWITA/refs/heads/master/it_postwita-ud-train.conllu",
    "it-postwita-dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-PoSTWITA/refs/heads/master/it_postwita-ud-dev.conllu",
    "it-postwita-test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-PoSTWITA/refs/heads/master/it_postwita-ud-test.conllu",
    
    "it-twittiro-train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-TWITTIRO/refs/heads/master/it_twittiro-ud-train.conllu",
    "it-twittiro-dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-TWITTIRO/refs/heads/master/it_twittiro-ud-dev.conllu",
    "it-twittiro-test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-TWITTIRO/refs/heads/master/it_twittiro-ud-test.conllu",

    "it-old-train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-Old/refs/heads/master/it_old-ud-train.conllu",

    "it-parlamint-train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ParlaMint/refs/heads/master/it_parlamint-ud-train.conllu",
    
    # English
    "en-ewt-train": str(Path(__file__).parent.parent / "sent_split_data" / "UD_English-EWT" / "en_ewt-ud-train.sent_split"),
    "en-ewt-dev": str(Path(__file__).parent.parent / "sent_split_data" / "UD_English-EWT" / "en_ewt-ud-dev.sent_split"),
    "en-ewt-test": str(Path(__file__).parent.parent / "sent_split_data" / "UD_English-EWT" / "en_ewt-ud-test.sent_split"),
    
    "en-gum-train": str(Path(__file__).parent.parent / "sent_split_data" / "UD_English-GUM" / "en_gum-ud-train.sent_split"),
    "en-gum-dev": str(Path(__file__).parent.parent / "sent_split_data" / "UD_English-GUM" / "en_gum-ud-dev.sent_split"),
    "en-gum-test": str(Path(__file__).parent.parent / "sent_split_data" / "UD_English-GUM" / "en_gum-ud-test.sent_split"),
    
    "en-partut-train": str(Path(__file__).parent.parent / "sent_split_data" / "UD_English-ParTUT" / "en_partut-ud-train.sent_split"),
    "en-partut-dev": str(Path(__file__).parent.parent / "sent_split_data" / "UD_English-ParTUT" / "en_partut-ud-dev.sent_split"),
    "en-partut-test": str(Path(__file__).parent.parent / "sent_split_data" / "UD_English-ParTUT" / "en_partut-ud-test.sent_split"),

    "en-pud-test": str(Path(__file__).parent.parent / "sent_split_data" / "UD_English-PUD" / "en_pud-ud-test.sent_split"),
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


def parse_sent_split(path: Path) -> list[list[str]]:
    """Parse a .sent_split file where sentences are separated by <EOS>."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by <EOS> and strip
    raw_sentences = [s.strip() for s in content.split("<EOS>")]
    raw_sentences = [s for s in raw_sentences if s]
    
    # Each sentence is a list of words
    sentences = [sentence.split() for sentence in raw_sentences]
    return sentences


def get_sentences_for_split(split: str) -> list[list[str]]:
    """
    Get sentences for a given split.
    Priority:
    1. If `split` is an existing file path -> load directly.
    2. If `split` is in `UD_URLS`:
       a. If `UD_URLS[split]` is an existing local file -> load it.
       b. If a matching `.sent_split` exists in `sent_split_data` (derived from URL) -> load it.
       c. Otherwise -> download `.conllu` and parse.
    3. If `split` matches a filename in `sent_split_data` -> load it.
    """
    
    # 1. Direct path check
    p = Path(split)
    if p.exists() and p.is_file():
        return parse_sent_split(p) if p.suffix == ".sent_split" else parse_conllu(p)

    # 2. UD_URLS check
    if split in UD_URLS:
        url_or_path = UD_URLS[split]
        path_in_urls = Path(url_or_path)
        
        # a. Local path in UD_URLS
        if path_in_urls.exists() and path_in_urls.is_file():
            print(f"Loading local file from UD_URLS['{split}']: {path_in_urls}")
            return parse_sent_split(path_in_urls) if path_in_urls.suffix == ".sent_split" else parse_conllu(path_in_urls)
            
        # b. Mapping to sent_split_data (for URLs)
        if url_or_path.startswith("http"):
            parts = url_or_path.split('/')
            folder_name = parts[-3]
            filename = parts[-1]
            local_sent_split = Path(__file__).parent.parent / "sent_split_data" / folder_name / filename.replace(".conllu", ".sent_split")
            if local_sent_split.exists():
                print(f"Found mapped sent_split for '{split}': {local_sent_split}")
                return parse_sent_split(local_sent_split)
        
        # c. Fallback to download conllu
        conllu_file = download_ud_file(split)
        return parse_conllu(conllu_file)
            
    # 3. Fuzzy search in sent_split_data
    sent_split_dir = Path(__file__).parent.parent / "sent_split_data"
    matches = list(sent_split_dir.rglob(f"*{split}*.sent_split"))
    if matches:
        print(f"Matching local file found for '{split}': {matches[0]}")
        return parse_sent_split(matches[0])
        
    raise ValueError(f"Could not find dataset for '{split}'.")


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
        # Remove restrictive split whitelist
        # Load and parse data (will use local .sent_split if available)
        self.sentences = get_sentences_for_split(split)

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
