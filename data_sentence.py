"""
Data pipeline for Sentence Splitter.

Loads Italian UD treebanks, creates character-level labels for sentence boundaries,
and builds PyTorch datasets for training.

Approach: The input text is continuous (spaces between words are preserved).
The MLP predicts which spaces/characters are SENTENCE BOUNDARIES
(label=1 means "sentence ends after this char").
"""

import urllib.request
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import conllu
import random
from sentence_embeddings import MODEL_NAME


UD_URLS = {
    "it-isdt-train": str(Path(__file__).parent / "sent_split_data" / "UD_Italian-ISDT" / "it_isdt-ud-train.sent_split"),
    "it-isdt-dev": str(Path(__file__).parent / "sent_split_data" / "UD_Italian-ISDT" / "it_isdt-ud-dev.sent_split"),
    "it-isdt-test": str(Path(__file__).parent / "sent_split_data" / "UD_Italian-ISDT" / "it_isdt-ud-test.sent_split"),
    "it-vit-train": str(Path(__file__).parent / "sent_split_data" / "UD_Italian-VIT" / "it_vit-ud-train.sent_split"),
    "it-vit-dev": str(Path(__file__).parent / "sent_split_data" / "UD_Italian-VIT" / "it_vit-ud-dev.sent_split"),
    "it-vit-test": str(Path(__file__).parent / "sent_split_data" / "UD_Italian-VIT" / "it_vit-ud-test.sent_split"),
    "it-partut-train": str(Path(__file__).parent / "sent_split_data" / "UD_Italian-ParTUT" / "it_partut-ud-train.sent_split"),
    "it-partut-dev": str(Path(__file__).parent / "sent_split_data" / "UD_Italian-ParTUT" / "it_partut-ud-dev.sent_split"),
    "it-partut-test": str(Path(__file__).parent / "sent_split_data" / "UD_Italian-ParTUT" / "it_partut-ud-test.sent_split"),
    "it-markit-train": str(Path(__file__).parent / "sent_split_data" / "UD_Italian-MarkIT" / "it_markit-ud-train.sent_split"),
    "it-markit-dev": str(Path(__file__).parent / "sent_split_data" / "UD_Italian-MarkIT" / "it_markit-ud-dev.sent_split"),
    "it-markit-test": str(Path(__file__).parent / "sent_split_data" / "UD_Italian-MarkIT" / "it_markit-ud-test.sent_split"),
    "it-postwita-train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-PoSTWITA/refs/heads/master/it_postwita-ud-train.conllu",
    "it-postwita-dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-PoSTWITA/refs/heads/master/it_postwita-ud-dev.conllu",
    "it-postwita-test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-PoSTWITA/refs/heads/master/it_postwita-ud-test.conllu",
    "it-twittiro-train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-TWITTIRO/refs/heads/master/it_twittiro-ud-train.conllu",
    "it-twittiro-dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-TWITTIRO/refs/heads/master/it_twittiro-ud-dev.conllu",
    "it-twittiro-test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-TWITTIRO/refs/heads/master/it_twittiro-ud-test.conllu",
    "it-old-train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-Old/refs/heads/master/it_old-ud-train.conllu",
    "it-parlamint-train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ParlaMint/refs/heads/master/it_parlamint-ud-train.conllu",
    "en-ewt-train": str(Path(__file__).parent / "sent_split_data" / "UD_English-EWT" / "en_ewt-ud-train.sent_split"),
    "en-ewt-dev": str(Path(__file__).parent / "sent_split_data" / "UD_English-EWT" / "en_ewt-ud-dev.sent_split"),
    "en-ewt-test": str(Path(__file__).parent / "sent_split_data" / "UD_English-EWT" / "en_ewt-ud-test.sent_split"),
    "en-gum-train": str(Path(__file__).parent / "sent_split_data" / "UD_English-GUM" / "en_gum-ud-train.sent_split"),
    "en-gum-dev": str(Path(__file__).parent / "sent_split_data" / "UD_English-GUM" / "en_gum-ud-dev.sent_split"),
    "en-gum-test": str(Path(__file__).parent / "sent_split_data" / "UD_English-GUM" / "en_gum-ud-test.sent_split"),
    "en-partut-train": str(Path(__file__).parent / "sent_split_data" / "UD_English-ParTUT" / "en_partut-ud-train.sent_split"),
    "en-partut-dev": str(Path(__file__).parent / "sent_split_data" / "UD_English-ParTUT" / "en_partut-ud-dev.sent_split"),
    "en-partut-test": str(Path(__file__).parent / "sent_split_data" / "UD_English-ParTUT" / "en_partut-ud-test.sent_split"),
    "en-pud-test": str(Path(__file__).parent / "sent_split_data" / "UD_English-PUD" / "en_pud-ud-test.sent_split"),
}

CACHE_DIR = Path(__file__).parent / "data_cache"


def download_ud_file(split: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{split.replace('-', '_')}.conllu"
    if not path.exists():
        urllib.request.urlretrieve(UD_URLS[split], path)
    return path


def parse_conllu(path: Path) -> list[list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = conllu.parse(f.read())
    return [[tok["form"] for tok in sent if isinstance(tok["id"], int)] for sent in data]


def parse_sent_split(path: Path) -> list[list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    raw_sentences = [s.strip() for s in content.split("<EOS>") if s.strip()]
    return [sentence.split() for sentence in raw_sentences]


def get_sentences_for_split(split: str) -> list[list[str]]:
    p = Path(split)
    if p.exists() and p.is_file():
        return parse_sent_split(p) if p.suffix == ".sent_split" else parse_conllu(p)

    if split in UD_URLS:
        source = UD_URLS[split]
        source_path = Path(source)
        if source_path.exists() and source_path.is_file():
            return parse_sent_split(source_path) if source_path.suffix == ".sent_split" else parse_conllu(source_path)

        if source.startswith("http"):
            parts = source.split("/")
            folder_name = parts[-3]
            filename = parts[-1]
            local_sent_split = Path(__file__).parent / "sent_split_data" / folder_name / filename.replace(".conllu", ".sent_split")
            if local_sent_split.exists():
                return parse_sent_split(local_sent_split)

        return parse_conllu(download_ud_file(split))

    matches = list((Path(__file__).parent / "sent_split_data").rglob(f"*{split}*.sent_split"))
    if matches:
        return parse_sent_split(matches[0])
    raise ValueError(f"Could not find dataset for '{split}'.")


def chunk_sentences_by_chars(sentences: list[list[str]], max_chars: int = 2048, stride_chars: int = 1024) -> list[tuple[list[list[str]], int]]:
    """Group sentences such that their combined length is approx `max_chars`, sliding by `stride_chars`."""
    chunks = []
    current_chunk = []
    current_len = 0
    current_offset = 0
    
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_len = sum(len(w) for w in sent) + len(sent)
        
        if current_len + sent_len > max_chars and current_chunk:
            chunks.append((list(current_chunk), current_offset))
            dropped_len = 0
            while dropped_len < stride_chars and current_chunk:
                dropped_sent = current_chunk.pop(0)
                d_len = sum(len(w) for w in dropped_sent) + len(dropped_sent)
                dropped_len += d_len
                current_offset += d_len
            current_len -= dropped_len
        else:
            current_chunk.append(sent)
            current_len += sent_len
            i += 1
            
    if current_chunk:
        chunks.append((current_chunk, current_offset))
        
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
        stride_chars: int = 1024,
        augment_prob: float = 0.0,
        augmentation_mode: str = "original", # "original", "augmented", "both"
    ):
        # Load and parse using centralized helper (prioritizes local .sent_split)
        self.sentences = get_sentences_for_split(split)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        else:
            self.tokenizer = tokenizer

        self.samples = []
        chunks_with_offsets = chunk_sentences_by_chars(self.sentences, max_chars=max_chars, stride_chars=stride_chars)
        
        print(f"Dataset '{split}': processing {len(chunks_with_offsets)} chunks (augment_prob={augment_prob})...")
        
        for chunk, offset in chunks_with_offsets:
            # 1. Original
            if augmentation_mode in ("original", "both"):
                self._add_sample(chunk, max_chars, offset)
            
            # 2. Augmented
            if augmentation_mode in ("augmented", "both"):
                if augment_prob > 0 and random.random() < augment_prob:
                    aug_chunk = augment_twitter_style(chunk)
                    if aug_chunk:
                        # Augmented samples use the same offset as the original
                        self._add_sample(aug_chunk, max_chars, offset)

    def _add_sample(self, chunk: list[list[str]], max_chars: int, char_offset: int = 0):
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
                    "char_offset": char_offset,
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
        "char_offset": torch.tensor([s.get("char_offset", 0) for s in batch], dtype=torch.long),
    }


def get_sentence_dataloader(
    split: str = "train",
    batch_size: int = 16,
    tokenizer: Optional[AutoTokenizer] = None,
    max_chars: int = 2048,
    stride_chars: int = 1024,
    shuffle: Optional[bool] = None,
    augment_prob: float = 0.0,
    augmentation_mode: str = "original",
) -> DataLoader:
    dataset = SentenceSplitDataset(
        split=split, 
        tokenizer=tokenizer, 
        max_chars=max_chars, 
        stride_chars=stride_chars,
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
