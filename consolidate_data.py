"""
Utility script to consolidate many small batch_*.pt files into a single consolidated.pt file.
This significantly improves I/O performance on macOS and other systems by using one large file.
"""

import torch
from pathlib import Path
from tqdm import tqdm
import argparse

def consolidate_split(cache_dir: Path, split_name: str, delete_original: bool = False):
    split_path = cache_dir / split_name
    if not split_path.exists():
        print(f"Skipping {split_name}: directory not found.")
        return

    files = sorted(split_path.glob("batch_*.pt"))
    if not files:
        print(f"No batch files found in {split_path}")
        return

    print(f"Consolidating {len(files)} files for split '{split_name}'...")
    
    all_embeddings = []
    all_labels = []
    all_masks = []
    all_spaceless = []

    max_seq_len = 0
    hidden_dim = 0

    # First pass: find max_seq_len
    for f in files:
        data = torch.load(f, weights_only=True)
        max_seq_len = max(max_seq_len, data["token_embeddings"].shape[1])
        hidden_dim = data["token_embeddings"].shape[2]
        del data

    print(f"  → Max sequence length for '{split_name}': {max_seq_len} (using bfloat16)")

    for f in tqdm(files):
        data = torch.load(f, weights_only=True)
        batch_size, seq_len, _ = data["token_embeddings"].shape
        
        # Convert to bfloat16 to save space and RAM
        emb = data["token_embeddings"].to(torch.bfloat16)
        
        if seq_len < max_seq_len:
            # Pad embeddings
            padded_emb = torch.zeros(batch_size, max_seq_len, hidden_dim, dtype=torch.bfloat16)
            padded_emb[:, :seq_len, :] = emb
            all_embeddings.append(padded_emb)
            
            # Pad labels with -1
            padded_labels = torch.full((batch_size, max_seq_len), -1.0)
            padded_labels[:, :seq_len] = data["token_labels"]
            all_labels.append(padded_labels)
            
            # Pad masks with False
            padded_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
            padded_mask[:, :seq_len] = data["token_mask"]
            all_masks.append(padded_mask)
        else:
            all_embeddings.append(emb)
            all_labels.append(data["token_labels"])
            all_masks.append(data["token_mask"])
            
        if "spaceless" in data:
            if isinstance(data["spaceless"], (list, tuple)):
                all_spaceless.extend(data["spaceless"])
            else:
                all_spaceless.append(data["spaceless"])
        del data

    # Concatenate all tensors
    consolidated_data = {
        "token_embeddings": torch.cat(all_embeddings, dim=0),
        "token_labels": torch.cat(all_labels, dim=0),
        "token_mask": torch.cat(all_masks, dim=0),
        "spaceless": all_spaceless if all_spaceless else None
    }

    save_path = split_path / "consolidated.pt"
    torch.save(consolidated_data, save_path)
    print(f"✓ Saved consolidated data to {save_path} ({consolidated_data['token_embeddings'].shape[0]} samples)")
    
    # Cleanup small files if requested
    if delete_original:
        print(f"  → Deleting {len(files)} original batch files...")
        for f in files:
            f.unlink()

def main():
    parser = argparse.ArgumentParser(description="Consolidate small embedding batch files.")
    parser.add_argument("--cache-dir", type=str, default="sentence_embedding_cache")
    parser.add_argument("--delete-original", action="store_true", help="Delete small files after successful consolidation.")
    args = parser.parse_args()

    cache_root = Path(args.cache_dir)
    if not cache_root.exists():
        print(f"Cache root {cache_root} does not exist.")
        return

    # Find all subdirectories which represent splits
    splits = [d.name for d in cache_root.iterdir() if d.is_dir()]
    
    for split in splits:
        consolidate_split(cache_root, split, delete_original=args.delete_original)

if __name__ == "__main__":
    main()
