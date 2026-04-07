import os
from pathlib import Path

def clean_and_separate_sent_split(base_dir: str):
    """
    Reads .sent_split files, removes <EOS> tags, and saves them into new separate files.
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Folder '{base_dir}' does not exist.")
        return

    # Recursively find all .sent_split files in subfolders
    # such as UD_Italian-ISDT or UD_English-EWT
    files = list(base_path.rglob("*.sent_split"))
    print(f"Found {len(files)} files to process.")

    for file_path in files:
        try:
            # Open the original file for reading
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Replace <EOS> tag with an empty string
            cleaned_content = content.replace("<EOS>", "")
            
            # Define new file path (e.g., filename.sent_split.cleaned)
            # This ensures the file is saved separately
            new_file_path = file_path.with_suffix(file_path.suffix + ".cleaned")

            # Write cleaned content to the new file
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)
            
            print(f"Created cleaned file: {new_file_path.name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Path to the directory containing datasets
    data_dir = "sent_split_data"
    clean_and_separate_sent_split(data_dir)