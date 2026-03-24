import os
from pathlib import Path

def clean_and_separate_sent_split(base_dir: str):
    """
    Legge i file .sent_split, rimuove i tag <EOS> e salva in nuovi file separati.
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Errore: La cartella '{base_dir}' non esiste.")
        return

    # Trova ricorsivamente tutti i file .sent_split nelle sottocartelle
    # come UD_Italian-ISDT o UD_English-EWT
    files = list(base_path.rglob("*.sent_split"))
    print(f"Trovati {len(files)} file da elaborare.")

    for file_path in files:
        try:
            # Apertura del file originale in lettura
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Sostituzione del tag <EOS> con il nulla (stringa vuota)
            cleaned_content = content.replace("<EOS>", "")
            
            # Definizione del nuovo percorso file (es. nomefile.sent_split.cleaned)
            # Questo garantisce che il file sia salvato separatamente
            new_file_path = file_path.with_suffix(file_path.suffix + ".cleaned")

            # Scrittura del contenuto pulito nel nuovo file
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)
            
            print(f"Creato file pulito: {new_file_path.name}")
            
        except Exception as e:
            print(f"Errore durante l'elaborazione di {file_path}: {e}")

if __name__ == "__main__":
    # Percorso della cartella contenente i dataset
    data_dir = "sent_split_data"
    clean_and_separate_sent_split(data_dir)