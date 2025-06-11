import stanza
import pandas as pd
import torch
import re
import time
from datetime import datetime
from pathlib import Path

class Stanza:
    
    def __init__(self, data:str=None, use_gpu:bool=False, logging:bool=True, log_folder:str=None, timer:bool=True, verbose:bool=False):
        self.data = data
        self.gpu = use_gpu
        self.timer = timer
        self.logging = logging
        self.log_folder = Path(log_folder)
        self.verbose = verbose

        if isinstance(self.data, str):
            self.data_df = pd.read_excel(self.data)
        else:
            print("Pas de fichier")

        # Load the French Pipeline (tokenize : slice the text, mwt: usefull for french word like "ajourd'hui", ner : analyse the text)
        self.nlp = stanza.Pipeline(lang="fr", processors='tokenize,mwt,ner', use_gpu=self.gpu)

        if self.verbose:
            print("PyTorch version :", torch.__version__)
            print("CUDA available :", torch.cuda.is_available())
            print("CUDA version (torch):", torch.version.cuda)
            print("Device name :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


    def log(self, step:str, duration:float):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        if self.log_folder.is_dir(): 
            log_file_path = self.log_folder / "log.txt" # if the path is a folder -> add a filename
        else:
            log_file_path = self.log_folder


        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{timestamp} - Stanza [{step}] finish in {duration:.2f} s.\n")

    def chrono(func):
        def wrapper(self, *args, **kwargs):
            if self.timer or self.logging:
                start = time.time()
            result = func(self, *args, **kwargs)
            if self.timer or self.logging:
                duration = time.time() - start
                if self.timer:
                    print(f"{func.__name__} in : {duration:.2f}s")
                if self.logging:
                    self.log(func.__name__, duration)
            return result
        return wrapper


    def get_context(self, desc: str, ner: str, window: int) -> str:
        """Return the original text context of an entity with surrounding words"""
        # Tokenize with positions
        matches = list(re.finditer(r"\w+|[^\w\s]", desc))
        tokens = [m.group(0) for m in matches]
        positions = [(m.start(), m.end()) for m in matches]

        # Tokenize ner in same way
        ner_tokens = re.findall(r"\w+|[^\w\s]", ner)
        len_ner = len(ner_tokens)

        # Match ner sequence in token list
        for i in range(len(tokens) - len_ner + 1):
            if [t.lower() for t in tokens[i:i + len_ner]] == [n.lower() for n in ner_tokens]:
                start_idx = max(0, i - window)
                end_idx = min(len(tokens), i + len_ner + window)

                start_char = positions[start_idx][0]
                end_char = positions[end_idx - 1][1]

                return desc[start_char:end_char]
        return desc
    
    @chrono
    def run(self):
        """Make a DataFrame with the result of Stanza analyses"""
        merged_text = ""
        idx_map = []  # (start_char, end_char, idx, row)
        current_pos = 0

        for idx, row in self.data_df.iterrows():
            desc = row.get("desc", "")
            if not isinstance(desc, str):
                continue
            marker = f"\n###DOCIDX={idx}###\n"
            desc_block = marker + desc.strip() + "\n"
            start = current_pos + len(marker)
            end = start + len(desc.strip())
            idx_map.append((start, end, idx, row))
            merged_text += desc_block
            current_pos += len(desc_block)

        doc = self.nlp(merged_text)

        rows = []
        for ent in doc.ents:
            ent_start = ent.start_char
            ent_end = ent.end_char

            # Trouver la description d’origine correspondant à l'entité
            for start, end, idx, row in idx_map:
                if start <= ent_start < end:
                    desc_text = self.data_df.loc[idx, "desc"]
                    rows.append({
                        "titles": row.get("titles", "Default title name"),
                        "NER": ent.text,
                        "NER_label": ent.type,
                        "desc": self.get_context(desc_text, ent.text, 5),
                        "method": "Stanza",
                        "file_id": idx
                    })
                    break

        self.df = pd.DataFrame(rows)

        if self.verbose:
            print(f"Stanza : {self.df.shape}")
        return self.df