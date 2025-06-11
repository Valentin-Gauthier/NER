import pandas as pd
import spacy
from pathlib import Path
import time
from datetime import datetime
import re

class SpaCy():

    def __init__(self, data:str=None, spaCy_model:str="fr_core_news_sm", timer_option:bool=False, log_option:bool=False, log_path:str="", verbose:bool=False):
        self.data = data
        self.spaCy_model = spaCy_model
        self.timer_option = timer_option
        self.log_option = log_option
        self.verbose = verbose
        self.log_path = log_path
        
        # -------------- Init -------------- #
        self.nlp = spacy.load(spaCy_model) # Load the spaCy model
        self.log_location = Path(log_path) if log_path else Path("Logs/log.txt")

        # if the file is given then load it
        if isinstance(self.data, str):
            self.data_df = pd.read_excel(self.data)

    # ---------------------------- TOOLS ----------------------- #
    def log(self, step:str, duration:float):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        if self.log_location.is_dir(): 
            log_file_path = self.log_location / "log.txt" # if the path is a folder -> add a filename
        else:
            log_file_path = self.log_location


        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{timestamp} - SpaCy [{step}] finish in {duration:.2f} s.\n")

    def chrono(func):
        def wrapper(self, *args, **kwargs):
            if self.timer_option or self.log_option:
                start = time.time()
            result = func(self, *args, **kwargs)
            if self.timer_option or self.log_option:
                duration = time.time() - start
                if self.timer_option:
                    print(f"{func.__name__} in : {duration:.2f}s")
                if self.log_option:
                    self.log(func.__name__, duration)
            return result
        return wrapper

    # ---------------------------------- METHODS ---------------------------------- #

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
    def run(self) -> pd.DataFrame:

        if self.verbose:
            print(f"[spaCy] spaCy version: {spacy.__version__}")
            print(f"[spaCy] spaCy model: {self.nlp.meta.get('name', 'unknown')}")

        rows = []
        for idx, row in self.data_df.iterrows():
            if not isinstance(row["desc"], str):
                continue
            doc = self.nlp(row["desc"])
            for ent in doc.ents:
                rows.append({
                    "titles" : self.data_df.loc[idx, "titles"],
                    "NER" : ent.text,
                    "NER_label" : ent.label_,
                    "desc" : self.get_context(row["desc"], ent.text, 5),
                    "method": "spaCy",
                    "file_id" : idx
                })

        self.df = pd.DataFrame(rows)

        if self.verbose:
            print(f"SpaCy : {self.df.shape}")

        return self.df