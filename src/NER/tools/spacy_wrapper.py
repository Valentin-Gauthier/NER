import pandas as pd
import spacy
import time
import datetime
from pathlib import Path


class SpaCyConfig:

    def __init__(self, data:pd.DataFrame, model:str="fr_core_news_sm", log_location:str=None, timer_option:bool=False, log_option:bool=False, verbose:bool=False):
        self.verbose = verbose
        self.log_location = Path(log_location) if log_location is not None  else None
        self.timer_option = timer_option
        self.log_option = log_option

        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise TypeError(f"Excpected a pandas DataFrame, got {format(type(data).__name__)}")
        
        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise ValueError(f"Invalid spaCy model name: '{model}'. Make sure it is installed.")



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


    # ========================================== METHODS =================================================



    @chrono
    def run(self) -> pd.DataFrame:
        """Make a DataFrame with data analyse from SpaCy""" 
        window_size = 30

        if self.verbose:
            print(f"[spaCy] spaCy version: {spacy.__version__}")
            print(f"[spaCy] spaCy model: {self.nlp.meta.get('name', 'unknown')}")
            print(f"[spaCy] window size of description: {window_size}")

        rows = []
        for idx, row in self.data.iterrows():
            if not isinstance(row["desc"], str):
                continue
            doc = self.nlp(row["desc"])
            for ent in doc.ents:
                start = max(ent.start_char - window_size, 0)
                end = min(ent.end_char + window_size, len(row["desc"]))
                context_window = row["desc"][start:end]

                rows.append({
                    "titles" : self.data.loc[idx, "titles"],
                    "NER" : ent.text,
                    "NER_label" : ent.label_,
                    "desc" : context_window,
                    "method": "spaCy",
                    "file_id" : idx,
                    "entity_start" : ent.start_char,
                    "entity_end" : ent.end_char
                })

        self.df = pd.DataFrame(rows)

        if self.verbose:
            print(f"SpaCy DataFrame shape: {self.df.shape}")

        return self.df