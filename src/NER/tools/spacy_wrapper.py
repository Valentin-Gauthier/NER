import pandas as pd
import spacy
import time
from datetime import datetime
from pathlib import Path
import yaml


class SpaCyConfig:

    def __init__(self, 
                 model:str="fr_core_news_sm",
                 production_mode:bool = True,
                 spacy_config:str=Path(__file__).parent / "config.yaml",
                 timer:bool=False, 
                 logging:bool=False, 
                 verbose:bool=False
                 ):
        
        self.production_mode = production_mode
        self.verbose = verbose
        self.timer = timer
        self.logging = logging
        self.spacy_config = Path(spacy_config)

        self.load_config()

        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise ValueError(f"Invalid spaCy model name: '{model}'. Make sure it is installed.")



    # ---------------------------- TOOLS ----------------------- #
    def log(self, step:str, duration:float):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        log_folder = Path(self.config["log_folder"])
        if log_folder.is_dir(): 
            log_file_path = log_folder / "log.txt" # if the path is a folder -> add a filename
        else:
            log_file_path = log_folder


        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{timestamp} - SpaCy [{step}] finish in {duration:.2f} s.\n")

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

    # ========================================== METHODS =================================================

    def load_config(self):
        """Load the JSON config about casEN"""

        if not self.spacy_config.is_file():
            raise FileNotFoundError(f"[load config] The provided file was not found ! {self.spacy_config}")
        else:
            with open(self.spacy_config, 'r', encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                if self.verbose: print(f"[load config] Config Loaded sucessfuly !")

    def load_data(self, data:pd.DataFrame | str):
        """Load the data"""
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            self.data = pd.read_excel(data)
        else:
            raise ValueError(f"Can't make DataFrame with the provided data ! {data}")


    @chrono
    def run(self, data:pd.DataFrame) -> pd.DataFrame:
        """Make a DataFrame with data analyse from SpaCy""" 

        # Load data
        self.load_data(data)

        window_size = self.config["description_window"]

        if self.verbose:
            print(f"[spaCy] spaCy version: {spacy.__version__}")
            print(f"[spaCy] spaCy model: {self.nlp.meta.get('name', 'unknown')}")

        rows = []
        for idx, row in self.data.iterrows():
            if not isinstance(row["desc"], str):
                continue
            doc = self.nlp(row["desc"])
            for ent in doc.ents:
                if self.production_mode:
                    rows.append({
                        "NER" : ent.text,
                        "NER_label" : ent.label_,
                        "method": "spaCy",
                        "file_id" : idx,
                        "entity_start" : ent.start_char,
                        "entity_end" : ent.end_char
                    })
                else:
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