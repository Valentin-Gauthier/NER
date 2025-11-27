import stanza
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import yaml
import tqdm

class StanzaConfig:

    def __init__(self, 
                 use_gpu:bool=True, 
                 production_mode:bool=True,
                 order_dataframe:bool=False,
                 stanza_config:str=Path(__file__).parent.parent / "config.yaml",
                 timer:bool=False, 
                 logging:bool=False, 
                 verbose:bool=False
                 ):
        self.use_gpu = use_gpu
        self.production_mode = production_mode
        self.order_dataframe = order_dataframe
        self.stanza_config = stanza_config
        self.verbose = verbose
        self.timer = timer
        self.logging = logging
    
        self.load_config()

        
        # Load the French Pipeline (tokenize : slice the text, mwt: usefull for french word like "ajourd'hui", ner : analyse the text)
        # download_method=None : doesnt look for update stanza (because if we don't have connection it trying to update and cause an error)
        self.nlp = stanza.Pipeline(lang="fr", processors='tokenize,mwt,ner', use_gpu=self.use_gpu, download_method=None)


    # ------------------- TOOLS --------------------------
    def log(self, step:str, duration:float):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        log_folder = Path(self.config["log_folder"])
        if log_folder.is_dir(): 
            log_file_path = log_folder / "log.txt" # if the path is a folder -> add a filename
        else:
            log_file_path = log_folder


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

    def load_config(self):
        """Load the JSON config about casEN"""

        if not self.stanza_config.is_file():
            raise FileNotFoundError(f"[load config] The provided file was not found ! {self.stanza_config}")
        else:
            with open(self.stanza_config, 'r', encoding="utf-8") as f:
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

    # ========================================== METHODS =================================================

    @chrono
    def order(self) -> pd.DataFrame:
        """
        Trier le dataframe dans l'ordre croissant des files_id
        """
        df = self.df.copy()

        df["first_file_id"] = df["files_id"].apply(lambda x: x[0])
        df = df.sort_values(by="first_file_id", ascending=True).reset_index(drop=True)
        df = df.drop(columns=["first_file_id"])

        return df


    @chrono
    def run(self, data:pd.DataFrame) -> pd.DataFrame:
        """Make a DataFrame with the result of Stanza analyses (bulk processing)"""
        self.load_data(data)
        window_size = self.config["description_window"]

        if self.verbose:
            print(f"[stanza] Stanza version: {stanza.__version__}")
            print(f"[stanza] Pipeline lang: {self.nlp.lang}")

        # Concaténer toutes les descriptions avec des séparateurs uniques
        text_blocks = []
        offsets = []
        current_offset = 0

        for _, row in self.data.iterrows():
            if not isinstance(row["desc"], str):
                continue
            text_blocks.append(row["desc"])
            offsets.append((row["files_id"], current_offset, len(row["desc"])))
            current_offset += len(row["desc"]) + 1  # +1 for newline separator

        full_text = "\n".join(text_blocks)
        doc = self.nlp(full_text)

        rows = []

        for ent in doc.ents:
            ent_start = ent.start_char
            ent_end = ent.end_char

            # Identifier à quel bloc cette entité appartient
            for files_id, block_start, block_len in offsets:
                block_end = block_start + block_len

                if block_start <= ent_start < block_end:
                    relative_start = ent_start - block_start
                    relative_end = ent_end - block_start

                    if self.production_mode:
                        rows.append({
                        "NE": ent.text,
                        "label": ent.type,
                        "method": "stanza",
                        "files_id": files_id,
                        "pos" : (relative_start, relative_end),
                    })
                    else:
                        row = self.data[self.data["files_id"] == files_id].iloc[0]
                        text = row["desc"]
                        context_start = max(relative_start - window_size, 0)
                        context_end = min(relative_end + window_size, len(text))
                        context_window = text[context_start:context_end]

                        rows.append({
                            "titles": row["titles"],
                            "NE": ent.text,
                            "label": ent.type,
                            "desc": context_window,
                            "method": "stanza",
                            "files_id": files_id,
                            "pos" : (relative_start, relative_end),
                        })
                    break  # une entité ne peut appartenir qu'à un seul bloc

        self.df = pd.DataFrame(rows)

        if self.order_dataframe:
            self.df = self.order()

        if self.verbose:
            print(f"Stanza DataFrame shape: {self.df.shape}")

        return self.df