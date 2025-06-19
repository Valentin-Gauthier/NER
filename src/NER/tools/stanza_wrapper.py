import stanza
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import yaml


DEFAULT_CONFIG_LOC = Path(__file__).parent / "config.yaml"

class StanzaConfig:

    def __init__(self, 
                 use_gpu:bool=True, 
                 light_mode:bool=True,
                 stanza_config:str=DEFAULT_CONFIG_LOC,
                 timer:bool=False, 
                 logging:bool=False, 
                 verbose:bool=False
                 ):
        self.use_gpu = use_gpu
        self.light_mode = light_mode
        self.stanza_config = stanza_config
        self.verbose = verbose
        self.timer = timer
        self.logging = logging
    
        self.load_config()

        # Load the French Pipeline (tokenize : slice the text, mwt: usefull for french word like "ajourd'hui", ner : analyse the text)
        self.nlp = stanza.Pipeline(lang="fr", processors='tokenize,mwt,ner', use_gpu=self.use_gpu)


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

        for idx, row in self.data.iterrows():
            if not isinstance(row["desc"], str):
                continue
            text_blocks.append(row["desc"])
            offsets.append((idx, current_offset, len(row["desc"])))
            current_offset += len(row["desc"]) + 1  # +1 for newline separator

        full_text = "\n".join(text_blocks)
        doc = self.nlp(full_text)

        rows = []

        for ent in doc.ents:
            ent_start = ent.start_char
            ent_end = ent.end_char

            # Identifier à quel bloc cette entité appartient
            for file_id, block_start, block_len in offsets:
                block_end = block_start + block_len

                if block_start <= ent_start < block_end:
                    relative_start = ent_start - block_start
                    relative_end = ent_end - block_start

                    if self.light_mode:
                        rows.append({
                        "NER": ent.text,
                        "NER_label": ent.type,
                        "method": "stanza",
                        "file_id": file_id,
                        "entity_start": relative_start,
                        "entity_end": relative_end
                    })
                    else:
                        text = self.data.loc[file_id, "desc"]
                        context_start = max(relative_start - window_size, 0)
                        context_end = min(relative_end + window_size, len(text))
                        context_window = text[context_start:context_end]

                        rows.append({
                            "titles": self.data.loc[file_id, "titles"],
                            "NER": ent.text,
                            "NER_label": ent.type,
                            "desc": context_window,
                            "method": "stanza",
                            "file_id": file_id,
                            "entity_start": relative_start,
                            "entity_end": relative_end
                        })
                    break  # une entité ne peut appartenir qu'à un seul bloc

        self.df = pd.DataFrame(rows)

        if self.verbose:
            print(f"Stanza DataFrame shape: {self.df.shape}")

        return self.df































# import stanza
# import pandas as pd
# import torch
# import re
# import time
# from datetime import datetime
# from pathlib import Path

# class Stanza:
    
#     def __init__(self, data:str=None, use_gpu:bool=False, logging:bool=True, log_folder:str=None, timer:bool=True, verbose:bool=False):
#         self.data = data
#         self.gpu = use_gpu
#         self.timer = timer
#         self.logging = logging
#         self.log_folder = Path(log_folder)
#         self.verbose = verbose

#         if isinstance(self.data, str):
#             self.data_df = pd.read_excel(self.data)
#         else:
#             print("Pas de fichier")

        # # Load the French Pipeline (tokenize : slice the text, mwt: usefull for french word like "ajourd'hui", ner : analyse the text)
        # self.nlp = stanza.Pipeline(lang="fr", processors='tokenize,mwt,ner', use_gpu=self.gpu)

#         if self.verbose:
#             print("PyTorch version :", torch.__version__)
#             print("CUDA available :", torch.cuda.is_available())
#             print("CUDA version (torch):", torch.version.cuda)
#             print("Device name :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


#     def log(self, step:str, duration:float):
#         timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

#         if self.log_folder.is_dir(): 
#             log_file_path = self.log_folder / "log.txt" # if the path is a folder -> add a filename
#         else:
#             log_file_path = self.log_folder


#         log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
#         with open(log_file_path, 'a', encoding="utf-8") as f:
#             f.write(f"{timestamp} - Stanza [{step}] finish in {duration:.2f} s.\n")

#     def chrono(func):
#         def wrapper(self, *args, **kwargs):
#             if self.timer or self.logging:
#                 start = time.time()
#             result = func(self, *args, **kwargs)
#             if self.timer or self.logging:
#                 duration = time.time() - start
#                 if self.timer:
#                     print(f"{func.__name__} in : {duration:.2f}s")
#                 if self.logging:
#                     self.log(func.__name__, duration)
#             return result
#         return wrapper


#     def get_context(self, desc: str, ner: str, window: int) -> str:
#         """Return the original text context of an entity with surrounding words"""
#         # Tokenize with positions
#         matches = list(re.finditer(r"\w+|[^\w\s]", desc))
#         tokens = [m.group(0) for m in matches]
#         positions = [(m.start(), m.end()) for m in matches]

#         # Tokenize ner in same way
#         ner_tokens = re.findall(r"\w+|[^\w\s]", ner)
#         len_ner = len(ner_tokens)

#         # Match ner sequence in token list
#         for i in range(len(tokens) - len_ner + 1):
#             if [t.lower() for t in tokens[i:i + len_ner]] == [n.lower() for n in ner_tokens]:
#                 start_idx = max(0, i - window)
#                 end_idx = min(len(tokens), i + len_ner + window)

#                 start_char = positions[start_idx][0]
#                 end_char = positions[end_idx - 1][1]

#                 return desc[start_char:end_char]
#         return desc
    
#     @chrono
#     def run(self):
#         """Make a DataFrame with the result of Stanza analyses"""
#         merged_text = ""
#         idx_map = []  # (start_char, end_char, idx, row)
#         current_pos = 0

#         for idx, row in self.data_df.iterrows():
#             desc = row.get("desc", "")
#             if not isinstance(desc, str):
#                 continue
#             marker = f"\n###DOCIDX={idx}###\n"
#             desc_block = marker + desc.strip() + "\n"
#             start = current_pos + len(marker)
#             end = start + len(desc.strip())
#             idx_map.append((start, end, idx, row))
#             merged_text += desc_block
#             current_pos += len(desc_block)

#         doc = self.nlp(merged_text)

#         rows = []
#         for ent in doc.ents:
#             ent_start = ent.start_char
#             ent_end = ent.end_char

#             # Trouver la description d’origine correspondant à l'entité
#             for start, end, idx, row in idx_map:
#                 if start <= ent_start < end:
#                     desc_text = self.data_df.loc[idx, "desc"]
#                     rows.append({
#                         "titles": row.get("titles", "Default title name"),
#                         "NER": ent.text,
#                         "NER_label": ent.type,
#                         "desc": self.get_context(desc_text, ent.text, 5),
#                         "method": "Stanza",
#                         "file_id": idx
#                     })
#                     break

#         self.df = pd.DataFrame(rows)

#         if self.verbose:
#             print(f"Stanza : {self.df.shape}")
#         return self.df