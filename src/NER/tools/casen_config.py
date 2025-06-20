from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import shutil
import re
from bs4 import BeautifulSoup
import yaml


class CasenConfig:

    def __init__(self, 
                 run_casen:bool=True,
                 single_corpus:bool=True, 
                 production_mode:bool=True,
                 remove_misc:bool=True, 
                 logging:bool=False,
                 timer:bool=False,
                 archiving_result:bool=False,
                 casen_config:str= Path(__file__).parent.parent / "config.yaml", 
                 verbose:bool=False
                 ):
        self.casen_config = Path(casen_config)
        self.production_mode = production_mode
        self.remove_misc = remove_misc
        self.single_corpus = single_corpus
        self.run_casen = run_casen
        self.logging = logging
        self.timer = timer
        self.archiving_result = archiving_result
        self.verbose = verbose


        
        # Load the CONFIG
        self.load_config()
        
    
    def log(self, step:str, duration:float):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        log_folder = Path(self.config["log_folder"])
        if log_folder.is_dir(): 
            log_file_path = log_folder / "log.txt" # if the path is a folder -> add a filename
        else:
            log_file_path = log_folder
       

        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{timestamp} - CasEN [{step}] finish in {duration:.2f} s.\n")

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
    
    @chrono
    def load_config(self):
        """Load the JSON config about casEN"""

        if not self.casen_config.is_file():
            raise FileNotFoundError(f"[load config] The provided file was not found ! {self.casen_config}")
        else:
            with open(self.casen_config, 'r', encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                if self.verbose: print(f"[load config] Config Loaded sucessfuly !")
    
    @chrono
    def load_data(self, data:pd.DataFrame | str):
        """Load the data"""
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            self.data = pd.read_excel(data)
        else:
            raise ValueError(f"Can't make DataFrame with the provided data ! {data}")

    @chrono
    def load_files(self) -> list[Path]:
        """Load CasEN result file(s)"""

        result_folder = Path(self.config["result_folder"])
        files = list(result_folder.glob("*.txt"))

        file_counts = len(files)
        if file_counts == 0:
            raise Exception(f"[casen] No file(s) to load")
        else:
            if self.verbose:
                print(f"[casen] {file_counts} file(s) loaded")
        
        return files
    
    def prepare_folder(self, name:str, folder_to_prepare:Path) -> str:
        """Clean the folder before CasEN analyse"""
        
        files = list(folder_to_prepare.iterdir())
        archive_folder = Path(self.config["archive_folder"])
        if not files and self.verbose:
            print(f"[prepare folder] Empty folder : {folder_to_prepare}")
        elif archive_folder and folder_to_prepare == "result_folder":
            # Make a directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = archive_folder / f"{timestamp}_{name}"
            target.mkdir()

            if self.verbose:
                print(f"[prepare folder] Archiving file(s) in : {target}")

            for file in files:
                try:
                    shutil.move(str(file), str(target/ file.name))
                    if self.verbose:
                        print(f"[prepare folder] Moved: {file.name} to {target}")
                except Exception as e:
                    print(f"[prepare folder] Failed to move {file}: {e}")

        else:
            for file in files:
                try:
                    if file.is_file() or file.is_symlink():
                        file.unlink()
                        if self.verbose:
                            print(f"[prepare folder] Deleted file : {file.name}")
                    elif file.is_dir():
                        shutil.rmtree(file)
                        if self.verbose:
                            print(f"[prepare folder] Deleted folder : {file.name}")
                except Exception as e:
                    print(f"[prepare folder] Failed to delete {file} : {e}")

    @chrono
    def generate_corpus(self):
        """ Generate CasEN corpus"""

        missing_desc = self.data["desc"].isna().sum()
        corpus_folder = Path(self.config["corpus_folder"])

        if self.single_corpus:
            corpus_file = corpus_folder / "corpus.txt"
            with open(corpus_file, 'w', encoding="utf-8") as f:
                for idx, row in self.data.iterrows():
                    desc = row["desc"]
                    f.write(f'<doc id="{idx}">')
                    f.write(str(desc) if not pd.isna(desc) else "")
                    f.write('</doc>\n')
            if self.verbose:
                print(f"[generate file(s)] Single corpus file generated: {corpus_file}")
                print(f"[generate file(s)] Missing description : {missing_desc}")

        else:
            for idx, row in self.data.iterrows():
                filename = corpus_folder / f"corpus_{idx}.txt"
                with open(filename, 'w', encoding="utf-8") as f:
                    desc = row["desc"]
                    f.write(f'<doc id="{idx}">')
                    f.write(str(desc) if not pd.isna(desc) else "")
                    f.write('</doc>\n')
            if self.verbose:
                print(f"[generate file(s)] {len(self.data)} individual corpus files generated in {corpus_folder}")
                print(f"[generate file(s)] Missing description : {missing_desc}")

    @chrono
    def run_casEN_on_corpus(self):
        """Run CasEN to analyse descriptions"""
        self.generate_corpus()
        get_ipython().run_line_magic('run', str(self.config["ipynb_file"]))

    def get_label(self, tag) -> str:
        """Return the appropriated label"""
        labels = self.config["labels"]
        if tag in labels["PER"]:
            return "PER"
        elif tag in labels["LOC"]:
            return "LOC"
        elif tag in labels["ORG"]:
            return "ORG"
        else:
            return "MISC"
    
    def get_entities_from_desc(self, soup_doc: BeautifulSoup, doc_id: int) -> list[dict]:
        entities = []
        desc = self.data.loc[doc_id, "desc"]
        char_pos = 0

        for elem in soup_doc.find_all(lambda tag: tag.has_attr("grf") and tag.name not in ["s", "p", "doc"]):
            if any(parent.has_attr("grf") for parent in elem.parents if parent.name not in ["s", "p", "doc"]):
                continue  #take only parent element

            # Loop through tags in reading order to find each entity's exact character position,
            # using the previous end position (char_pos) to avoid matching earlier duplicate occurrences.
            entity_text = elem.get_text()
            start = desc.find(entity_text, char_pos)

            if start == -1:
                start = desc.find(entity_text)
                if start == -1:
                    continue
            end = start + len(entity_text)

            main_grf = elem["grf"]
            children = [child for child in elem.find_all(recursive=False) if child.has_attr("grf")]
            second = children[0]["grf"] if len(children) >= 1 else ""
            third = children[1]["grf"] if len(children) >= 2 else ""

            entities.append({
                "file_id":       doc_id,
                "tag":           elem.name,
                "text":          entity_text,
                "grf":           main_grf,
                "second_graph":  second,
                "third_graph":   third,
                "entity_start":  start,
                "entity_end":    end
            })

            char_pos = end

        return entities
    
    @chrono
    def get_entities(self) -> list[dict]:
        """Return every entities founds in corpus text(s)"""
        entities = []

        if not self.files:
            raise Exception(f"CasEN results were never loaded")
        
        for file in self.files:
            with open(file, 'r', encoding="utf-8") as f:
                content = f.read()

            content = re.sub(r'</?s\b[^>]*>', '', content) # remove <s> & </s>
            content = re.sub(r"</?s>", "", content)

            soup = BeautifulSoup(content, "html.parser")
            for doc in soup.find_all("doc"):
                doc_id = int(doc.attrs.get("id"))
                entities.extend(self.get_entities_from_desc(doc, doc_id))

        return entities
    
    @chrono
    def CasEN(self) -> pd.DataFrame:
        """Build a DataFrame with CasEN analyse"""

        window_size = self.config["description_window"]

        if self.data is None:
            self.load_data()
        
        entities = self.get_entities()
        rows = []
        for entity in entities:
            ner_label = self.get_label(entity["tag"])
            if self.remove_misc and ner_label == "MISC":
                continue
            else:
                file_id = entity["file_id"]
                ner =  entity["text"]

                entity_start = int(entity.get("entity_start"))
                entity_end = int(entity.get("entity_end"))
                
                if self.production_mode:
                    rows.append({
                    "NER" : ner,
                    "NER_label" : ner_label,
                    "method" : "casEN",
                    "main_graph" : entity["grf"],
                    "second_graph" : entity.get("second_graph", ""),
                    "third_graph" : entity.get("third_graph", ""),
                    "file_id" : file_id,
                    "entity_start" : entity_start,
                    "entity_end" : entity_end,
                })
                else:
                    desc = self.data.loc[file_id,"desc"]
                    start = max(entity_start - window_size, 0)
                    end = min(entity_end + window_size, len(desc))
                    context_window = desc[start:end]

                    rows.append({
                        "titles" : self.data.loc[file_id, "titles"],
                        "NER" : ner,
                        "NER_label" : ner_label,
                        "desc" : context_window,
                        "method" : "casEN",
                        "main_graph" : entity["grf"],
                        "second_graph" : entity.get("second_graph", ""),
                        "third_graph" : entity.get("third_graph", ""),
                        "file_id" : file_id,
                        "entity_start" : entity_start,
                        "entity_end" : entity_end,
                    })

        self.df = pd.DataFrame(rows)
        return self.df

    @chrono
    def run(self, data:pd.DataFrame | str) ->pd.DataFrame:
        """Run everything to make a DataFrame with CasEN analyse"""

        # Load data
        self.load_data(data)

        # Clean the Corpus folder and the CasEN result folder before running CasEN
        if self.run_casen:
            self.prepare_folder("corpus", Path(self.config["corpus_folder"]))
            self.prepare_folder("results",Path(self.config["result_folder"]))
            self.run_casEN_on_corpus()

        # Take all CasEN result files
        self.files = self.load_files()
        # Generate DataFrames
        self.CasEN()

        if self.verbose:
            print(f"CasEN : {self.df.shape}")
            print(f"Memory usage : {self.df.memory_usage(deep=True).sum()} bytes")

        return self.df
