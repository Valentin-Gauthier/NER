from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import shutil
import re
from bs4 import BeautifulSoup

class CasEN:
    def __init__(self,path:str, corpus_folder:str, result_folder:str, data:str=None, remove_MISC:bool=True, archive_folder:str=None, single_corpus:bool=True, run_casEN:bool=False, logging:bool=True, log_folder:str=None, timer:bool=True, verbose:bool=False):
        self.path = path  # The path for CasEN ipynb 
        self.data = data  # The path for the Excel file
        self.corpus_folder = Path(corpus_folder)  # The path for the folder of CasEN corpus
        self.remove_MISC = remove_MISC
        self.result_folder = Path(result_folder) # The path for the folder of CasEN result
        self.archive_folder = archive_folder # option to archive files before erase them with casEN corpus and results
        self.single_corpus = single_corpus
        self.run_casEN = run_casEN
        self.logging = logging
        self.timer = timer
        self.log_folder = log_folder
        self.verbose = verbose

        # Check every provided folder
        if self.corpus_folder is not None:
            self.check_folder(self.corpus_folder)
        if self.result_folder is not None:
            self.check_folder(self.result_folder)
        if self.archive_folder is not None:
            self.archive_folder = Path(self.archive_folder)
            self.check_folder(self.archive_folder)
        if self.log_folder is not None:
            self.log_folder = Path(self.log_folder)
            self.check_folder(self.log_folder)
            

        # Load the Excel file
        if isinstance(self.data, str):
            self.data = Path(self.data)
            self.data_df = self.load_data()

    # ---------------------------- TOOLS ----------------------- #
    def log(self, step:str, duration:float):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        if self.log_folder.is_dir(): 
            log_file_path = self.log_folder / "log.txt" # if the path is a folder -> add a filename
        else:
            log_file_path = self.log_folder


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
    # ---------------------------------------------------------- #
    
    def check_folder(self,folder:Path) -> bool:
        """ Check if a folder exist """
        if not folder.is_dir():
            raise NotADirectoryError(f"[prepare folder] The provided path is not a folder : {folder}")
        if not folder.exists():
            raise FileNotFoundError(f"[prepare folder] The provided folder does not exist : {folder}")
        return True

    def load(self) -> list[Path]:
        """Load CasEN result file(s)"""

        files = list(self.result_folder.glob("*.txt"))

        file_counts = len(files)
        if file_counts == 0:
            raise Exception(f"[load] No file(s) to load")
        else:
            if self.verbose:
                print(f"[load] {file_counts} file(s) loaded")
        
        return files

    @chrono
    def load_data(self) -> pd.DataFrame:
        """Load Excel file"""
        if not self.data.is_file():
            raise FileNotFoundError(f"The Excel file doesn't exist : {self.data}")
        df = pd.read_excel(self.data)
        df["desc"] = df["desc"].fillna("") # clean the empty description

        return df

    @chrono
    def generate_corpus(self):
        """ Generate CasEN corpus"""

        missing_desc = (self.data_df["desc"] == "").sum()

        if self.single_corpus:
            corpus_file = self.corpus_folder / "corpus.txt"
            with open(corpus_file, 'w', encoding="utf-8") as f:
                for idx, row in self.data_df.iterrows():
                    f.write(f'<doc id="{idx}">')
                    f.write(str(row["desc"]))
                    f.write('</doc>\n')
            if self.verbose:
                print(f"[generate file(s)] Single corpus file generated: {corpus_file}")
                print(f"[generate file(s)] Missing description : {missing_desc}")

        else:
            for idx, row in self.data_df.iterrows():
                filename = self.corpus_folder / f"doc_{idx}.txt"
                with open(filename, 'w', encoding="utf-8") as f:
                    f.write(f'<doc id="{idx}">')
                    f.write(str(row["desc"]))
                    f.write('</doc>\n')
            if self.verbose:
                print(f"[generate file(s)] {len(self.data_df)} individual corpus files generated in {self.corpus_folder}")
                print(f"[generate file(s)] Missing description : {missing_desc}")

    @chrono
    def run_casEN_on_corpus(self):
        """Run CasEN to analyse descriptions"""
        self.generate_corpus()
        get_ipython().run_line_magic('run', str(self.path))

    def prepare_folder(self, name:str, folder_to_prepare:Path) -> str:
        """Clean the folder before CasEN analyse"""
        
        files = list(folder_to_prepare.iterdir())
        if not files and self.verbose:
            print(f"[prepare folder] Empty folder : {folder_to_prepare}")
        elif self.archive_folder:
            # Make a directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = self.archive_folder / f"{timestamp}_{name}"
            target.mkdir()

            if self.verbose:
                print(f"[prepare folder] Archiving file(s) in : {target}")

            for file in files:
                try:
                    shutil.move(str(file), str(target/ file.name))
                    if self.verbose:
                        print(f"[prepare folder] Moved: {file.name} → {target}")
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

    def get_label(self, tag) -> str:
        """Return the appropriated label"""
        if tag in ["persname", "surname", "forename"]:
            return "PER"
        elif tag in ['placename','geoname', 'adress', 'adrline']:
            return "LOC"
        elif tag in ["orgname", "org"]:
            return "ORG"
        else:
            return "MISC"

    def get_context(self, desc:str, ner:str, window:int) -> str:
        """Return the context of entity"""
        tokenized_desc = re.findall(r"\w+|[^\w\s]", desc.lower())
        tokenized_ner = re.findall(r"\w+|[^\w\s]", ner.lower())
        len_ner = len(tokenized_ner)

        for i in range(len(tokenized_desc) - len_ner + 1):
            if tokenized_desc[i:i + len_ner] == tokenized_ner:
                start = max(0, i - window)
                end = min(len(tokenized_desc), i + len_ner + window)
                return ' '.join(tokenized_desc[start:end])
        return desc

    def get_entities_from_desc(self, soup_doc:BeautifulSoup, doc_id:int) -> list[dict]:
        """Return every entities founds in one description"""
        entities = []
        excluded_tags = ["s", "p", "doc"]
        for element in soup_doc.find_all(lambda tag : tag not in excluded_tags and tag.has_attr("grf")):
            if any(ancestor.has_attr("grf") for ancestor in element.parents if ancestor.name not in excluded_tags):
                continue
            main_grf = element["grf"]
            children = [child for child in element.find_all(recursive=False) if child.has_attr("grf")]
            second = children[0]["grf"] if len(children) >= 1 else ""
            third  = children[1]["grf"] if len(children) >= 2 else ""

            entities.append({
                "file_id":       doc_id,
                "tag":           element.name,
                "text":          element.get_text(),
                "grf":           main_grf,
                "second_graph":  second,
                "third_graph":   third
            })

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
            content = re.sub(r'</?s\b[^>]*>', '', content)
            soup = BeautifulSoup(content, "html.parser")
            for doc in soup.find_all("doc"):
                doc_id = int(doc.attrs.get("id"))
                entities.extend(self.get_entities_from_desc(doc, doc_id))

        return entities

    @chrono
    def CasEN(self) -> pd.DataFrame:
        """Build a DataFrame with CasEN analyse"""
        
        if self.data_df is None:
            self.load_data()
        
        entities = self.get_entities()
        rows = []
        for entity in entities:
            ner_label = self.get_label(entity["tag"])
            if self.remove_MISC and ner_label == "MISC":
                continue
            else:
                file_id = entity["file_id"]
                ner =  entity["text"]
                context = self.get_context(self.data_df.loc[file_id, "desc"], ner, window=10)
                rows.append({
                    "titles" : self.data_df.loc[file_id, "titles"],
                    "NER" : ner,
                    "NER_label" : ner_label,
                    "desc" : context,
                    "method" : "casEN",
                    "main_graph" : entity["grf"],
                    "second_graph" : entity.get("second_graph", ""),
                    "third_graph" : entity.get("third_graph", ""),
                    "file_id" : file_id
                })

        self.df = pd.DataFrame(rows)
        return self.df
        
    @chrono
    def run(self) -> pd.DataFrame:
        """Execute all steps to make a DataFrame with CasEN"""

        if self.run_casEN:
            self.prepare_folder("corpus",self.corpus_folder)
            self.prepare_folder("results",self.result_folder)
            self.run_casEN_on_corpus()

        self.files = self.load()
        self.CasEN()

        return self.df


        

        
