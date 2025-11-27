from pathlib import Path
import yaml
import pandas as pd
from bs4 import BeautifulSoup, NavigableString
import re
import ast
from datetime import datetime
import shutil
import lxml
import time
from datetime import datetime

class CasEN:

    def __init__(self,
                 generate_new_corpus : bool = True,
                 corpus_mode : str = "single",
                 lightmode : bool = True,
                 include_tags_name : bool = False,
                 define_label_with_grf: bool = True,
                 grf_limit : bool = None,
                 remove_undefined_labels : bool = True,
                 config_path: Path = Path(__file__).parent.parent / "config.yaml",
                 archiving_result : bool = False,
                 auto_analyse : bool = True,
                 order_dataframe : bool = False,
                 timer : bool = True,
                 logging : bool = False,
                 verbose:bool=False
                 ):
        self.generate_new_corpus = generate_new_corpus
        self.corpus_mode = corpus_mode
        self.lightmode = lightmode
        self.include_tags_name = include_tags_name
        self.define_label_with_grf = define_label_with_grf
        self.grf_limit = grf_limit
        self.remove_undefined_labels = remove_undefined_labels
        self.config_path = config_path
        self.archiving_result = archiving_result
        self.auto_analyse = auto_analyse
        self.order_dataframe = order_dataframe
        self.timer = timer
        self.logging = logging
        self.verbose = verbose

    # TOOLS

    def chrono(func):
        def wrapper(self, *args, **kwargs):
            if self.timer or self.logging:
                start = time.time()
            result = func(self, *args, **kwargs)
            if self.timer or self.logging:
                duration = time.time() - start
                if self.timer:
                    print(f"[{func.__name__}] run in : {duration:.2f}s")
                if self.logging:
                    self.log(func.__name__, duration)
            return result
        return wrapper
    
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
    
    # CODE

    def get_config(self):
        """Load the JSON config about casEN"""

        if not self.config_path.is_file():
            raise FileNotFoundError(f"[get_config] The provided file was not found ! {self.config_path}")
        else:
            with open(self.config_path, 'r', encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                if self.verbose: print(f"[get_config] Config Loaded sucessfuly !")

    def get_files(self) -> list[Path]:
        """Recuperer tous les fichiers .txt d'un dossier"""
        # Recuperer le dossier 
        folder = Path(self.config["result_folder"])
        # Verifier que c'est bien un dossier
        if folder.is_dir():
            files = list(folder.glob("*.txt"))
            if self.verbose:
                print(f"[get_files] Founds {len(files)} .txt files.")
            return files        
        else:
            raise NotADirectoryError(f"The provided path is not a folder : {folder}")

    @staticmethod
    def prepare_folder(name:str, folder_to_prepare:Path, config:dict, verbose:bool) -> str:
        """Clean the folder before CasEN analyse"""
        
        files = list(folder_to_prepare.iterdir())
        archive_folder = Path(config["archive_folder"])
        if not files and verbose:
            print(f"[prepare folder] Empty folder : {folder_to_prepare}")
        elif archive_folder and folder_to_prepare == "result_folder":
            # Make a directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = archive_folder / f"{timestamp}_{name}"
            target.mkdir()

            if verbose:
                print(f"[prepare folder] Archiving file(s) in : {target}")

            for file in files:
                try:
                    shutil.move(str(file), str(target/ file.name))
                    if verbose:
                        print(f"[prepare folder] Moved: {file.name} to {target}")
                except Exception as e:
                    print(f"[prepare folder] Failed to move {file}: {e}")

        else:
            for file in files:
                try:
                    if file.is_file() or file.is_symlink():
                        file.unlink()
                        if verbose:
                            print(f"[prepare folder] Deleted file : {file.name}")
                    elif file.is_dir():
                        shutil.rmtree(file)
                        if verbose:
                            print(f"[prepare folder] Deleted folder : {file.name}")
                except Exception as e:
                    print(f"[prepare folder] Failed to delete {file} : {e}")

    @chrono
    def generate_corpus(self):
        """Generate the appropriate corpus"""

        if self.corpus_mode not in ["single", "multiple", "collection"]:
            raise ValueError(f"Invalid corpus_mode: '{self.corpus_mode}'. Expected one of: 'single', 'collection', 'multiple'.")

        # Recuperer le chemin de destination du Corpus
        corpus_path = Path(self.config["corpus_folder"])

        match self.corpus_mode:
            case "single":
                path = corpus_path /  "corpus.txt"
                with open(path, 'w', encoding="utf-8") as f:
                    f.write(f'<root>')
                    for idx, row in self.raw_data.iterrows():
                        f.write(f'<doc id="{str(row["files_id"])}">{row["desc"]}</doc>\n')
                    f.write(f'</root>')
            case "multiple":
                for idx, row in self.raw_data.iterrows():
                    path = corpus_path / f"row_{idx}.txt"
                    with open(path, 'w', encoding="utf-8") as f:
                        f.write(f'<root>')
                        f.write(f'<doc id="{str(row["files_id"])}">{row["desc"]}</doc>\n')
                        f.write(f'</root>')

            case "collection":
                if not "collection_id" in self.raw_data.columns:
                    raise KeyError(f'Missing column :  collection_id,  to process.')
                grouped = self.raw_data.groupby("collection_id")
                for collection_id, group in grouped:
                    path = corpus_path / f"collection_{collection_id}.txt"
                    with open(path, 'w', encoding="utf-8") as f:
                        f.write(f'<root>')
                        for idx, row in group.iterrows():
                            f.write(f'<doc id="{row["files_id"]}">{row["desc"]}</doc>\n')
                        f.write(f'</root>')
                        
    @chrono
    def run_casen(self):
        """Run CasEN to analyse descriptions"""
        self.generate_corpus()
        get_ipython().run_line_magic('run', str(self.config["ipynb_file"]))

    @staticmethod
    def clean_description(desc:str) -> str:
        """Remove every <s>, </s> from desc"""
        desc = re.sub(r'</?s\b[^>]*>', '', desc)
        desc = re.sub(r"</?s>", "", desc)
        return desc
    
    @staticmethod
    def clean_balise(balise:str) -> list[int]:
        """Transform the balise str into a list of int"""

        # Netoyer les \ qui se mettent apres analyse
        balise = re.sub(r'\\', "", balise)

        try:
            result = ast.literal_eval(balise)
            if isinstance(result, tuple) and all(isinstance(x, int) for x in result):
                return result
            else:
                raise ValueError(f"Probleme dans la balise : {balise}")
            
        except (ValueError, SyntaxError) as e:
            print(f"Erreur dans le parsing de la balise : {balise} -> {e}")
            return []

    # define the label of the entities
    def get_entity_label_tag(self, balise_name:str) -> str:
        """Return the Label of the entity"""
        for label, balise in self.config["labels"].items():
            if balise_name in balise:
                return label
        return "Undefined"
    
    def get_entity_label_grf(self, grf_name:str) -> str:
        """Return the Label of the entity"""
        for label, grf in self.config["labels_grf"].items():
            if grf_name in grf:
                return label
        return "Undefined"


    # extract entities
    def get_entities(self, desc: str | BeautifulSoup, id: list[int]) -> list[dict]:
        """
        Return a list of every outermost entity found in a description,
        avec leurs positions character-offset (start, end) dans le texte brut.
        """
        
        if isinstance(desc, str):
            desc = BeautifulSoup(desc, "lxml-xml")

        ignored = {"root", "doc", "s", "p", "html", "body", "[document]"}

        entities: list[dict] = []
        offset = 0
        text_content = desc.get_text()

        def recurse(node):
            nonlocal offset, entities  #nonlocal permet de modifier les variables definis dans la fonction parent (sinon ca creer des variables propre a cette fonction)

            # Si c'est du texte on ajoute la taille dans l'offset
            if isinstance(node, NavigableString):
                offset += len(node)
                return

            # Si c'est une balise on regarde si on dois l'ignorer ou pas 
            name = node.name if node.name else None
            is_candidate = name.lower() not in ignored and not any(
                (parent.name.lower() not in ignored) for parent in node.parents
            )

            if is_candidate:
                start = offset

            # On visite toujours les enfants (pour faire avancer offset)
            for child in node.children:
                recurse(child)

            if is_candidate:
                end = offset

                # on récolte les grf de la racine + descendants
                grfs = []
                if "grf" in node.attrs:
                    grfs.append((node.name, node.attrs["grf"]))
                for child in node.find_all(attrs={"grf": True}):
                    grfs.append((child.name, child.attrs.get("grf")))

                label = self.get_entity_label_grf(node.attrs.get("grf")) if self.define_label_with_grf else self.get_entity_label_tag(node.name)


                # description
                window_size = self.config["description_window"]
                left = max(0, start - window_size)
                right = min(len(text_content), end + window_size)
                context = text_content[left:right]

                # on construit le dict
                if (self.remove_undefined_labels and label != "Undefined") or not self.remove_undefined_labels:
                    entity = {
                        "NE": node.get_text(),
                        "label": label,
                        "files_id": id,
                        "pos" : (start, end),
                        "method" : "casEN",
                    }

                    # ajoute la desc
                    if not self.lightmode:
                        entity["desc"] = context
                        
                    # on ajoute les colonnes grf_i[_tag]
                    if not grfs:
                        if self.include_tags_name:
                            entity["tag_1"] = node.name
                    else:
                        limit = self.grf_limit if self.grf_limit is not None else len(grfs)
                        for idx, (tag, grf) in enumerate(grfs[:limit], start=1):
                            if self.include_tags_name:
                                entity[f"tag_{idx}"] = tag
                            entity[f"grf_{idx}"] = grf

                    

                    entities.append(entity)

        #  On lance la recursion sur les enfants directs de <doc>
        for child in desc.children:
            recurse(child)

        return entities
    
    def analyse_file(self, file_content:str) -> list[dict]:
        """Return a list of every entities found in the file"""
        file_entities = []

        # Nettoyer les descriptions
        clean_desc = self.clean_description(file_content)

        # Recuperer les balises des descriptions
        soup = BeautifulSoup(clean_desc, "lxml-xml")
        for doc in soup.find_all("doc"):
            id = doc.attrs.get("id")
            clean_id = self.clean_balise(id)
            # print(f"desc : {doc}")
            # print(f"id : {balise}")
            file_entities.extend(self.get_entities(doc, clean_id))
            
        return file_entities

    @chrono
    def analyse_files(self) -> pd.DataFrame:
        """Generate a DataFrame with CasEN entities"""
        if len(self.files) == 0:
            raise FileNotFoundError(f"Can't generate DataFrame, missing CasEN results files !")
        else:
            rows = []
            for file in self.files:
                with open(file, 'r', encoding="utf-8") as f:
                    content = f.read()
                entities = self.analyse_file(content)
                rows.extend(entities)

            df = pd.DataFrame(rows)

            return df

    # pre-analyse the resutls
    @chrono
    def self_analyse(self):
        """Analyse generated DataFrame"""
        print(f"#################################### CASEN DATAFRAME ####################################")
        mem_bits = self.df.memory_usage(deep=True).sum()
        mem_mo = mem_bits / 8 / 1024 / 1024
        print(f"DataFrame size : {mem_mo:.2f} Mo ({mem_bits} bits), shape: {self.df.shape}")

        print(f"Total NE founds : {self.df['NE'].count()}")
        unique_ne_count = self.df[['NE', 'files_id']].assign(files_id=self.df['files_id'].apply(lambda x: tuple(x) if isinstance(x, list) else x)).drop_duplicates().shape[0]
        print(f"Unique NE founds (by NE + files_id): {unique_ne_count}")

        labels = self.df['label'].value_counts().to_dict()
        print(" ---- Labels founds ---- :")
        for label, count in labels.items():
            print(f"{label:<10} : {count} ({count/self.df.shape[0] * 100:.2f}%)")

        if self.include_tags_name:
            tag_columns = [col for col in self.df.columns if col.startswith('tag_')]
            tags_flat = self.df[tag_columns].melt(value_name="tag")["tag"].dropna()
            tags_flat.name = None

            tag_counts = tags_flat.value_counts()
            total_tags = tag_counts.sum()

            print(f" ---- Nametag frequency ---- :")
            for tag, count in tag_counts.items():
                percentage = (count / total_tags) * 100
                print(f"{tag:<20} : {count} ({percentage:.2f}%)")
            print()

        grf_columns = [col for col in self.df.columns if col.startswith('grf_')]
        grfs_flat = self.df[grf_columns].melt(value_name="grf")["grf"]
        grfs_flat.name = None
        grf_counts = grfs_flat.value_counts()
        total_grfs = grf_counts.sum()

        print(f"---- Graphs frequency ----:")
        for grf, count in grf_counts.items():
            percentage = (count / total_grfs) * 100
            print(f"{grf:<20} : {count} ({percentage:.2f}%)")
        print()


        print(f"#########################################################################################")

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
    def run(self, raw_data:pd.DataFrame) -> pd.DataFrame:
        """Run CasEN and make a DataFrame with results"""

        # Init CasEN Config
        self.get_config()
        # Load raw datas
        self.raw_data = raw_data
        # Generate Corpus  & Run CasEN on corpus
        if self.generate_new_corpus:
            self.prepare_folder("corpus", Path(self.config["corpus_folder"]), self.config, self.verbose)
            self.prepare_folder("results",Path(self.config["result_folder"]), self.config, self.verbose)
            self.run_casen()

        # Load result Files
        self.files = self.get_files()

        # Entities Analyse
        self.df = self.analyse_files()

        if self.order_dataframe:
            self.df = self.order()

        # Auto analyse
        if self.auto_analyse:
            self.self_analyse()

        return self.df

