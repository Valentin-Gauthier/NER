from .spacy_wrapper import SpaCy
from .casen import CasEN
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import json

class NER:
    def __init__(self, spaCy:SpaCy | pd.DataFrame, casEN:CasEN | pd.DataFrame, data:str, casEN_priority_merge:bool, casEN_graph_validation:str, extent_optimisation:bool, remove_duplicate_rows:bool, ner_result_folder:str,excluded_names:str , correction:str=None, logging:bool=True, log_folder:str=None, timer:bool=True, verbose:bool=False):
        self.spaCy = spaCy
        self.casEN = casEN
        self.data = Path(data)
        self.ner_result_folder = Path(ner_result_folder)

        # ------------------ NER OPTIMISATION -------------------- #
        self.casEN_priority_merge = casEN_priority_merge     
        self.casEN_graph_validation = Path(casEN_graph_validation) if casEN_graph_validation else None
        self.remove_duplicate_rows = remove_duplicate_rows
        self.excluded_names = excluded_names
        self.extent_optimisation = extent_optimisation

        # ------------------- NER OPTIONS ----------------------- #
        self.correction = Path(correction) if correction else None
        self.logging = logging
        self.log_folder = Path(log_folder) if log_folder else None
        self.timer = timer
        self.verbose = verbose


        # -------- NER INIT ------- #
        if self.log_folder is not None:
            self.check_folder(self.log_folder)

    # -------- TOOLS ------------ #
    def log(self, step:str, duration:float):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        if self.log_folder.is_dir(): 
            log_file_path = self.log_folder / "log.txt" # if the path is a folder -> add a filename
        else:
            log_file_path = self.log_folder


        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{timestamp} - Pipeline [{step}] finish in {duration:.2f} s.\n")

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

    def check_folder(self,folder:Path) -> bool:
        """ Check if a folder exist """
        if not folder.is_dir():
            raise NotADirectoryError(f"[prepare folder] The provided path is not a folder : {folder}")
        if not folder.exists():
            raise FileNotFoundError(f"[prepare folder] The provided folder does not exist : {folder}")
        return True
    
    # ------------------------ #

    def get_method(self, row):
        """Return the methods"""
        if row['_merge'] == 'both':
            return "intersection"
        elif row['_merge'] == 'left_only':
            return row['method_spaCy']
        elif row['_merge'] == 'right_only':
            return row['method_casEN']
        else:
            raise Exception(f"[merge] Error in the get_method while")

    @chrono
    def merge(self) -> pd.DataFrame:
        """ Merge spaCy & CasEN results

            When SpaCy & casEN found same entity -> keep on with updated method column to "intersection"
            Can't use pandas merge because it make Cartesian product...
        """
        self.spaCy_df = self.spaCy_df.copy()
        self.casEN_df = self.casEN_df.copy()

        if self.verbose:
            print(f"[merge] CasEN's DataFrame : {self.casEN_df.shape}")
            print(f"[merge] SpaCy's DataFrame : {self.spaCy_df.shape}")

        # Ajoute un ID d'occurrence par groupe unique (NER, label, file)
        for df in [self.spaCy_df, self.casEN_df]:
            df['occurrence_id'] = df.groupby(['NER', 'NER_label', 'file_id']).cumcount()

        # Ajoute une clé unique pour le merge
        self.spaCy_df['key'] = self.spaCy_df[['NER', 'NER_label', 'file_id', 'occurrence_id']].apply(tuple, axis=1)
        self.casEN_df['key'] = self.casEN_df[['NER', 'NER_label', 'file_id', 'occurrence_id']].apply(tuple, axis=1)

        # Fusion exacte sur la clé, évite produit cartésien
        merged = pd.merge(
            self.spaCy_df,
            self.casEN_df,
            on='key',
            how='outer',
            suffixes=('_spacy', '_casen'),
            indicator=True
        )

        # Marque la méthode
        source_map = {
            'left_only': 'spaCy',
            'right_only': 'casEN',
            'both': 'intersection'
        }
        merged['method'] = merged['_merge'].map(source_map)

        # Fusionne les colonnes
        def fuse(col):
            return merged[f"{col}_spacy"].combine_first(merged[f"{col}_casen"])

        for col in ['NER', 'NER_label', 'file_id', 'desc', 'titles', 'main_graph', 'second_graph', 'third_graph']:
            if f"{col}_spacy" in merged.columns and f"{col}_casen" in merged.columns:
                merged[col] = fuse(col)
                merged.drop([f"{col}_spacy", f"{col}_casen"], axis=1, inplace=True)

        # Nettoie les colonnes techniques
        merged.drop(columns=['_merge', 'key', 'occurrence_id'], inplace=True, errors='ignore')

        # Trie pour lisibilité
        merged = merged.sort_values(by=["file_id", "NER", "method"]).reset_index(drop=True)

        final_columns = ["titles", "NER", "NER_label", "desc","method", "main_graph", "second_graph", "third_graph", "file_id"]
        merge_df = merged[final_columns]

        if self.verbose:
            print(f"[merge] Merge's DataFrame : {merge_df.shape}")
            source_counts = merge_df['method'].value_counts()
            casen_count = source_counts.get('casEN', 0)
            spacy_count = source_counts.get('spaCy', 0)
            intersection_count = source_counts.get('intersection', 0)

            print(f"[merge] SpaCy only      : {spacy_count} lignes")
            print(f"[merge] CasEN only      : {casen_count} lignes")
            print(f"[merge] Intersection    : {intersection_count} lignes")
                    

        self.df = merge_df
        return self.df

    @chrono
    def merge2(self) -> pd.DataFrame:
        """ Merge spaCy & CasEN result"""

        if self.verbose:
            print(f"[merge] CasEN's DataFrame : {self.casEN_df.shape}")
            print(f"[merge] SpaCy's DataFrame : {self.spaCy_df.shape}")
            df_not_empty = self.data_df[self.data_df["desc"] != ""]
            print(f"[merge] Data's DataFrame {self.data_df.shape} ")

    
        self.spaCy_df["key"] = self.spaCy_df[["NER", "NER_label", "file_id"]].apply(lambda x: tuple(x), axis=1)
        self.casEN_df["key"] = self.casEN_df[["NER", "NER_label", "file_id"]].apply(lambda x: tuple(x), axis=1)

        if self.verbose:
            # spaCy
            spa_counts = self.spaCy_df["key"].value_counts()
            spa_dup_keys = spa_counts[spa_counts > 1]
            if not spa_dup_keys.empty:
                print(f"[merge] {len(spa_dup_keys)} clés dupliquées dans spaCy avant merge (num occurrences > 1) :")
                # éventuellement afficher les 5 premières
                for k, cnt in spa_dup_keys.head(5).items():
                    print(f"    Key={k}  occurs {cnt} times")
            else:
                print("[merge] Aucune clé dupliquée dans spaCy avant merge.")

            # CasEN
            cas_counts = self.casEN_df["key"].value_counts()
            cas_dup_keys = cas_counts[cas_counts > 1]
            if not cas_dup_keys.empty:
                print(f"[merge] {len(cas_dup_keys)} clés dupliquées dans CasEN avant merge (num occurrences > 1) :")
                for k, cnt in cas_dup_keys.head(5).items():
                    print(f"    Key={k}  occurs {cnt} times")
            else:
                print("[merge] Aucune clé dupliquée dans CasEN avant merge.")



        merge_df = pd.merge(self.spaCy_df, self.casEN_df, on="key", how="outer", suffixes=["_spaCy", "_casEN"], indicator=True)

        # Fix shared columns
        merge_df["titles"] = merge_df["titles_spaCy"].combine_first(merge_df["titles_casEN"])
        merge_df["NER"] = merge_df["NER_spaCy"].combine_first(merge_df["NER_casEN"])
        merge_df["NER_label"] = merge_df["NER_label_spaCy"].combine_first(merge_df["NER_label_casEN"])
        merge_df["desc"] = merge_df["desc_spaCy"].combine_first(merge_df["desc_casEN"])
        merge_df["method"] = merge_df.apply(self.get_method, axis=1) # Update for intersection
        merge_df["file_id"] = merge_df["file_id_spaCy"].combine_first(merge_df["file_id_casEN"])

        if self.remove_duplicate_rows:
            merge_df.drop_duplicates(subset=["NER", "NER_label", "method", "main_graph", "second_graph", "third_graph", "file_id"], inplace=True)
            if self.verbose:
                print(f"[merge] Dropping duplicate rows")

        if self.verbose:
            post_counts = merge_df["key"].value_counts()
            multi_merge_keys = post_counts[post_counts > 1]
            if not multi_merge_keys.empty:
                print(f"[merge] Après merge, {len(multi_merge_keys)} clés apparaissent plusieurs fois (ligne > 1) :")
                for key, cnt in multi_merge_keys.head(5).items():
                    print(f"    Key={key}  apparaît {cnt} fois dans merge_df")
            else:
                print("[merge] Aucune clé multiple après merge.")

        merge_df = merge_df.sort_values(by=["file_id"], ascending=True).reset_index(drop=True)
        
        final_columns = ["titles", "NER", "NER_label", "desc","method", "main_graph", "second_graph", "third_graph", "file_id"]
        self.df = merge_df[final_columns]

        if self.verbose:
            files_with_entities = set(merge_df["file_id"])
            total_with_desc = len(df_not_empty)
            desc_without_entity = total_with_desc - len(files_with_entities)
            print(f"[merge] description without entities : {desc_without_entity}")

        return self.df

    @chrono
    def apply_correction(self) -> pd.DataFrame:
        """Auto correct """

        columns = ["manual cat", "extent", "correct", "category"]

        for col in columns:
            self.df[col] = ""

        correction_df = pd.read_excel(self.correction)

        correction_df["key"] = correction_df[["NER", "NER_label", "file_id"]].apply(lambda x: tuple(x), axis=1)
        self.df["key"] = self.df[["NER", "NER_label", "file_id"]].apply(lambda x: tuple(x), axis=1)
        
        correction_df = correction_df.drop_duplicates(subset=["key"])
        correction_dict = correction_df.set_index("key")[columns].to_dict(orient="index")

        for col in columns:
            self.df[col] = self.df["key"].map(lambda k: correction_dict.get(k, {}).get(col, ""))

        self.df.drop(columns=["key"], inplace=True)

        final_columns = ['manual cat', 'correct', 'extent', 'category','titles', 'NER', 'NER_label', 'desc', 'method',
                        'main_graph', 'second_graph', 'third_graph', "file_id"]
        
        self.df = self.df[final_columns]

        return self.df

    @chrono
    def casEN_optimisation(self) -> pd.DataFrame:
        """ Change the method casEN to casEN_opti when the graphs in the JSON trustable graphs"""
        with open(self.casEN_graph_validation, 'r', encoding="utf-8") as f:
            valid_graphs = json.load(f)

        def is_allowed(row):
            for combo in valid_graphs:
                if all(row.get(col) == val for col, val in combo.items()):
                    return True
            return False

        def upgrade_method(row):
            if row["method"] == "casEN" and is_allowed(row):
                return "casEN_opti"
            else:
                return row["method"]

        self.df["method"] = self.df.apply(upgrade_method, axis=1)

        self.df = self.df.reset_index(drop=True)
        if self.verbose:
            source_counts = self.df['method'].value_counts()
            casen_count = source_counts.get('casEN', 0)
            casen_opti_count = source_counts.get('casEN_opti', 0)
            spacy_count = source_counts.get('spaCy', 0)
            intersection_count = source_counts.get('intersection', 0)

            print(f"[casEN_optimisation] SpaCy only      : {spacy_count} lignes")
            print(f"[casEN_optimisation] CasEN only      : {casen_count} lignes")
            print(f"[casEN_optimisation] CasEN_opti only      : {casen_opti_count} lignes")
            print(f"[casEN_optimisation] Intersection    : {intersection_count} lignes")


        return self.df

    @chrono
    def casEN_priority(self) -> pd.DataFrame:
        """Keep entities founds by SpaCy & CasEN with differents categories with casEN_priority method"""

        spaCy_df = self.df[self.df["method"] == "spaCy"]
        casEN_df = self.df[self.df["method"] == "casEN"]

        merged = pd.merge(spaCy_df, casEN_df, on=["NER", "file_id"], suffixes=("_spacy","_casen"))

        conflicts = merged[merged["NER_label_spacy"] != merged["NER_label_casen"]]

        if self.verbose:
            print(f"[casEN_priority] {len(conflicts)} conflicting entities found (spaCy vs casEN)")

        with open(self.excluded_names, 'r', encoding="utf-8") as f:
            names = json.load(f)

        name_list = names[0].get("NER")

        new_rows = []
        for _, row in conflicts.iterrows():
            if row["NER_label_casen"] == "PER" and row["NER"].lower() not in [name.lower() for name in name_list]:
                new_rows.append({
                    "titles": row["titles_spacy"],
                    "NER": row["NER"],
                    "NER_label": row["NER_label_casen"],
                    "desc": row["desc_spacy"],           
                    "method": "casEN_priority",
                    "main_graph" : row["main_graph_casen"],
                    "second_graph" : row["second_graph_casen"],
                    "third_graph" : row["third_graph_casen"],
                    "file_id": row["file_id"]
                })
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            new_df.drop_duplicates(subset=["titles", "NER", "NER_label", "desc", "method", "main_graph", "second_graph", "third_graph", "file_id"], inplace=True)

            self.df = pd.concat([self.df, new_df], ignore_index=True)
            
            if self.verbose:
                print(f"[casEN_priority] {len(new_rows)} added.")

        self.df = self.df.sort_values(by=["file_id"], ascending=True).reset_index(drop=True)

        if self.verbose:
            source_counts = self.df['method'].value_counts()
            casen_count = source_counts.get('casEN', 0)
            casen_opti_count = source_counts.get('casEN_opti', 0)
            casen_priority_count = source_counts.get('casEN_priority', 0)
            spacy_count = source_counts.get('spaCy', 0)
            intersection_count = source_counts.get('intersection', 0)

            print(f"[casEN_priority] SpaCy only      : {spacy_count} lignes")
            print(f"[casEN_priority] CasEN only      : {casen_count} lignes")
            print(f"[casEN_priority] CasEN_opti only      : {casen_opti_count} lignes")
            print(f"[casEN_priority] CasEN_priority only      : {casen_priority_count} lignes")
            print(f"[casEN_priority] Intersection    : {intersection_count} lignes")

        return self.df

    @chrono
    def extent_optimisations(self) -> pd.DataFrame:
        """Try to optimise the extent of entities"""

        spaCy_df = self.df[self.df["method"] == "spaCy"]
        casEN_df = self.df[self.df["method"] == "casEN"]

        merge = pd.merge(spaCy_df, casEN_df, on=["file_id"], suffixes=["_spacy", "_casen"])

        def found_extent_conflict(row):
            ner_spacy = row["NER_spacy"]
            ner_casen = row["NER_casen"]

            return (
                ner_spacy != ner_casen and
                (ner_spacy in ner_casen) or (ner_casen in ner_spacy)
            )
        
        merge["extent_conflict"] = merge.apply(found_extent_conflict, axis=1)
        conflicts = merge[merge["extent_conflict"] == True]
        if self.verbose:
            print(f"[extent_optimisations] Number of extent conflicts: {len(conflicts)}")

        new_rows = []
        for _, row in conflicts.iterrows():
            if row["NER_label_casen"] in ["LOC", "ORG"]:
                new_rows.append({
                    "titles": row["titles_casen"],
                    "NER": row["NER_casen"],
                    "NER_label": row["NER_label_casen"],
                    "desc": row["desc_casen"],           
                    "method": "extent_opti",
                    "main_graph" : row["main_graph_casen"],
                    "second_graph" : row["second_graph_casen"],
                    "third_graph" : row["third_graph_casen"],
                    "file_id": row["file_id"]
            })
            
        if new_rows:
            self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)
            self.df = self.df.drop_duplicates(subset=["titles", "NER", "NER_label", "desc", "method", "main_graph", "second_graph", "third_graph", "file_id"])
        if self.verbose:
            print(f"[extent_optimisations] {len(new_rows)} new rows added.")

        self.df = self.df.sort_values(by=["file_id"]).reset_index(drop=True)

        return self.df

    @chrono
    def save_dataframe(self, filename: str, ) -> str:
        """Save a DataFrame in an Excel file, avoiding overwrite"""
        
        if self.ner_result_folder:
            path = self.ner_result_folder
        else:
            path = Path.cwd()

        if not path.exists():
            raise FileNotFoundError(f"[save] The provided folder does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"[save] The provided path is not a folder: {path}")
        
        base_filename = path / f"{filename}.xlsx"
        file_to_save = base_filename
        counter = 1

        while file_to_save.exists():
            file_to_save = path / f"{filename}({counter}).xlsx"
            counter += 1

        self.df.to_excel(file_to_save, index=False, engine="openpyxl")

        return f"File saved in : {str(file_to_save)}"

    @chrono
    def run(self) -> str:
        """ Run SpaCy & CasEN et merge both result with NER optimisations"""
        self.data_df = pd.read_excel(self.data) # Load data
        # spaCy config
        if not isinstance(self.spaCy, pd.DataFrame):
            self.spaCy.data_df = self.data_df
            self.spaCy_df = self.spaCy.run()
        else:
            self.spaCy_df = self.spaCy

        # casEN config
        if not isinstance(self.casEN, pd.DataFrame):
            self.casEN.data_df = self.data_df
            self.casEN_df = self.casEN.run()
        else:
            self.casEN_df = self.casEN

        # --- MERGE --- #  
        self.merge()

        # -------- OPTIMISATIONS -------- #
        if self.casEN_graph_validation is not None:
            self.casEN_optimisation()

        if self.casEN_priority_merge:
            self.casEN_priority()

        if self.extent_optimisation:
            self.extent_optimisations()
        # ----------- CORRECTION ------- #
        if self.correction is not None:
            self.apply_correction()


        # # Save 
        filename = f"NER"
        saved = self.save_dataframe(filename)

        return saved



        