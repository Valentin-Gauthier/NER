from .spacy_wrapper import SpaCy
from .casen import CasEN
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import json

class NER:
    def __init__(self, spaCy:SpaCy | pd.DataFrame, casEN:CasEN | pd.DataFrame, data:str, casEN_priority_merge:bool, casEN_graph_validation:str, remove_duplicate_rows:bool, NER_result_folder:str,excluded_names:str , correction:str=None, logging:bool=True, log_folder:str=None, timer:bool=True, verbose:bool=False):
        self.spaCy = spaCy
        self.casEN = casEN
        self.data = Path(data)
        self.NER_result_folder = Path(NER_result_folder)

        # ------------------ NER OPTIMISATION -------------------- #
        self.casEN_priority_merge = casEN_priority_merge     
        self.casEN_graph_validation = Path(casEN_graph_validation) if casEN_graph_validation else None
        self.remove_duplicate_rows = remove_duplicate_rows
        self.excluded_names = excluded_names

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
        """ Merge spaCy & CasEN result"""
        if self.verbose:
            df_not_empty = self.data_df[self.data_df["desc"] != ""]
            print(f"[merge] Found {len(df_not_empty)} valid rows")

        self.spaCy_df["key"] = self.spaCy_df[["NER", "NER_label", "file_id"]].apply(lambda x: tuple(x), axis=1)
        self.casEN_df["key"] = self.casEN_df[["NER", "NER_label", "file_id"]].apply(lambda x: tuple(x), axis=1)

        merge_df = pd.merge(self.spaCy_df, self.casEN_df, on="key", how="outer", suffixes=["_spaCy", "_casEN"], indicator=True)

        # Fix shared columns
        merge_df["titles"] = merge_df["titles_spaCy"].combine_first(merge_df["titles_casEN"])
        merge_df["NER"] = merge_df["NER_spaCy"].combine_first(merge_df["NER_casEN"])
        merge_df["NER_label"] = merge_df["NER_label_spaCy"].combine_first(merge_df["NER_label_casEN"])
        merge_df["desc"] = merge_df["desc_spaCy"].combine_first(merge_df["desc_casEN"])
        merge_df["method"] = merge_df.apply(self.get_method, axis=1) # Update for intersection
        merge_df["file_id"] = merge_df["file_id_spaCy"].combine_first(merge_df["file_id_casEN"])

        if self.remove_duplicate_rows:
            merge_df.drop_duplicates(subset=["NER", "NER_label", "method", "main_graph", "second_graph", "third_graph", "file_id"])
            if self.verbose:
                print(f"[merge] Dropping duplicate rows")

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

        if self.verbose:
            print("[opt] count par méthode après :")
            print(self.df["method"].value_counts(dropna=False))


        self.df = self.df.reset_index(drop=True)
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
            self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)
            self.df = self.df.drop_duplicates(subset=["titles", "NER", "NER_label", "desc", "method","main_graph", "second_graph", "third_graph", "file_id"])
            if self.verbose:
                print(f"[casEN_priority] {len(new_rows)} added.")

        self.df = self.df.sort_values(by=["file_id"], ascending=True).reset_index(drop=True)

        return self.df

    def save_dataframe(self, filename: str, ) -> str:
        """Save a DataFrame in an Excel file, avoiding overwrite"""
        
        if self.NER_result_folder:
            path = self.NER_result_folder
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

        # ----------- CORRECTION ------- #
        if self.correction is not None:
            self.apply_correction()

        # Save 
        filename = f"NER"
        saved = self.save_dataframe(filename)

        return saved



        