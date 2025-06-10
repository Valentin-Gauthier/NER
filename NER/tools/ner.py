from .spacy_wrapper import SpaCy
from .casen import CasEN
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import json

class Ner:
    def __init__(self, 
                 data:str | pd.DataFrame, 
                 casEN_priority_merge:bool, 
                 casEN_graph_validation:str, 
                 extent_optimisation:bool, 
                 remove_duplicate_rows:bool, 
                 ner_result_folder:str,
                 excluded_names:str , 
                 dfs:list[pd.DataFrame]=None, 
                 make_excel_file:bool=True,
                 correction:str=None, 
                 logging:bool=True, 
                 log_folder:str=None, 
                 timer:bool=True, 
                 verbose:bool=False):
        
        
        if not all(isinstance(df, pd.DataFrame) for df in dfs):
            raise ValueError("[NER init] dfs must be a list of pandas DataFrames.")
        self.dfs = [df.copy() for df in dfs]

        self.data = data
        self.ner_result_folder = Path(ner_result_folder)

        if dfs is not None and not all(isinstance(df, pd.DataFrame) for df in dfs):
            raise ValueError("[NER init] A list of pandas DataFrames is required")
        
        if all(isinstance(df, pd.DataFrame)for df in dfs):
            self.dfs = [df.copy() for df in dfs] # we can use 'self.dfs = dfs', if dataframes take to much RAM

        # ------------------ NER OPTIMISATION -------------------- #
        self.casEN_priority_merge = casEN_priority_merge     
        self.casEN_graph_validation = Path(casEN_graph_validation) if casEN_graph_validation else None
        self.remove_duplicate_rows = remove_duplicate_rows
        self.excluded_names = excluded_names
        self.extent_optimisation = extent_optimisation

        # ------------------- NER OPTIONS ----------------------- #
        self.make_excel_file = make_excel_file
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
        """Merge a list of DataFrames (without using pd.merge method because it make cartesian product)"""

        if self.verbose:
            for i, df in enumerate(self.dfs):
                print(f"[merge] shape of dataframe n°{i} : {df.shape}")

        # Étape 1 : préparation des DataFrames
        prepared_dfs = []
        for i, df in enumerate(self.dfs):
            df = df.copy()
            df['method'] = df.get('method', f"df{i}")
            df['occurrence_id'] = df.groupby(["NER", "NER_label", "file_id"]).cumcount()
            df['key'] = df[["NER", "NER_label", "file_id", "occurrence_id"]].apply(tuple, axis=1)
            prepared_dfs.append(df)

        # Étape 2 : fusion progressive
        merged_df = prepared_dfs[0]

        for i, df in enumerate(prepared_dfs[1:], start=1):
            df = df.copy()
            
            merged_df.rename(columns={'method': 'method_left'}, inplace=True)
            df.rename(columns={'method': 'method_right'}, inplace=True)

            merged_df = pd.merge(
                merged_df,
                df,
                on="key",
                how="outer",
                suffixes=("_left", "_right"),
                indicator=True
            )

            # Étape 3 : déterminer dynamiquement la méthode combinée
            def resolve_method(row):
                left = row.get('method_left')
                right = row.get('method_right')
                if row["_merge"] == "both":
                    return f"{left}_{right}"
                elif row["_merge"] == "left_only":
                    return left
                else:
                    return right

            merged_df['method'] = merged_df.apply(resolve_method, axis=1)

            # Étape 4 : fusionner les colonnes dupliquées
            columns_to_merge = ['NER', 'NER_label', 'file_id', 'desc', 'titles',
                                'main_graph', 'second_graph', 'third_graph']
            for col in columns_to_merge:
                col_left = f"{col}_left"
                col_right = f"{col}_right"
                if col_left in merged_df.columns and col_right in merged_df.columns:
                    merged_df[col] = merged_df[col_left].combine_first(merged_df[col_right])
                    merged_df.drop([col_left, col_right], axis=1, inplace=True)

            # Nettoyage
            merged_df.drop(columns=['_merge', 'method_left', 'method_right'], inplace=True, errors='ignore')

        # Finalisation
        merged_df.drop(columns=['key', 'occurrence_id'], inplace=True, errors='ignore')
        merged_df = merged_df.sort_values(by=["file_id", "NER", "method"]).reset_index(drop=True)

        # Réorganise les colonnes
        final_columns = ["titles", "NER", "NER_label", "desc", "method",
                        "main_graph", "second_graph", "third_graph", "file_id"]
        merged_df = merged_df[[col for col in final_columns if col in merged_df.columns]]

        if self.verbose:
            print(f"[merge] Final merged DataFrame shape: {merged_df.shape}")
            print("[merge] method value counts:")
            print(merged_df['method'].value_counts())

        self.df = merged_df

        return self.df

    @chrono
    def merge2(self) -> pd.DataFrame:
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
    def composite_entity_priority(self) -> pd.DataFrame:
        """Update composite method rows (e.g., casEN_Stanza) to _priority if they conflict with atomic methods and label is PER."""

        with open(self.excluded_names, 'r', encoding="utf-8") as f:
            name_list = json.load(f)[0].get("NER", [])

        all_methods = self.df["method"].unique()

        composite_methods = [m for m in all_methods if "_" in m and m != "casEN_opti"]
        atomic_methods = [m for m in all_methods if "_" not in m]

        if self.verbose:
            print(f"[composite_entity_priority] Composite methods: {composite_methods}")
            print(f"[composite_entity_priority] Atomic methods: {atomic_methods}")

        rows_to_update = {}

        for composite_method in composite_methods:
            composite_df = self.df[self.df["method"] == composite_method]

            for atomic_method in atomic_methods:
                atomic_df = self.df[self.df["method"] == atomic_method]

                merged = pd.merge(
                    composite_df, atomic_df,
                    on=["NER", "file_id"],
                    suffixes=("_composite", "_atomic")
                )

                conflicts = merged[merged["NER_label_composite"] != merged["NER_label_atomic"]]

                for _, row in conflicts.iterrows():
                    if (
                        row["NER_label_composite"] == "PER" and
                        row["NER"].lower() not in [name.lower() for name in name_list]
                    ):
                        matching_rows = self.df[
                            (self.df["method"] == composite_method) &
                            (self.df["NER"] == row["NER"]) &
                            (self.df["file_id"] == row["file_id"])
                        ]
                        for idx in matching_rows.index:
                            current_method = self.df.at[idx, "method"]
                            if not current_method.endswith("_priority"):
                                rows_to_update[idx] = f"{current_method}_priority"

        # Appliquer les changements
        for idx, new_method in rows_to_update.items():
            self.df.at[idx, "method"] = new_method

        if self.verbose:
            print(f"[composite_entity_priority] Updated {len(rows_to_update)} rows to _priority.")

            source_counts = self.df["method"].value_counts()
            for method, count in source_counts.items():
                print(f"[composite_entity_priority] {method} : {count} lignes")

        self.df = self.df.sort_values(by=["file_id"]).reset_index(drop=True)
        return self.df



    @chrono
    def apply_correction(self) -> pd.DataFrame:
        """Auto correct """

        columns = ["manual cat", "extent", "correct", "category"]

        for col in columns:
            self.df[col] = ""

        correction_df = pd.read_excel(self.correction)

        correction_df["key"] = correction_df[["NER", "NER_label", "hash"]].apply(lambda x: tuple(x), axis=1)
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
        """Prioritize CasEN labels (only PER) over conflicting labels from other methods by modifying 'method' field."""

        casEN_mask = self.df["method"] == "casEN"
        casEN_df = self.df[casEN_mask]
        other_methods = self.df["method"].unique()
        other_methods = [m for m in other_methods if m != "casEN" and "_" not in m]

        if self.verbose:
            print(f"[casEN_priority] Comparing casEN with: {', '.join(other_methods)}")

        with open(self.excluded_names, 'r', encoding="utf-8") as f:
            name_list = json.load(f)[0].get("NER", [])

        # Set of indices to modify in original df
        indices_to_update = set()

        for method in other_methods:
            other_df = self.df[self.df["method"] == method]
            merged = pd.merge(other_df, casEN_df, on=["NER", "file_id"], suffixes=(f"_{method}", "_casen"))
            conflicts = merged[merged[f"NER_label_{method}"] != merged["NER_label_casen"]]

            for _, row in conflicts.iterrows():
                if (
                    row["NER_label_casen"] == "PER"
                    and row["NER"].lower() not in [name.lower() for name in name_list]
                ):
                    # Locate the original row index in self.df
                    original_row = self.df[
                        (self.df["method"] == "casEN") &
                        (self.df["NER"] == row["NER"]) &
                        (self.df["file_id"] == row["file_id"])
                    ]
                    indices_to_update.update(original_row.index.tolist())

        # Apply the method update
        self.df.loc[list(indices_to_update), "method"] = "casEN_priority"

        if self.verbose:
            print(f"[casEN_priority] Updated {len(indices_to_update)} rows from 'casEN' to 'casEN_priority'")

            source_counts = self.df["method"].value_counts()
            for method in source_counts.index:
                print(f"[casEN_priority] {method} : {source_counts[method]} lignes")

        self.df = self.df.sort_values(by=["file_id"]).reset_index(drop=True)
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
        """Run SpaCy & CasEN or use provided DataFrames, then merge results with NER optimisations."""
        if isinstance(self.data, pd.DataFrame):
            self.data_df = self.data
        else:
            self.data_df = pd.read_excel(self.data)  # Load input data

        # --- MERGE --- #
        self.merge()

        # -------- OPTIMISATIONS -------- #
        if self.casEN_graph_validation is not None:
            self.casEN_optimisation()

        if self.casEN_priority_merge:
            #self.casEN_priority()
            self.composite_entity_priority()

        if self.extent_optimisation:
            self.extent_optimisations()

        # ----------- CORRECTION -------- #
        if self.correction is not None:
            self.apply_correction()

        # --- SAVE --- #
        if self.make_excel_file:
            filename = "NER"
            saved = self.save_dataframe(filename)
            print(f"File saved at : {saved}")

        return self.df




        