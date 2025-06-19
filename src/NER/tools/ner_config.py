from .spacy_wrapper import SpaCyConfig
from .casen import CasEN
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import yaml


class NerConfig:

    def __init__(self,
                 process_priority_merge:bool=True,
                 process_casen_opti:bool=True,
                 remove_duplicated_entity_per_desc:bool=True,
                 keep_only_trustable_methods:bool=True,
                 make_excel_file:bool=False,
                 production_mode:bool=True,
                 ner_config:str=Path(__file__).parent / "config.yaml",
                 logging:bool=False,
                 timer:bool=False,
                 verbose:bool=False
                 ):
        self.process_priority_merge = process_priority_merge
        self.process_casen_opti = process_casen_opti
        self.remove_duplicated_entity_per_desc = remove_duplicated_entity_per_desc
        self.keep_only_trustable_methods = keep_only_trustable_methods
        self.make_excel_file = make_excel_file
        self.production_mode = production_mode
        self.ner_config = Path(ner_config)
        self.logging = logging
        self.timer = timer
        self.verbose = verbose

        # Load the config
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
            f.write(f"{timestamp} - NER [{step}] finish in {duration:.2f} s.\n")

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

        if not self.ner_config.is_file():
            raise FileNotFoundError(f"[load config] The provided file was not found ! {self.ner_config}")
        else:
            with open(self.ner_config, 'r', encoding="utf-8") as f:
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

    def merge(self) -> pd.DataFrame:
        """Merge Every DataFrame"""

        if self.verbose:
            print(f"[merge] Shape of DataFrames : {[df.shape for df in self.dfs]}")

        #prepare dfs
        prepared_dfs = []
        for i, df in enumerate(self.dfs):
            df['method'] = df.get('method', f"df{i}")
            df['key'] = df[["NER", "NER_label","file_id", "entity_start", "entity_end"]].apply(tuple, axis=1)
            prepared_dfs.append(df)

        # merge
        merged_df = prepared_dfs[0]

        for i, df in enumerate(prepared_dfs[1:], start=1):
            
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

            # Change the method
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

            # merge duplicated columns
            columns_to_merge = ['NER', 'NER_label', 'file_id', 'desc', 'titles',
                                'main_graph', 'second_graph', 'third_graph', "entity_start", "entity_end"]
            for col in columns_to_merge:
                col_left = f"{col}_left"
                col_right = f"{col}_right"
                if col_left in merged_df.columns and col_right in merged_df.columns:
                    merged_df[col] = merged_df[col_left].combine_first(merged_df[col_right])
                    merged_df.drop([col_left, col_right], axis=1, inplace=True)

            # cleaning
            merged_df.drop(columns=['_merge', 'method_left', 'method_right'], inplace=True, errors='ignore')

        merged_df.drop(columns=['key'], inplace=True, errors='ignore')

        # sort dataframe
        merged_df = merged_df.sort_values(by=["file_id", "entity_start"]).reset_index(drop=True)

        final_columns = ["titles", "NER", "NER_label", "desc", "method",
                        "main_graph", "second_graph", "third_graph", "file_id", "entity_start", "entity_end"]
        
        merged_df = merged_df[[col for col in final_columns if col in merged_df.columns]]

        if self.verbose:
            print(f"[merge] Final merged DataFrame shape: {merged_df.shape}")
            print("[merge] method value counts:")
            print(merged_df['method'].value_counts())

        self.df = merged_df

        return self.df

    def casEN_optimisation(self) -> pd.DataFrame:
        """ Change the method casEN to casEN_opti when the graphs in the JSON trustable graphs"""

        valid_graphs = self.config["casEN_opti"]
        
        def is_allowed(row):
            for combo in valid_graphs:
                if all((str(row.get(col)) if pd.notna(row.get(col)) else "") == val for col, val in combo.items()):
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
            casen_opti_count = source_counts.get('casEN_opti', 0)
            
            print(f"[casEN_optimisation] CasEN_opti only : {casen_opti_count} lignes")

        return self.df

    def priority_merge(self) -> pd.DataFrame:
        """Update composite method rows (e.g., casEN_Stanza) to _priority if they conflict with atomic methods and label is PER."""

        name_list = self.config["excluded_names_list"]

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
                    on=["NER","file_id", "entity_start", "entity_end"],
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

        for idx, new_method in rows_to_update.items():
            self.df.at[idx, "method"] = new_method

        if self.verbose:
            print(f"[composite_entity_priority] Updated {len(rows_to_update)} rows to _priority.")

            source_counts = self.df["method"].value_counts()
            for method, count in source_counts.items():
                print(f"[composite_entity_priority] {method} : {count} lignes")

        self.df = self.df.sort_values(by=["file_id", "entity_start"]).reset_index(drop=True)
        return self.df

    def apply_correction(self, correction: str) -> pd.DataFrame:
        """Applique les corrections depuis un fichier Excel sur les colonnes cibles, selon correspondance exacte."""

        # Colonnes à corriger si match trouvé
        correction_columns = ["manual cat", "correct", "extent", "NER_category"]

        # Lecture du fichier Excel de correction
        correction_df = pd.read_excel(correction)

        # Supprime les doublons dans correction_df (au cas où)
        correction_df = correction_df.drop_duplicates(subset=["NER", "NER_label", "file_id"])

        # Garde uniquement les colonnes nécessaires pour la jointure
        merge_columns = ["NER", "NER_label", "file_id"] + [col for col in correction_columns if col in correction_df.columns]
        correction_df = correction_df[merge_columns]

        # Merge avec self.df sur les 3 colonnes clés
        self.df = self.df.merge(correction_df, on=["NER", "NER_label", "file_id"], how="left", suffixes=("", "_corr"))

        # Copie les valeurs de correction uniquement si présentes
        for col in correction_columns:
            corr_col = f"{col}_corr"
            if corr_col in self.df.columns:
                self.df[col] = self.df[corr_col].combine_first(self.df[col])
                self.df = self.df.drop(columns=[corr_col])

        #change columns order
        columns_order = correction_columns + [col for col in self.df.columns if col not in correction_columns]
        self.df = self.df[columns_order]

        if self.verbose:
            print(f"[apply_correction] Corrections appliquées aux colonnes : {correction_columns}")
            print(f"[apply_correction] self.df shape = {self.df.shape}")

        return self.df

    def save(self):
        """Save the DataFrame into the corresponding format"""
        extention = self.config["extention"]
        folder = Path(self.config["ner_result_folder"])

        # Create the file name with corresponding options
        filename = str(self.data["days"].iloc[0])
        if self.process_priority_merge:
            filename += "_priority"
        if self.process_casen_opti:
            filename += "_CasenOpti"
        if not self.remove_duplicated_entity_per_desc:
            filename += "_Duplicate"
        if self.keep_only_trustable_methods:
            filename += "_TrustMethods"
        if self.production_mode:
            filename += "_prod"

        final_filename = folder / f"{filename}.{extention}"

        save = final_filename
        counter = 1

        while save.exists():
            save = folder / f"{filename}({counter}).{extention}"
            counter+=1
        
        if extention == "xlsx":
            self.df.to_excel(save, index=False, engine="openpyxl")
        if extention == "csv":
            self.df.to_csv(save, index=False)   

        return str(save) 
    
    def clean(self) -> pd.DataFrame:
        """Clean the DataFrames to removes unecessary rows and columns"""


        # Keep only trustable methods
        if self.keep_only_trustable_methods:
            before = self.df.shape[0]
            final_methods = self.config["final_methods_to_keep"]
            self.df = self.df[self.df["method"].isin(final_methods)]
            if self.verbose:
                print(f"Trustable methods : {final_methods}")
                print(f"[cleaning] {before - self.df.shape[0]} rows were removed")

        # Remove all duplicated rows on specific columns
        if self.remove_duplicated_entity_per_desc:
            before_remove_duplicate = self.df.shape[0]
            self.df = self.df.drop_duplicates(subset=["NER", "NER_label","file_id"])
            if self.verbose:
                print(f"[cleaning] {before_remove_duplicate - self.df.shape[0]} Duplicated rows were removed")


        # Choose the last columns to keep
        if self.production_mode:
            final_columns = self.config["columns"]
            missing_cols = [col for col in final_columns if col not in self.df.columns and col in self.data.columns]
            if self.verbose:
                print(f"Missing columns : {missing_cols}")
            for col in missing_cols:
                self.df[col] = self.df["file_id"].apply(lambda idx: self.data.at[idx, col] if idx in self.data.index else None)

            self.df = self.df[[col for col in final_columns if col in self.df.columns]]

            if self.verbose:
                print(f"[columns] Final columns in df: {self.df.columns.tolist()}")
        else:
            # production mode false , so grab the description for analyses and correction
            window = self.config["description_window"]
            def extract_context(row):
                file_id = row["file_id"]
                start = int(row["entity_start"])
                end = int(row["entity_end"])
                
                if file_id in self.data.index:
                    desc = self.data.at[file_id, "desc"]
                    if pd.isna(desc) or not isinstance(desc, str):
                        return ""
                    # Calcul des bornes avec protection contre les débordements
                    start_idx = max(0, start - window)
                    end_idx = min(len(desc), end + window)
                    return desc[start_idx:end_idx]
                else:
                    return ""

            # Appliquer l'extraction à chaque ligne
            self.df["desc"] = self.df.apply(extract_context, axis=1)
            order = ["NER", "NER_label", "desc", "method","main_graph", "second_graph", "third_graph", "file_id", "entity_start", "entity_end"]
            self.df = self.df[order]

        return self.df



    def run(self, data:pd.DataFrame, dfs:list[pd.DataFrame]=None, correction:str=None) -> pd.DataFrame:
        """"""
        # Check if the list contains only DataFrame
        if not all(isinstance(df, pd.DataFrame)for df in dfs):
            raise ValueError("[NER init] dfs must be a list of pandas DataFrames.")
        self.dfs = [df.copy() for df in dfs]


        # Load the data
        self.load_data(data)

        # ------------- CONSENSUS -------------- #
        # merge every dataframes
        self.merge()
        # optimisations
        if self.process_casen_opti:
            self.casEN_optimisation()

        if self.process_priority_merge:
            self.priority_merge()

        # Cleaning DataFrames
        self.clean()

        # correction
        if correction is not None:
            self.apply_correction(correction)

        # --- SAVE --- #
        if self.make_excel_file:
            saved = self.save()
            print(f"File saved at : {saved}")
        return self.df


