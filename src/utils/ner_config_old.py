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
                 save_to_file:bool=False,
                 production_mode:bool=True,
                 ner_config:str=Path(__file__).parent.parent / "config.yaml",
                 logging:bool=False,
                 timer:bool=False,
                 verbose:bool=False
                 ):
        self.process_priority_merge = process_priority_merge
        self.process_casen_opti = process_casen_opti
        self.remove_duplicated_entity_per_desc = remove_duplicated_entity_per_desc
        self.keep_only_trustable_methods = keep_only_trustable_methods
        self.save_to_file = save_to_file
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

    @staticmethod
    def merge_dataframes(dfs:list[pd.DataFrame], keys:list[str]=["NE", "NE_label", "files_id", "entity_start", "entity_end"], verbose:bool=False) -> pd.DataFrame:
        """
        Merge a list of dataframe on a given key

        
        """

        if verbose:
            print(f"[merge dataframes] Sape of every dataframe in order : {[df.shape for df in dfs]}")

        # Step 1 : prepare DataFrames before merge (generate a unique key)
        prepared_dfs = []
        for i, df in enumerate(dfs):
            df = df.copy()
            if "method" not in df.columns:
                df["method"] = f'df_{i}'
            prepared_dfs.append(df)

        # Step 2 : merge DataFrames
        merged_df = prepared_dfs[0]
        for idx, df in enumerate(prepared_dfs[1:], start=1):
            merged_df = merged_df.rename(columns={'method': "method_left"})
            right_df = df.rename(columns={"method": "method_right"})

            merged_df = pd.merge(
                left=merged_df,
                right=right_df,
                on=keys,
                how="outer",
                suffixes=("_left", "_right"),
                indicator=True
            )

            # Change the 'method' column
            def resolve_method(row):
                left = row.get('method_left')
                right = row.get('method_right')
                if row["_merge"] == "both":
                    return f'{left}_{right}'
                elif row["_merge"] == "left_only":
                    return left
                else:
                    return right
            
            merged_df["method"] = merged_df.apply(resolve_method, axis=1)
            merged_df = merged_df.drop(columns=["_merge", "method_left", "method_right"], errors="ignore")

        # Step 3 : Colonne a conservé
        keeped_cols = [c for c in merged_df.columns if not( c.endswith("_left") or c.endswith("_right"))]

        # Parmis les colonnes conflictuelle on prend la colonne non nul
        for col in merged_df.columns:
            if col.endswith("_left"):
                base = col[:-5]
                if base not in keeped_cols:
                    keeped_cols.append(base)
        
        print(f"KEEPING COLUMNS : {keeped_cols}")




        keeped_cols = [c for c in merged_df.columns if not( c.endswith("_left") or c.endswith("_right"))]

        final_df = merged_df[keeped_cols]

        for col in merged_df.columns:
            if col not in keeped_cols and col.endswith("_left"):
                base = col[:-5]
                right_col = f"{base}_right"
                # combine_first garde la valeur non-NaN
                final_df[col] = merged_df[col].combine_first(merged_df[right_col])
                merged_df.drop([col, right_col], axis=1, inplace=True)

        if verbose:
            print(f"[merge] Final merged DataFrame shape: {merged_df.shape}")
            print("[merge] method value counts:")
            print(merged_df['method'].value_counts())

        return merged_df

    def apply_priority(self, df: pd.DataFrame, name_list: list, verbose: bool = False) -> pd.DataFrame:
        """Update composite method rows (e.g., casEN_Stanza) to _priority if they conflict with
        atomic methods and label is PER."""

        # Construire la clé normalisée une fois pour toutes
        df = df.copy()
        df["files_id_key"] = df["files_id"].apply(
            lambda x: tuple(sorted(x)) if isinstance(x, list) else (x,)
        )

        all_methods = df["method"].unique()
        composite_methods = [m for m in all_methods if "_" in m and m]
        atomic_methods = [m for m in all_methods if "_" not in m]

        if verbose:
            print(f"[composite_entity_priority] Composite methods: {composite_methods}")
            print(f"[composite_entity_priority] Atomic methods: {atomic_methods}")

        rows_to_update = {}

        for composite_method in composite_methods:
            composite_df = df[df["method"] == composite_method]

            for atomic_method in atomic_methods:
                atomic_df = df[df["method"] == atomic_method]

                merged = pd.merge(
                    composite_df, atomic_df,
                    on=["NE", "files_id_key", "start", "end"],
                    suffixes=("_composite", "_atomic")
                )

                conflicts = merged[merged["label_composite"] != merged["label_atomic"]]

                for _, row in conflicts.iterrows():
                    if (
                        row["label_composite"] == "PER"
                        and row["NE"].lower() not in [name.lower() for name in name_list]
                    ):
                        matching_rows = df[
                            (df["method"] == composite_method)
                            & (df["NE"] == row["NE"])
                            & (df["files_id_key"] == row["files_id_key"])
                        ]

                        for idx in matching_rows.index:
                            current_method = df.at[idx, "method"]
                            if not current_method.endswith("_priority"):
                                rows_to_update[idx] = f"{current_method}_priority"

        # Mise à jour en masse
        for idx, new_method in rows_to_update.items():
            df.at[idx, "method"] = new_method

        if verbose:
            print(f"[composite_entity_priority] Updated {len(rows_to_update)} rows to _priority.")

            source_counts = df["method"].value_counts()
            for method, count in source_counts.items():
                print(f"[composite_entity_priority] {method} : {count} lignes")

        # Trier proprement
        df = df.sort_values(by=["files_id_key", "start"]).reset_index(drop=True)
        df = df.drop(columns=["files_id_key"])
        return df

    def keep_precise_graphs(self, df:pd.DataFrame, graphs:list[dict], verbose:bool=False) -> pd.DataFrame:
        """
        Change the method name of entity when only casEN found them with precise graphs
        :param df: 
        :param graph: list of every trustable graphs

        :return pd.DataFrame
        """

        def graph_allowed(row):
            for combo in graphs:
                if all((str(row.get(col)) if pd.notna(row.get(col)) else "") == val for col, val in combo.items()):
                    return True
            return False

        
        def change_method(row):
            if row["method"] == "casEN" and graph_allowed(row):
                return "casENOpti"
            return row["method"]
        

        df["method"] = df.apply(change_method, axis=1)
        df = df.reset_index(drop=True)

        if self.verbose:
            methods = df["method"].value_counts()
            opti = methods.get("casENOpti", 0)
            print(f'[keep precise graphs] CasENOpti : {opti} lines')

        return df

    def apply_correction(self, correction: str) -> pd.DataFrame:
        """Applique les corrections depuis un fichier Excel sur les colonnes cibles, selon correspondance exacte."""

        # Colonnes à corriger si match trouvé
        correction_columns = ["manual cat", "correct", "extent", "NER_category"]

        # Lecture du fichier Excel de correction
        correction_df = pd.read_excel(correction)

        # Supprime les doublons dans correction_df (au cas où)
        correction_df = correction_df.drop_duplicates(subset=["NER", "NER_label", "files_id"])

        # Garde uniquement les colonnes nécessaires pour la jointure
        merge_columns = ["NER", "NER_label", "files_id"] + [col for col in correction_columns if col in correction_df.columns]
        correction_df = correction_df[merge_columns]

        # Merge avec self.df sur les 3 colonnes clés
        self.df = self.df.merge(correction_df, on=["NER", "NER_label", "files_id"], how="left", suffixes=("", "_corr"))

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
            self.df = self.df.drop_duplicates(subset=["NER", "NER_label","files_id"])
            if self.verbose:
                print(f"[cleaning] {before_remove_duplicate - self.df.shape[0]} Duplicated rows were removed")


        # Choose the last columns to keep
        if self.production_mode:
            final_columns = self.config["columns"]
            missing_cols = [col for col in final_columns if col not in self.df.columns and col in self.data.columns]
            if self.verbose:
                print(f"Missing columns : {missing_cols}")
            for col in missing_cols:
                self.df[col] = self.df["files_id"].apply(lambda idx: self.data.at[idx, col] if idx in self.data.index else None)

            self.df = self.df[[col for col in final_columns if col in self.df.columns]]

            if self.verbose:
                print(f"[columns] Final columns in df: {self.df.columns.tolist()}")
        else:
            # production mode false , so grab the description for analyses and correction
            window = self.config["description_window"]
            def extract_context(row):
                files_id = row["files_id"]
                start = int(row["start"])
                end = int(row["end"])
                
                if files_id in self.data.index:
                    desc = self.data.at[files_id, "desc"]
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
            order = ["NE", "label", "desc", "method","files_id", "start", "end"]
            other_cols = [c for c in self.df.columns if c not in order]
            self.df = self.df[order + other_cols]

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
        self.df = self.merge_dataframes(dfs=dfs, keys=["NE", "label", "files_id", "start", "end"], verbose=self.verbose)

        if self.process_priority_merge:
            name_list = self.config["excluded_names_list"]
            self.df = self.apply_priority(df=self.df, name_list=name_list, verbose=self.verbose)

        # optimisations
        if self.process_casen_opti:
            graphs = self.config["casENOpti_grf"]
            self.df = self.keep_precise_graphs(df=self.df, graphs=graphs, verbose=self.verbose)

        # Cleaning DataFrames
        # self.clean()

        # correction
        if correction is not None:
            self.apply_correction(correction)

        # --- SAVE --- #
        if self.save_to_file:
            saved = self.save()
            print(f"File saved at : {saved}")
        return self.df




    def apply_priority(self, df: pd.DataFrame, name_list: list, verbose: bool = False) -> pd.DataFrame:
        """Update composite method rows (e.g., casEN_Stanza) to _priority if they conflict with
        atomic methods and label is PER."""

        # Construire la clé normalisée une fois pour toutes
        df = df.copy()
        df["files_id_key"] = df["files_id"].apply(
            lambda x: tuple(sorted(x)) if isinstance(x, list) else (x,)
        )

        all_methods = df["method"].unique()
        composite_methods = [m for m in all_methods if "_" in m and m]
        atomic_methods = [m for m in all_methods if "_" not in m]

        if verbose:
            print(f"[composite_entity_priority] Composite methods: {composite_methods}")
            print(f"[composite_entity_priority] Atomic methods: {atomic_methods}")

        rows_to_update = {}

        for composite_method in composite_methods:
            composite_df = df[df["method"] == composite_method]

            for atomic_method in atomic_methods:
                atomic_df = df[df["method"] == atomic_method]

                merged = pd.merge(
                    composite_df, atomic_df,
                    on=["NE", "files_id_key", "start", "end"],
                    suffixes=("_composite", "_atomic")
                )

                conflicts = merged[merged["label_composite"] != merged["label_atomic"]]

                for _, row in conflicts.iterrows():
                    if (
                        row["label_composite"] == "PER"
                        and row["NE"].lower() not in [name.lower() for name in name_list]
                    ):
                        matching_rows = df[
                            (df["method"] == composite_method)
                            & (df["NE"] == row["NE"])
                            & (df["files_id_key"] == row["files_id_key"])
                        ]

                        for idx in matching_rows.index:
                            current_method = df.at[idx, "method"]
                            if not current_method.endswith("_priority"):
                                rows_to_update[idx] = f"{current_method}_priority"

        # Mise à jour en masse
        for idx, new_method in rows_to_update.items():
            df.at[idx, "method"] = new_method

        if verbose:
            print(f"[composite_entity_priority] Updated {len(rows_to_update)} rows to _priority.")

            source_counts = df["method"].value_counts()
            for method, count in source_counts.items():
                print(f"[composite_entity_priority] {method} : {count} lignes")

        # Trier proprement
        df = df.sort_values(by=["files_id_key", "start"]).reset_index(drop=True)
        df = df.drop(columns=["files_id_key"])
        return df

    def keep_precise_graphs(self, df:pd.DataFrame, graphs:list[dict], verbose:bool=False) -> pd.DataFrame:
        """
        Change the method name of entity when only casEN found them with precise graphs
        :param df: 
        :param graph: list of every trustable graphs

        :return pd.DataFrame
        """

        def graph_allowed(row):
            for combo in graphs:
                if all((str(row.get(col)) if pd.notna(row.get(col)) else "") == val for col, val in combo.items()):
                    return True
            return False

        
        def change_method(row):
            if row["method"] == "casEN" and graph_allowed(row):
                return "casENOpti"
            return row["method"]
        

        df["method"] = df.apply(change_method, axis=1)
        df = df.reset_index(drop=True)

        if self.verbose:
            methods = df["method"].value_counts()
            opti = methods.get("casENOpti", 0)
            print(f'[keep precise graphs] CasENOpti : {opti} lines')

        return df

    def apply_correction(self, correction: str) -> pd.DataFrame:
        """Applique les corrections depuis un fichier Excel sur les colonnes cibles, selon correspondance exacte."""

        # Colonnes à corriger si match trouvé
        correction_columns = ["manual cat", "correct", "extent", "NER_category"]

        # Lecture du fichier Excel de correction
        correction_df = pd.read_excel(correction)

        # Supprime les doublons dans correction_df (au cas où)
        correction_df = correction_df.drop_duplicates(subset=["NER", "NER_label", "files_id"])

        # Garde uniquement les colonnes nécessaires pour la jointure
        merge_columns = ["NER", "NER_label", "files_id"] + [col for col in correction_columns if col in correction_df.columns]
        correction_df = correction_df[merge_columns]

        # Merge avec self.df sur les 3 colonnes clés
        self.df = self.df.merge(correction_df, on=["NER", "NER_label", "files_id"], how="left", suffixes=("", "_corr"))

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
            self.df = self.df.drop_duplicates(subset=["NER", "NER_label","files_id"])
            if self.verbose:
                print(f"[cleaning] {before_remove_duplicate - self.df.shape[0]} Duplicated rows were removed")


        # Choose the last columns to keep
        if self.production_mode:
            final_columns = self.config["columns"]
            missing_cols = [col for col in final_columns if col not in self.df.columns and col in self.data.columns]
            if self.verbose:
                print(f"Missing columns : {missing_cols}")
            for col in missing_cols:
                self.df[col] = self.df["files_id"].apply(lambda idx: self.data.at[idx, col] if idx in self.data.index else None)

            self.df = self.df[[col for col in final_columns if col in self.df.columns]]

            if self.verbose:
                print(f"[columns] Final columns in df: {self.df.columns.tolist()}")
        else:
            # production mode false , so grab the description for analyses and correction
            window = self.config["description_window"]
            def extract_context(row):
                files_id = row["files_id"]
                start = int(row["start"])
                end = int(row["end"])
                
                if files_id in self.data.index:
                    desc = self.data.at[files_id, "desc"]
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
            order = ["NE", "label", "desc", "method","files_id", "start", "end"]
            other_cols = [c for c in self.df.columns if c not in order]
            self.df = self.df[order + other_cols]

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
        self.df = self.merge_dataframes(dfs=dfs, keys=["NE", "label", "files_id", "start", "end"], verbose=self.verbose)

        if self.process_priority_merge:
            name_list = self.config["excluded_names_list"]
            self.df = self.apply_priority(df=self.df, name_list=name_list, verbose=self.verbose)

        # optimisations
        if self.process_casen_opti:
            graphs = self.config["casENOpti_grf"]
            self.df = self.keep_precise_graphs(df=self.df, graphs=graphs, verbose=self.verbose)

        # Cleaning DataFrames
        # self.clean()

        # correction
        if correction is not None:
            self.apply_correction(correction)

        # --- SAVE --- #
        if self.save_to_file:
            saved = self.save()
            print(f"File saved at : {saved}")
        return self.df


