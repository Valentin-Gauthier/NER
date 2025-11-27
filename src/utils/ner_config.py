from pathlib import Path
import pandas as pd
import yaml


class NerConfig:

    def __init__(self, 
                 process_priority_merge: bool = True,
                 process_casen_opti: bool = True,
                 remove_duplicated_entity_per_desc: bool = True,
                 ner_config: str = Path(__file__).parent.parent / "config.yaml",
                 verbose: bool = False
                 ):
        self.process_priority_merge = process_priority_merge
        self.process_casen_opti = process_casen_opti
        self.remove_duplicated_entity_per_desc = remove_duplicated_entity_per_desc
        self.ner_config = Path(ner_config)
        self.verbose = verbose

        self.load_config()


    def load_config(self):
        """Load the YAML config about the NER"""

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
    def order(dataframe:pd.DataFrame) -> pd.DataFrame:
        """
        Trier le dataframe dans l'ordre croissant des files_id
        """
        df = dataframe.copy()
        df["first_file_id"] = df["files_id"].apply(lambda x: x[0])
        df = df.sort_values(by="first_file_id", ascending=True).reset_index(drop=True)
        df = df.drop(columns=["first_file_id"])

        return df

    @staticmethod
    def _merge(dfs:list[pd.DataFrame], 
               keys:list[str]=["NE", "label", "files_id", "pos"], 
               column:str="method",
               verbose:bool=False
               ) -> pd.DataFrame:
        
        """Fusionner plusieurs dataframes
        """

        if verbose:
            print(f"[_merge] Shape of every DataFrame : {[df.shape for df in dfs]}")
        
        # Verifier que les dataframes ont tous la colonne que l'on fusionnera "method"
        prepared_dfs = []
        for i, df in enumerate(dfs):
            df = df.copy()
            if column not in df.columns:
                df[column] = f'df_{i}'
            prepared_dfs.append(df)

        
        def resolve_method(row):
            left = row.get(f'{column}_left')
            right = row.get(f'{column}_right')
            if row["_merge"] == "both":
                return f'{left}_{right}'
            elif row["_merge"] == "left_only":
                return left
            else:
                return right

        # merge
        merged_df = prepared_dfs[0]
        for idx, df in enumerate(prepared_dfs[1:], start=1):
            merged_df = merged_df.rename(columns={column : f"{column}_left"}) # method_left
            right_df = df.rename(columns={column : f"{column}_right"}) # method_right

            merged_df = pd.merge(
                left=merged_df,
                right=right_df,
                on=keys,
                how="outer",
                suffixes=("_left", "_right"),
                indicator=True
            )

            merged_df[column] = merged_df.apply(resolve_method, axis=1)

            # Fusionner les autres colonnes proprement
            for col in [c for c in merged_df.columns if c.endswith("_left")]:
                basename = col[:-5]
                col_right = f"{basename}_right"

                if basename != column and col_right in merged_df.columns:
                    merged_df[basename] = merged_df[col].combine_first(merged_df[col_right])
                    merged_df = merged_df.drop(columns=[col, col_right]) # on supprime _left et _right
            
            merged_df = merged_df.drop(columns=["_merge", f"{column}_left", f"{column}_right"], errors="ignore")

        return merged_df


    @staticmethod
    def priority_merge(df:pd.DataFrame, labels:list[str]=["PER", "LOC", "ORG", "MISC"]) -> pd.DataFrame:
        """
            Applique une priorité (ajoute '_priority' à la méthode) SI ET SEULEMENT SI :
            1. Le label est dans la liste 'labels'
            2. La méthode est issue de plus d'outils que les méthodes concurrentes (Majorité stricte).
            
            Exemple :
            - SpaCy (1) vs CasEN_Stanza (2) -> CasEN_Stanza gagne (2 > 1)
            - SpaCy (1) vs Stanza (1)       -> Personne ne gagne (1 == 1)
        """
        df = df.copy()
        # Calcule du "poids" de chaque méthode
        df["__weight"] = df["method"].str.count("_") + 1
        # On compare les meme NE mais avec differents labels
        cols = ["files_id", "pos", "NE"]
        
        df['__conflict_size'] = df.groupby(cols)['files_id'].transform('count')
        df['__max_weight'] = df.groupby(cols)['__weight'].transform('max')

        def count_winner(x):
            return (x == x.max()).sum()
        
        df["__nb_winners"] = df.groupby(cols)["__weight"].transform(count_winner)

        mask = (
            (df["label"].isin(labels)) &
            (df["__conflict_size"] > 1) &
            (df["__weight"] == df["__max_weight"]) &
            (df["__nb_winners"] == 1)
        )

        df.loc[mask, "method"] = df.loc[mask, "method"] + "_priority"
        df = df.drop(columns=["__weight", "__max_weight", "__nb_winners", "__conflict_size"])


        return df


    @staticmethod
    def keep_precise_graphs(df: pd.DataFrame, graphs: list[dict], verbose: bool = False) -> pd.DataFrame:
        """
        Change the method name of entity when only casEN found them with precise graphs
        Optimized version using vectorization.
        """
        df = df.copy()
        # 1. On crée un masque initial : tout est Faux
        valid_graph_mask = pd.Series(False, index=df.index)

        # 2. On itère sur les configurations de graphes
        for combo in graphs:
            current_combo_mask = pd.Series(True, index=df.index)
            
            for col, val in combo.items():
                if col not in df.columns:
                    current_combo_mask = False 
                    break
            
                current_combo_mask &= (df[col] == val)
            
            valid_graph_mask |= current_combo_mask


        
        # 3. Application de la modification sur casEN et casEN_stanza etc
        has_casen = df["method"].str.contains("casEN", na=False, regex=False)
        condition = has_casen & valid_graph_mask
        df.loc[condition, "method"] = df.loc[condition, "method"].str.replace("casEN", "casENOpti", regex=False)

        # # 3. Application de la modification (uniquement sur CasEN)
        # condition = (df["method"] == "casEN") & valid_graph_mask
        # df.loc[condition, "method"] = "casENOpti"

        if verbose: 
            nb_opti = condition.sum()
            print(f'[keep precise graphs] CasENOpti : {nb_opti} lines updated')

        return df

    def run(self, data:pd.DataFrame, dfs:list[pd.DataFrame]):

        if dfs is None or not all(isinstance(df, pd.DataFrame) for df in dfs):
            raise ValueError("[run] 'dfs' must be a list of pandas DataFrames.")
        
        self.load_data(data)
        
        # 1- Merge DataFrames
        df = self._merge(dfs, verbose=self.verbose)

        # 2- Priority
        if self.process_priority_merge:
            df = self.priority_merge(df) # labels=["PER"]

        # 3- CasEN Graphs
        if self.process_casen_opti:
            graphs = self.config["casEN_opti2"]
            df = self.keep_precise_graphs(df, graphs, self.verbose)

        # Order the DataFrames
        df = self.order(df)

        return df
        
