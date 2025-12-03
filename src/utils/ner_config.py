from pathlib import Path
import pandas as pd
import yaml


class NerConfig:

    def __init__(self, 
                 process_priority_merge: bool = True,
                 labels_priority:list[str] = ["PER", "LOC", "ORG", "MISC"],
                 process_casen_opti: bool = True,
                 remove_duplicated_entity_per_desc: bool = True,
                 ner_config: str = Path(__file__).parent.parent / "config.yaml",
                 verbose: bool = False
                 ):
        self.process_priority_merge = process_priority_merge
        self.labels_priority = labels_priority
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
        # has_casen = df["method"].str.contains("casEN", na=False, regex=False)
        # condition = has_casen & valid_graph_mask
        # df.loc[condition, "method"] = df.loc[condition, "method"].str.replace("casEN", "casENOpti", regex=False)

        # 3. Application de la modification (uniquement sur CasEN)
        condition = (df["method"] == "casEN") & valid_graph_mask
        df.loc[condition, "method"] = "casENOpti"

        if verbose: 
            nb_opti = condition.sum()
            print(f'[keep precise graphs] CasENOpti : {nb_opti} lines updated')

        return df


    @staticmethod
    def apply_correction(df: pd.DataFrame, correction: str, verbose: bool = False) -> pd.DataFrame:
        """
        Applique les corrections d'un Excel en gérant le fait que df['files_id'] est un tuple 
        et correction['files_id'] est un entier unique.
        """
        
        # 1. Chargement et Nettoyage de la correction
        correction_columns = ["manual cat", "correct", "extent", "NER_category"]
        
        try:
            corr_df = pd.read_excel(correction)
        except Exception as e:
            print(f"[apply_correction] Erreur lecture fichier: {e}")
            return df

        # On s'assure que les colonnes clés existent
        if not {"NE", "label", "files_id"}.issubset(corr_df.columns):
            print(f"[apply_correction] Colonnes manquantes dans le fichier Excel.")
            return df

        # Nettoyage des doublons dans la correction
        corr_df = corr_df.drop_duplicates(subset=["NE", "label", "files_id"])
        
        # On ne garde que les colonnes utiles (Clés + Colonnes à importer)
        cols_to_import = [c for c in correction_columns if c in corr_df.columns]
        corr_df = corr_df[["NE", "label", "files_id"] + cols_to_import]

        # S'assurer que files_id est du même type (int) pour le merge
        # On drop les NaNs dans l'ID de correction car inexploitable
        corr_df = corr_df.dropna(subset=["files_id"])
        corr_df["files_id"] = corr_df["files_id"].astype(int)

        if verbose:
            print(f"[apply_correction] {len(corr_df)} corrections chargées.")

        # 2. Préparation du DataFrame principal (Explode Strategy)
        # On travaille sur une copie pour ne pas casser l'index ou l'ordre
        df_w_index = df.copy()
        
        # On sauvegarde l'index original pour pouvoir reconstruire le dataframe après l'explode
        df_w_index.index.name = "original_index"
        df_w_index = df_w_index.reset_index()

        # --- Etape Clé : EXPLODE ---
        # Transforme les tuples (1,2,12) en lignes distinctes:
        # Ligne A: id=1
        # Ligne A: id=2
        # Ligne A: id=12
        exploded_df = df_w_index.explode("files_id")

        # Conversion files_id en int (car explode peut laisser des objets si mélange)
        # Gestion des cas où files_id serait vide ou NaN
        exploded_df = exploded_df.dropna(subset=["files_id"])
        exploded_df["files_id"] = exploded_df["files_id"].astype(int)

        # 3. Jointure (Merge)
        # On merge sur les ID éclatés
        merged_df = pd.merge(
            exploded_df, 
            corr_df, 
            on=["NE", "label", "files_id"], 
            how="left", 
            suffixes=("", "_corr")
        )

        # 4. Réagrégation (Collapse)
        # Maintenant on a potentiellement plusieurs lignes pour le même index original.
        # On veut récupérer les corrections s'il y en a une.
        # 'first' suffit car on suppose qu'une entité n'a pas deux corrections contradictoires pour ses différents IDs
        # Si vous voulez prioriser, il faut trier avant.
        
        # On groupe par l'index original et on prend la première valeur non-nulle trouvée pour les colonnes de correction
        corrections_found = merged_df.groupby("original_index")[cols_to_import].first()

        # 5. Injection des données dans le DF original
        # On joint les corrections récupérées sur l'index du df original
        df_final = df.join(corrections_found, rsuffix='_new')

        # Mise à jour des colonnes : si la colonne existe déjà, on la met à jour, sinon on la crée
        for col in cols_to_import:
            if col in df_final.columns and f"{col}_new" in df_final.columns:
                # combine_first : priorise la valeur existante non-null, 
                # MAIS ici on veut appliquer la correction, donc on priorise la correction (_new)
                # update: df_final[col] prend la valeur de _new si _new n'est pas Na
                df_final[col] = df_final[f"{col}_new"].combine_first(df_final[col])
                df_final = df_final.drop(columns=[f"{col}_new"])
            elif f"{col}_new" in df_final.columns:
                # La colonne n'existait pas, on la renomme simplement
                df_final = df_final.rename(columns={f"{col}_new": col})

        # Réorganisation des colonnes
        final_cols_order = []
        # D'abord les colonnes de correction
        for c in correction_columns:
            if c in df_final.columns:
                final_cols_order.append(c)
        # Ensuite le reste
        for c in df_final.columns:
            if c not in final_cols_order:
                final_cols_order.append(c)
        
        df_final = df_final[final_cols_order]

        if verbose:
            print(f"[apply_correction] Shape final: {df_final.shape}")
        
        return df_final

    def run(self, data:pd.DataFrame, dfs:list[pd.DataFrame], correction:str=None):

        if dfs is None or not all(isinstance(df, pd.DataFrame) for df in dfs):
            raise ValueError("[run] 'dfs' must be a list of pandas DataFrames.")
        
        self.load_data(data)
        
        # 1- Merge DataFrames
        df = self._merge(dfs, verbose=self.verbose)

        # 2- Priority
        if self.process_priority_merge:
            df = self.priority_merge(df, labels=self.labels_priority) # labels=["PER"]

        # 3- CasEN Graphs
        if self.process_casen_opti:
            graphs = self.config["casEN_opti2"]
            df = self.keep_precise_graphs(df, graphs, self.verbose)


        # Order the DataFrames
        df = self.order(df)

        # correction
        if correction is not None:
            df = self.apply_correction(df, correction, self.verbose)

        

        return df
        
