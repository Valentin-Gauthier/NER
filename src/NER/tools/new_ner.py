import pandas as pd

class NER:

    def __init__(self,dfs:list[pd.DataFrame], verbose:bool=False):
        self.verbose = verbose

        if len(dfs) == 0 or  not isinstance(dfs[0], pd.DataFrame):
            raise "[NER init] DataFrame required to process the NER"
        else: 
            self.dfs = dfs


    def merge(self, dfs: list[pd.DataFrame] = None,
                dedup_on: list[str] = ["file_id", "NER", "NER_label"],
                method_col: str = "method",
                merged_method_name: str = "intersection",
                ) -> pd.DataFrame:
        """
        Fusionne plusieurs DataFrames d'entités NER en regroupant celles qui sont identiques
        selon 'dedup_on' et garde les colonnes spécifiées.

        Args:
            dfs: liste de DataFrames à fusionner.
            dedup_on: colonnes utilisées pour détecter les doublons.
            method_col: nom de la colonne identifiant la méthode (ex. "spacy", "stanza", etc.)
            merged_method_name: valeur à utiliser si plusieurs méthodes trouvent la même entité.

        Returns:
            Un DataFrame fusionné.
        """
        dfs = dfs or self.dfs

        result = pd.concat(dfs, ignore_index=True)

        # Colonnes à conserver (si elles existent)
        columns_to_keep = ["titles", "NER", "NER_label", "desc", "method",
                        "main_graph", "second_graph", "third_graph", "file_id"]
        existing_cols = [col for col in columns_to_keep if col in result.columns]

        # Construction du dictionnaire d'agrégation
        agg_dict = {
            method_col: lambda x: sorted(set(x))  # toutes les méthodes ayant trouvé l'entité
        }

        # Pour toutes les autres colonnes : on garde la première valeur (supposée identique)
        for col in existing_cols:
            if col not in dedup_on and col != method_col:
                agg_dict[col] = "first"

        # Groupement
        grouped = result.groupby(dedup_on, as_index=False).agg(agg_dict)

        # Mise à jour de la colonne method
        grouped[method_col] = grouped[method_col].apply(
            lambda x: merged_method_name if len(x) > 1 else x[0]
        )

        # Tri final
        grouped.sort_values(by="file_id", inplace=True)

        self.df = grouped
        return grouped

    

    def apply_correction(self) -> pd.DataFrame:
        """Auto correct """

        columns = ["manual cat", "extent", "correct", "category"]

        for col in columns:
            self.df[col] = ""

        correction_df = pd.read_excel("C:\\Users\\valen\\Documents\\Informatique-L3\\Stage_NER\\NER\\Ressources\\20231101_correction.xlsx")

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

        
