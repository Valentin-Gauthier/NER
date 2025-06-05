import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker_tab')

class NLTK:
    def __init__(self, data:str, verbose:bool=False):
        self.data = data
        self.df = None
        self.verbose = verbose

        if isinstance(self.data, str):
            self.data_df = pd.read_excel(self.data)


    def run(self) -> pd.DataFrame:
        """Make a DataFrame with the result of NLTK NER analysis (only PERSON, ORGANIZATION, GPE)"""
        rows = []
        for idx, row in self.data_df.iterrows():
            desc = row.get("desc", "")
            if not isinstance(desc, str):
                continue

            # Tokenize and POS-tag
            tokens = nltk.word_tokenize(desc)
            pos_tags = nltk.pos_tag(tokens)

            # Named Entity Recognition (shallow parsing)
            ne_tree = nltk.ne_chunk(pos_tags)

            for subtree in ne_tree:
                if isinstance(subtree, nltk.Tree):
                    label = subtree.label()
                    if label in ("PERSON", "ORGANIZATION", "GPE"):  # Only keep desired types
                        entity = " ".join([token for token, tag in subtree])
                        if self.verbose:
                            print(f"[nltk] file {idx} entity {entity} label {label}")
                        rows.append({
                            "titles": row.get("titles", ""),
                            "NER": entity,
                            "NER_label": label,
                            "desc": desc,
                            "method": "NLTK",
                            "file_id": idx
                        })

        self.df = pd.DataFrame(rows)
        return self.df
