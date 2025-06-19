import pandas as pd
from .spacy_wrapper import SpaCyConfig
from .casen_config import CasenConfig
from .stanza_wrapper import StanzaConfig
from .ner_config import NerConfig



def NER_Consensus(data:pd.DataFrame) -> pd.DataFrame:
    """
        Process every system to recover every entity of the description column from the data's DataFrame

        Parameters:
            - data (pd.DataFrame) : 

        Returns:
            - pd.DataFrame
    
    """
    # init system
    c = CasenConfig()
    sp = SpaCyConfig()
    st = StanzaConfig()
    
    Casen_df = c.run(data)
    Spacy_df =  sp.run(data)
    Stanza_df = st.run(data)

    ner = NerConfig()

    ner_df = ner.run(data=data, dfs=[Casen_df, Spacy_df ,Stanza_df])

    return ner_df



