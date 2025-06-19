import pandas as pd
from .spacy_wrapper import SpaCyConfig
from .casen_config import Casen
from .stanza_wrapper import StanzaConfig
from .ner import Ner



def NER_Consensus(data:pd.DataFrame) -> pd.DataFrame:
    """
        Process every system to recover every entity of the description column

        Parameters:
            - data (pd.DataFrame) : 

        Returns:
            - pd.DataFrame
    
    """
    # init system
    c = Casen()
    Stanza = StanzaConfig()
    Spacy = SpaCyConfig()

    Casen_df = Casen.run(data)
    Stanza_df = StanzaConfig.run(data)
    Spacy_df =  SpaCyConfig.run(data)

    ner = Ner()

    ner_df = Ner.run()

    return ner_df



