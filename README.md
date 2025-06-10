# 🧠 Named Entity Recognition (NER)

Optimisation of a natural language processing (NLP) project to detect and classify named entities (persons, locations, organizations, etc.) in French texts. 

On utilise diffèrentes methodes : 
- CasEN (un outils basé sur Unitex, fait par des linguistes)
- SpaCy 
- Stanza

Dans un premier temps, on analyse toutes nos descriptions par chaque methodes. Puis c'est lors de la fusion des résultats qu'on utilise de la cross-validation. Si plusieurs system ont trouvé la même entitées alors on le precise, et on obtiens donc une ligne (ex : CasEN_Stanza :  entité trouvé par CasEN et Stanza).
Ensuite on applique on principe de priorité sur les entitées trouvé par le plus de system possible mais avec differentes catégories.
Par exemple si on à CasEN_Stanza qui ont trouvé la même entitées et Spacy qui lui a trouvé aussi mais avec une catégories differentes alors cette entité encore plus de change d'être valid mais surement avec la catégorie de CasEN et Stanza.

![resultat Excel](images/image.png)

On voit bien ici, que casEN_Stanza ont trouvé 'Nora' comme etant une 'PER' mais spacy lui à trouvé 'Nora' comme 'LOC' donc on modifie casEN_Stanza par casEN_Stanza_priority.
 


---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Valentin-Gauthier/NER.git
cd NER
```

### 2. requirement
```bash
pip install -r requirements.txt
```

## ✍️ Author

Valentin — Bachelor’s degree, 3rd year, Computer Science  
Internship at LIFAT - 2025