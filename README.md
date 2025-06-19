# 🧠 Named Entity Recognition (NER)

This project focuses on optimizing a natural language processing (NLP) pipeline to detect and classify named entities in **French texts**, across the following categories:

* `PER` – Person
* `LOC` – Location
* `ORG` – Organization
* `MISC` – Miscellaneous

We leverage **multiple NER tools** to maximize accuracy:

* **CasEN**: A linguistic rule-based system based on **Unitex**, developed by linguists.
* **spaCy**: A fast and efficient NLP library.
* **Stanza**: A deep learning-based NLP library from Stanford, well-suited for morphologically rich languages.

---

### 📁 Single vs. Multiple Corpus Processing

We implemented an option that lets you choose whether to generate **one file per description** or a **single file for all descriptions combined**.

To preserve the traceability of each description's origin, we wrap them with custom tags in the merged file:

```xml
<doc id="X">
    [description content]
</doc>
```

This allows the system to:

- ✅ Significantly reduce execution time (more than 2× faster in our tests)

- ✅ Better exploit generic graph-based rules, which can tag all similar entities once one is found

📊 Entity Detection Results

| Mode                     | Total Entities Found | Gain    |
| ------------------------ | -------------------- | ------- |
| One file per description | 9,446                | —       |
| One file for all         | 13,233               | +40.09% |


---

## 🚀 CasEN Optimization

We then evaluated the **precision** and **entity yield** of each graph individually.

This analysis helped us identify certain graphs—or combinations of graphs—that provided the most benefit. We leveraged this insight to **prioritize and retain their extracted entities**, even if they were not detected by other systems.

### 🔍 Example of a Graph Sequence

| Step            | Graph Name               |
|------------------|--------------------------|
| main_graph      | `grfpersCivilitePersonne` |
| second_graph  | `grftagCiviliteS`         |
| third_graph   | `grftagNomFamille`        |

These optimized sequences allow us to improve both recall and consistency across descriptions by capturing entities that would otherwise be missed.


---
## 🔄 Multi-Model Entity Detection & Cross-Validation

Each text description is first processed individually by all three systems (**CasEN**, **spaCy**, and **Stanza**).
Then, we apply a **cross-validation strategy** during result fusion:

### 🧹 Cross-System Agreement

* If multiple systems detect the **same entity**, we merge their outputs and label them accordingly.
* Example: If both **CasEN** and **Stanza** detect "Nora" as a `PER`, the merged method becomes `CasEN_Stanza`.

### ⚖️ Conflict Resolution with Priority Rules

When an entity is detected by **multiple systems with different labels**, we apply **priority rules**:

* Entities found by **more systems** are considered more reliable.
* If systems agree on the **entity** but not on the **label**, we prioritize the **most frequent or reliable label** among agreeing systems.

#### 🧠 Example

![Excel Result Preview](src/images/image.png)

As shown above:

* Both **CasEN** and **Stanza** classify **“Nora”** as a **Person (`PER`)**.
* **spaCy**, however, classifies it as a **Location (`LOC`)**.

📌 As a result, the merged label becomes:

```txt
CasEN_Stanza_priority
```

This indicates that CasEN and Stanza agreed on both the entity and the label, and their interpretation takes precedence over spaCy’s.

---
## 📊 Named Entity Recognition (NER) – Evaluation Results

This section presents the evolution of NER performance across different configurations using **CasEN**, **SpaCy**, **Stanza**, and optimized graph sequences.



### 🧪 Initial Evaluation (CasEN ∩ SpaCy)

Entities detected using the intersection of CasEN and SpaCy systems at the beginning of the pipeline.
```bash
| Category | Total Entities | Accuracy |
|----------|----------------|----------|
| NE       | 4,085          | 97.67%   |
| PER      | 2,744          | 98.69%   |
| LOC      | 1,212          | 98.68%   |
| ORG      | 129            | 66.67%   |
| MISC     | 0              | 0.00%    |

```

### 📁 CasEN on Single Corpus File (CasEN ∩ SpaCy)

Performance after switching to a **single concatenated file** approach for CasEN.
```bash
| Category | Total Entities | Accuracy | Entity Gain | Accuracy Loss |
|----------|----------------|----------|--------------|----------------|
| NE       | 5,327          | ✅ 97.61%   | 🔼 +30.40%     | 🔽 -0.06%         |
| PER      | 4,236          | ✅ 98.31%   | 🔼 +51.37%     | 🔽 -0.37%         |
| LOC      | 952            | ✅ 98.83%   | 🔽 -21.45%     | 🔼 +0.15%         |
| ORG      | 139            | ⚠️ 66.92%   | 🔼 +7.75%      | 🔽 -0.26%         |
| MISC     | 0              | ❌ 0.00%    | ➖ 0.00%       | ➖ 0.00%          |

```

### 🚀 CasEN + Optimized Graphs

Results using **CasEN with graph optimization** strategies.
```bash
| Category | Total Entities | Accuracy | Entity Gain | Accuracy Loss |
|----------|----------------|----------|--------------|----------------|
| NE       | 6,010          | ✅ 97.14%   | 🔼 +12.82%     | 🔽 -0.47%         |
| PER      | 4,491          | ✅ 98.00%   | 🔼 +6.02%      | 🔽 -0.31%         |
| LOC      | 1,294          | ✅ 97.78%   | 🔼 +35.92%     | 🔼 +1.05%         |
| ORG      | 225            | ⚠️ 75.12%   | 🔼 +61.87%     | 🔽 -8.20%         |
| MISC     | 0              | ❌ 0.00%    | ➖ 0.00%       | ➖ 0.00%          |

```

### 🧠 Full System: CasEN + SpaCy + Stanza + Optimization & Priority Rules

Final performance combining **all systems** with **graph priority strategies** and **CasEN optimizations**.
```bash
| Category | Total Entities | Accuracy | Entity Gain | Accuracy Loss |
|----------|----------------|----------|--------------|----------------|
| NE       | 7,086          | ✅ 97.08%   | 🔼 +17.90%     | 🔽 -0.06%         |
| PER      | 5,592          | ✅ 97.37%   | 🔼 +24.52%     | 🔽 -0.63%         |
| LOC      | 1,267          | ✅ 98.30%   | 🔽 -2.09%      | 🔼 +0.52%         |
| ORG      | 227            | ⚠️ 82.84%   | 🔼 +0.89%      | 🔽 -7.72%         |
| MISC     | 0              | ❌ 0.00%    | ➖ 0.00%       | ➖ 0.00%          |
```


#### ✅ Summary

```bash
| Category | Total Entities | Accuracy | Entity Gain | Accuracy Loss |
|----------|----------------|----------|--------------|----------------|
| NE       | 7,086          | ✅97.08%   | 🔼 +73.46%     | 🔽 -0.60%         |
| PER      | 5,592          | ✅97.37%   | 🔼 +103.79%     | 🔽 -1.31%        |
| LOC      | 1,267          | ✅98.30%   | 🔼 +4.54%      | 🔽 -0.38%         |
| ORG      | 227            | ⚠️ 82.84%   | 🔼 +75.97%      | 🔼 +16.18%         |
| MISC     | 0              | ❌ 0.00%    | ➖ 0.00%       | ➖ 0.00%          |
```


---
## 📅 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Valentin-Gauthier/NER.git
cd NER
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ✍️ Author

Valentin — Bachelor’s degree, 3rd year, Computer Science
Internship at LIFAT - 2025
