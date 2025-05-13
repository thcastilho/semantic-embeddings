# semantic-embeddings

[![PyPI version](https://img.shields.io/pypi/v/semantic-embeddings.svg)](https://pypi.org/project/semantic-embeddings/)
[![License](https://img.shields.io/pypi/l/semantic-embeddings.svg)](https://github.com/thcastilho/semantic-embeddings/blob/main/LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Official_Repo-blue?logo=github)](https://github.com/thcastilho/semantic-embeddings)

**semantic-embeddings** is the official implementation of GRaCE algorithm for generating rank-based semantic and contextual embeddings from top-K similarity lists.

This library implements the **RaDE** and **GRaCE** algorithms, which use graph-based measures to create **interpretable**, **effective**, and **unsupervised** embeddings for retrieval, clustering, classification, and visualization.

---

## 🔍 Overview

Unlike traditional embedding techniques that require raw features or supervised training, this package builds representations **entirely from ranked similarity lists** (e.g., from a kNN graph or retrieval system). Each embedding dimension corresponds to a "leader" (reference node).

Key benefits:
- **Unsupervised**: No labels or ground truth needed.
- **Explainable**: Embedding dimensions are semantically grounded.
- **Versatile**: Works for text, images, graphs—any domain with top-K similarities.

---

## 📦 Installation

```bash
pip install semantic-embeddings
```

**Dependencies:**  
- `numpy`
- `tqdm`

Requires Python ≥ 3.7.

---

## 🧠 Algorithms

### RaDE (Rank-based Diffusion Embedding)
- Selects leaders by propagating rank-based affinities through a diffusion process.

### GRaCE (Graph and Rank-based Contextual Embeddings)
- Extends RaDE with *unsupervised effectiveness estimation* (e.g., Reciprocal Density, Accumulated JacMax) and *rank correlation measures* (e.g., Reciprocal Distance, JacMax).

---

## 🛠 Usage

### Input Format

Your input must be a `.txt` file with one ranked list per line (space-separated item IDs):

```
15 3 8 22 7 9 ...
3 2 11 5 6 ...
...
```

Each line is a query, and each number is a retrieved item.

---

### RaDE Example

```python
from sembeddings import RaDE

# Initialize
rade = RaDE(rks_path="data/ranked_lists.txt", rks_size_L=20)

# Compute internal structure
rade.fit(num_candidates=1000, num_leaders=128, t=2)

# Get embedding vectors
embeddings = rade.transform()

# Or do both in one call
embeddings = rade.fit_transform(num_candidates=1000, num_leaders=128, t=2)
```

---

### GRaCE Example

```python
from sembeddings import GRaCE

grace = GRaCE(
    rks_path="data/ranked_lists.txt",
    top_K=20,
    correlation_measure="jacmax",  # or "reciprocal"
    estimation_measure="reciprocal_density",  # or "accjacmax"
    alpha=0.95
)

# Compute internal structure
grace.fit(num_leaders=128)

# Get embedding vectors
embeddings = grace.transform()

# Or do both in one call
embeddings = grace.fit_transform(num_leaders=128)
```

---

## 🔬 Example Applications

### Retrieval

```python
from sklearn.metrics.pairwise import cosine_similarity

query_idx = 0
sims = cosine_similarity(embeddings)
top_k = sims[query_idx].argsort()[::-1][1:11]
print("Top-10 results for query:", top_k)
```

### Clustering

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5).fit(embeddings)
print(kmeans.labels_)
```

### Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels)
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))
```

### Visualization

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

proj = TSNE(n_components=2).fit_transform(embeddings)
plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10")
plt.title("2D Visualization of RaDE Embeddings")
plt.show()
```

---

## 📁 Package Structure

```
semantic-embeddings/
│
├── rade.py                 # RaDE implementation
├── grace.py                # GRaCE implementation
├── utils.py                # Ranked list reader
└── measures/
    ├── qpp.py              # Query performance prediction measures (AccJacMax, Reciprocal Density)
    └── correlation.py      # Rank correlation measures (JacMax, Reciprocal KNN)

```

---

## 📚 Citation

If you use this library in your research, please cite:

### GRaCE *(Accepted, pending publication)*

> **Almeida, T. C. C., Letício, G. R., Valem, L. P., Freitas, A., Pedronette, D. C. G.**  
> *Effective Graph and Rank-based Contextual Embeddings for Textual and Multimedia Data*  
> 2025 International Joint Conference on Neural Networks (IJCNN), Rome – Italy.  
> [![View Paper](https://img.shields.io/badge/Accepted-Pending%20Publication-blue)](#)

---

### RaDE

> **De Fernando, F. A., Pedronette, D. C. G., De Sousa, G. J., Valem, L. P., Guilherme, I. R.**  
> *RaDE: A Rank-based Graph Embedding Approach*  
> 15th International Conference on Computer Vision Theory and Applications (VISAPP), 2020.  
> [![RaDE](https://img.shields.io/badge/View%20Paper-RaDE-blue)](https://doi.org/10.5220/0008985901420152)

> *RaDE+: A Semantic Rank-based Graph Embedding Algorithm*  
> International Journal of Information Management Data Insights, 2022.  
> [![RaDE+](https://img.shields.io/badge/View%20Paper-RaDE%2B-blue)](https://doi.org/10.1016/j.patrec.2022.03.015)

---

## 🤝 Contact

- Thiago César Castilho Almeida: `tc.almeida@unesp.br`  
- Lucas Pascotti Valem: `lucaspascottivalem@gmail.com`  
- Daniel Carlos Guimarães Pedronette: `pedronette@gmail.com`