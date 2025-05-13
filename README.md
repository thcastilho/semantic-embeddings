# semantic-embeddings

Package for creating rank-based semantic and contextual embeddings via RaDE and GRaCE algorithms.

## Features

- **Rank-based Graph Embedding**  
  No costly optimization—embeddings derive solely from top-K ranked similarity lists.
- **High-Effectiveness Representatives**  
  Selects “leader” nodes by rank-based measures to act as semantically meaningful dimensions.
- **Interpretable Embeddings**  
  Each vector dimension corresponds to similarity to a representative node.

## Installation

```bash
pip install semantic-embeddings