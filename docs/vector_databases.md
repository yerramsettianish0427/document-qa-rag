# Vector Databases and Embeddings

## What is a vector embedding

An embedding is a list of numbers (a vector) that represents the meaning of a piece of text. The all-MiniLM-L6-v2 model produces 384 numbers for any input string. The key property: two strings with similar meaning produce vectors that point in a similar direction in that 384-dimensional space.

Cosine similarity measures how close two vectors are in direction (ignoring magnitude). A score of 1.0 means identical direction (same meaning), 0 means unrelated, -1 means opposite.

## What ChromaDB does

ChromaDB is an open source vector database built for AI apps. You give it:
- a unique id for each record
- the embedding vector
- the original text
- optional metadata (like which file it came from)

And it can search across millions of vectors in milliseconds using an algorithm called HNSW (Hierarchical Navigable Small World). The store persists to disk so you don't have to re-embed everything when you restart.

## HNSW search

HNSW builds a multi-layer graph over the vectors. Higher layers have fewer nodes and bigger jumps — they're like highway routes. Lower layers are dense with short connections. Search starts at the top and greedily navigates toward the query vector, dropping to lower layers as it closes in. This gives approximate nearest-neighbor results in roughly O(log N) time.

## Why cosine similarity

Raw distance between vectors depends on magnitude, which can vary based on text length. Cosine similarity only measures the angle between vectors, so long and short texts with the same meaning still score as similar. ChromaDB stores cosine distance (1 - cosine_similarity), so a distance of 0 means identical and 1 means completely unrelated.

## Choosing an embedding model

| Model | Dimensions | Speed | Notes |
|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | very fast | good for most use cases, runs locally |
| all-mpnet-base-v2 | 768 | moderate | better accuracy, uses more memory |
| text-embedding-3-small | 1536 | API call | OpenAI hosted, no local setup needed |

This project uses all-MiniLM-L6-v2 because it runs locally (no API cost for embedding), loads fast, and works well for document QA.