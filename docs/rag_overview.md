# Retrieval-Augmented Generation (RAG)

RAG is an AI architecture that improves LLM answers by pulling in relevant documents at query time instead of relying only on what the model learned during training.

## Why it matters

LLMs have a training cutoff and sometimes hallucinate facts. RAG fixes both:
- lets the model answer questions about documents it never saw during training
- grounds answers in real retrieved text, which reduces made-up responses
- makes answers auditable since you can see exactly which chunk was cited

## How it works

1. Documents get split into chunks and embedded into vectors
2. The vectors are stored in a database (ChromaDB in this project)
3. When a question comes in, it also gets embedded
4. The most similar chunks are retrieved (cosine similarity)
5. Those chunks get passed to the LLM as context along with the question
6. The LLM generates an answer based only on that context

## Chunking

Long documents get split into 500-character chunks with 50-character overlap between them. The overlap makes sure sentences don't get cut off at boundaries. Each chunk is embedded independently.

## Embeddings

This project uses sentence-transformers (all-MiniLM-L6-v2) to generate embeddings locally. It runs on CPU and produces 384-dimensional vectors. Two semantically similar sentences will have a high cosine similarity even if they don't share any words — that's what makes semantic search work.

## ChromaDB

ChromaDB is the vector database. It stores the raw text, the embedding vector, and metadata (which file it came from). It uses HNSW for fast approximate nearest-neighbor search and cosine distance as the similarity metric.

## The language model's role

Once chunks are retrieved, the LLM (Claude) reads them and synthesizes an answer. The prompt explicitly tells it to only use the provided context and cite sources. This turns the problem from open-ended generation into reading comprehension, which LLMs are much more reliable at.