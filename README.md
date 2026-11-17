# Document QA with RAG

A RAG pipeline with a Streamlit chat UI and a small eval script. Upload `.txt` or `.md` files, ask questions about them, and get answers that cite the source documents.

Built with Python, ChromaDB, sentence-transformers, and the Anthropic API.

## How it works

`rag_pipeline.py` splits each file into 500-character chunks with 50 characters of overlap, embeds them locally with `all-MiniLM-L6-v2`, and stores them in a persistent ChromaDB collection (cosine distance). A question is embedded the same way, the top 5 chunks are pulled, and Claude answers using only those chunks, citing them as [1], [2] and so on. `evaluate.py` asks five fixed questions and passes each one if at least half its keywords show up in the answer and the top retrieval score is above 0.3.

## Setup

```bash
git clone https://github.com/yerramsettianish0427/document-qa-rag.git
cd document-qa-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add your Anthropic API key to .env
```

## Run the UI

```bash
streamlit run app.py
```

Then open `http://localhost:8501`, upload documents in the sidebar, and start asking questions.

## Run from Python

```python
from rag_pipeline import ingest, ask

ingest("./docs")
result = ask("What is RAG?")
print(result["answer"])
print("sources:", result["sources"])
```

## Run eval

```bash
python evaluate.py
```

## Example

Question: What is the role of the language model in RAG?

Answer:
> The language model reads the retrieved chunks and synthesizes a grounded answer [1]. The prompt tells it to only use the provided context and cite sources with [1], [2], etc. [1]. This turns the task from open-ended generation into reading comprehension, which makes answers more reliable [1].

Sources: `rag_overview.md`

## Stack

- Embeddings: sentence-transformers `all-MiniLM-L6-v2`, run locally
- Vector DB: ChromaDB, persistent, cosine similarity
- LLM: Claude Haiku 5.5
- UI: Streamlit

Built by Anish R. Yerramsetti, Data Engineering and Mathematics, Texas A&M
