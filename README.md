# Document QA — RAG System

A RAG (Retrieval-Augmented Generation) pipeline with a Streamlit chat UI and a simple eval framework. Upload `.txt` or `.md` files, ask questions about them, and get answers with citations pointing back to the source documents.

Built with Python, ChromaDB, sentence-transformers, and the Anthropic API.

---

## How it works
---

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

---

## Example

**Question:** What is the role of the language model in RAG?

**Answer:**
> The language model reads the retrieved chunks and synthesizes a grounded answer [1]. The prompt tells it to only use the provided context and cite sources with [1], [2], etc. [1]. This turns the task from open-ended generation into reading comprehension, which makes answers more reliable [1].

**Sources:** `rag_overview.md`

---

## Files
---

## Stack

- **Embeddings** — sentence-transformers / all-MiniLM-L6-v2 (local, no API cost)
- **Vector DB** — ChromaDB (persistent, cosine similarity)
- **LLM** — Anthropic Claude 3.5 Haiku
- **UI** — Streamlit

---

Built by Anish R. Yerramsetti — Data Engineering and Mathematics @ Texas A&M