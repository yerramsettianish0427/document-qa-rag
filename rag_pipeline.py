import os
import glob
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv

load_dotenv()


EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
OVERLAP = 50
TOP_K = 5


embedder = None
collection = None


def get_embedder():
    global embedder
    if embedder is None:
        print("loading embedding model...")
        embedder = SentenceTransformer(EMBED_MODEL)
    return embedder


def get_collection():
    global collection
    if collection is None:
        client = chromadb.PersistentClient(path="./chroma_store")
        collection = client.get_or_create_collection(
            name="my_docs",
            metadata={"hnsw:space": "cosine"}
        )
    return collection


def chunk_text(text, source_name):
    """split a document into overlapping chunks"""
    chunks = []
    i = 0
    idx = 0
    while i < len(text):
        chunk = text[i : i + CHUNK_SIZE].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "source": source_name,
                "idx": idx
            })
        i += CHUNK_SIZE - OVERLAP
        idx += 1
    return chunks


def ingest(docs_folder="./docs"):
    """read all txt/md files, chunk them, embed, store in chroma"""
    col = get_collection()
    model = get_embedder()

    files = glob.glob(os.path.join(docs_folder, "*.txt")) + \
            glob.glob(os.path.join(docs_folder, "*.md"))

    if not files:
        raise FileNotFoundError(f"no files found in {docs_folder}")

    all_chunks = []
    for f in files:
        fname = Path(f).name
        text = open(f, encoding="utf-8").read()
        all_chunks.extend(chunk_text(text, fname))

    print(f"ingesting {len(all_chunks)} chunks from {len(files)} files...")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    ids = [f"{c['source']}_{c['idx']}" for c in all_chunks]
    metadatas = [{"source": c["source"], "idx": c["idx"]} for c in all_chunks]

    # upsert so i can re-run without duplicating everything
    col.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"done. {col.count()} total chunks in db")
    return len(all_chunks)


def retrieve(query, top_k=TOP_K):
    """embed the query and pull the most relevant chunks"""
    model = get_embedder()
    col = get_collection()

    q_embed = model.encode([query]).tolist()

    results = col.query(
        query_embeddings=q_embed,
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "score": round(1 - dist, 4)
        })

    return chunks


def ask(query, top_k=TOP_K):
    """full rag pipeline - retrieve chunks then ask claude"""
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {"answer": "nothing in the db yet, run ingest() first", "sources": [], "chunks": []}

    # build context block with chunk numbers for citations
    context = ""
    for i, c in enumerate(chunks, 1):
        context += f"[{i}] source: {c['source']} (score: {c['score']})\n{c['text']}\n\n"

    prompt = f"""Answer the question below using ONLY the context provided.
Cite sources with [1], [2], etc.
If you can't answer from the context, say so.

Context:
{context}

Question: {query}"""

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=4096,  # thinking counts against this
        messages=[{"role": "user", "content": prompt}]
    )

    # the first block can be a thinking block, so take the text ones
    answer = "".join(b.text for b in resp.content if b.type == "text")
    sources = list(set(c["source"] for c in chunks))

    return {"answer": answer, "sources": sources, "chunks": chunks}