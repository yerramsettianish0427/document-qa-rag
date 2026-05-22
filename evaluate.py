"""
simple eval script to test if retrieval is actually working
runs a few test questions and checks if the right keywords show up in answers
"""

from rag_pipeline import retrieve, ask

# test questions - edit these to match your actual documents
# each one has keywords that should appear if retrieval is working
TEST_QUESTIONS = [
    {
        "question": "What is retrieval-augmented generation?",
        "keywords": ["retrieval", "generation", "context"],
    },
    {
        "question": "What are vector embeddings used for?",
        "keywords": ["similarity", "semantic", "embed"],
    },
    {
        "question": "How does ChromaDB store documents?",
        "keywords": ["chroma", "vector", "store"],
    },
    {
        "question": "What chunking strategy is used?",
        "keywords": ["chunk", "overlap"],
    },
    {
        "question": "What is the role of the language model in RAG?",
        "keywords": ["answer", "context", "generate"],
    },
]


def run_eval():
    passed = 0

    for item in TEST_QUESTIONS:
        q = item["question"]
        keywords = item["keywords"]

        # check retrieval score
        chunks = retrieve(q, top_k=5)
        top_score = chunks[0]["score"] if chunks else 0.0

        # get full answer
        result = ask(q)
        answer_text = result["answer"].lower()

        # count keyword hits
        hits = sum(1 for kw in keywords if kw.lower() in answer_text)
        hit_rate = hits / len(keywords)

        # pass if at least half the keywords appear and retrieval score is decent
        ok = hit_rate >= 0.5 and top_score > 0.3

        if ok:
            passed += 1

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {q}")
        print(f"       keywords: {hits}/{len(keywords)} | top score: {top_score:.3f}")
        print(f"       answer: {result['answer'][:100]}...\n")

    print("=" * 50)
    print(f"results: {passed}/{len(TEST_QUESTIONS)} passed")
    print(f"pass rate: {round(passed/len(TEST_QUESTIONS)*100)}%")
    print("=" * 50)


if __name__ == "__main__":
    run_eval()