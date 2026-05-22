import streamlit as st
import time
import tempfile
from pathlib import Path

from rag_pipeline import ingest, ask, get_collection

st.set_page_config(page_title="Document QA", page_icon="📄", layout="wide")

# sidebar
with st.sidebar:
    st.header("📄 Document QA")
    st.caption("upload docs, ask questions")
    st.divider()

    st.subheader("Upload Documents")
    uploads = st.file_uploader(
        "drop .txt or .md files here",
        type=["txt", "md"],
        accept_multiple_files=True
    )

    if st.button("Ingest files", use_container_width=True, type="primary"):
        if not uploads:
            st.warning("upload something first")
        else:
            with st.spinner("chunking and embedding..."):
                with tempfile.TemporaryDirectory() as tmp:
                    for f in uploads:
                        Path(tmp, f.name).write_bytes(f.read())
                    n = ingest(tmp)
            st.success(f"ingested {n} chunks from {len(uploads)} file(s)")

    st.divider()

    # show how many chunks are loaded
    try:
        col = get_collection()
        st.metric("chunks in db", col.count())
    except:
        st.metric("chunks in db", 0)

    st.divider()
    top_k = st.slider("chunks to retrieve", 1, 10, 5)
    show_chunks = st.toggle("show retrieved chunks", value=False)


# main chat area
st.title("Ask your documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# render previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption(f"sources: {', '.join(msg['sources'])}")
        if show_chunks and msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander("retrieved chunks"):
                for i, c in enumerate(msg["chunks"], 1):
                    st.markdown(f"**[{i}]** `{c['source']}` — score `{c['score']}`")
                    st.text(c["text"][:300])

# chat input
if user_q := st.chat_input("ask something about your docs..."):
    st.session_state.chat_history.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            t0 = time.time()
            try:
                result = ask(user_q, top_k=top_k)
                elapsed = round(time.time() - t0, 2)

                st.markdown(result["answer"])

                if result["sources"]:
                    st.caption(f"sources: {', '.join(result['sources'])} ({elapsed}s)")

                if show_chunks:
                    with st.expander("retrieved chunks"):
                        for i, c in enumerate(result["chunks"], 1):
                            st.markdown(f"**[{i}]** `{c['source']}` — score `{c['score']}`")
                            st.text(c["text"][:300])

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "chunks": result["chunks"]
                })
            except Exception as e:
                st.error(f"something went wrong: {e}")