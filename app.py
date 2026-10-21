"""Streamlit UI for the Document Q&A RAG Assistant."""
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import build_vector_store, answer_question, DEFAULT_DB_DIR

load_dotenv()
st.set_page_config(page_title="Document Q&A Assistant", page_icon="📚", layout="wide")

st.title("📚 Document Q&A Assistant")
st.caption("Ask questions about your own PDF and text files using Retrieval-Augmented Generation (RAG).")

with st.sidebar:
    st.header("1. Add documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Text-based PDFs work best. Scanned PDFs may need OCR.",
    )
    st.info("Your documents are indexed locally in ChromaDB. Text is sent to the configured API for embedding and answering.")
    if st.button("Build / Refresh Knowledge Base", type="primary", use_container_width=True):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is missing. Add it to your .env file, then restart the app.")
        elif not uploaded_files:
            st.warning("Upload at least one PDF or TXT file first.")
        else:
            upload_dir = Path("data/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            saved_paths = []
            try:
                for old in upload_dir.iterdir():
                    if old.is_file():
                        old.unlink()
                for uploaded in uploaded_files:
                    # Keep filenames simple and avoid path traversal from uploaded filenames.
                    safe_name = Path(uploaded.name).name
                    destination = upload_dir / safe_name
                    destination.write_bytes(uploaded.getvalue())
                    saved_paths.append(str(destination))
                with st.spinner("Loading, splitting, embedding, and indexing documents..."):
                    _, document_count, chunk_count = build_vector_store(saved_paths, DEFAULT_DB_DIR)
                st.session_state["knowledge_base_ready"] = True
                st.session_state["indexed_files"] = [Path(p).name for p in saved_paths]
                st.success(f"Knowledge base ready: {document_count} page/document(s), {chunk_count} chunks.")
            except Exception as exc:
                st.error(f"Could not build the knowledge base: {exc}")

    if st.session_state.get("knowledge_base_ready"):
        st.success("Knowledge base is ready.")
        st.write("Indexed files:")
        for name in st.session_state.get("indexed_files", []):
            st.write(f"• {name}")

st.divider()
st.header("2. Ask a question")
question = st.text_input(
    "Your question",
    placeholder="Example: What are the main stages of a RAG pipeline?",
)
if st.button("Ask", use_container_width=False):
    if not question.strip():
        st.warning("Enter a question first.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing. Add it to your .env file and restart the app.")
    else:
        try:
            with st.spinner("Searching documents and preparing an answer..."):
                answer, sources, matched_docs = answer_question(question)
            st.subheader("Answer")
            st.write(answer)
            st.subheader("Sources")
            if sources:
                for source in sources:
                    st.write(f"- {source}")
            else:
                st.write("No source passages were retrieved.")
            with st.expander("View retrieved passages"):
                for i, doc in enumerate(matched_docs, start=1):
                    st.markdown(f"**Passage {i} — {Path(str(doc.metadata.get('source', 'Unknown'))).name}**")
                    st.caption(f"Page metadata: {doc.metadata.get('page', 'unknown')}")
                    st.write(doc.page_content)
        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")

st.markdown("---")
st.caption("Tip: Ask specific questions and check the cited source passages. RAG can retrieve irrelevant text, so verify important answers.")
