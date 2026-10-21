"""Core RAG pipeline: load, chunk, embed, store, retrieve, and answer."""
import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DEFAULT_DB_DIR = "data/chroma_db"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_documents(paths: Iterable[str]):
    """Load PDF pages and TXT files into LangChain Documents with source metadata."""
    documents = []
    for item in paths:
        path = Path(item)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.name}. Use PDF or TXT.")
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
            for doc in loaded:
                doc.metadata["page"] = 1
        for doc in loaded:
            doc.metadata["source"] = path.name
        documents.extend(loaded)
    if not documents:
        raise ValueError("No readable text was found in the supplied documents.")
    return documents


def split_documents(documents, chunk_size: int = 900, chunk_overlap: int = 150):
    """Split documents into overlapping chunks while retaining source/page metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def get_embeddings():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model, api_key=api_key)


def build_vector_store(paths: Iterable[str], persist_directory: str = DEFAULT_DB_DIR):
    """Create a fresh persistent ChromaDB index from PDF/TXT documents."""
    documents = load_documents(paths)
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks were produced from the documents.")

    db_path = Path(persist_directory)
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(db_path),
        collection_name="document_qa",
    )
    return vector_store, len(documents), len(chunks)


def open_vector_store(persist_directory: str = DEFAULT_DB_DIR):
    """Open an existing ChromaDB index."""
    db_path = Path(persist_directory)
    if not db_path.exists() or not any(db_path.iterdir()):
        raise FileNotFoundError(
            "Knowledge base not found. Build it first by uploading documents or running ingest."
        )
    return Chroma(
        persist_directory=str(db_path),
        embedding_function=get_embeddings(),
        collection_name="document_qa",
    )


def format_sources(docs):
    """Create readable source/page citations and de-duplicate matching references."""
    citations = []
    seen = set()
    for doc in docs:
        source = Path(str(doc.metadata.get("source", "Unknown source"))).name
        page = doc.metadata.get("page")
        # PyPDFLoader page metadata is zero-based; show human-friendly page numbers.
        page_label = f"page {int(page) + 1}" if isinstance(page, int) and source.lower().endswith(".pdf") else (
            f"page {page}" if page is not None else "page unknown"
        )
        key = (source, page_label)
        if key not in seen:
            seen.add(key)
            citations.append(f"{source}, {page_label}")
    return citations


def answer_question(question: str, vector_store=None, persist_directory: str = DEFAULT_DB_DIR, k: int = 4):
    """Retrieve relevant chunks and answer only from their context."""
    if not question or not question.strip():
        raise ValueError("Please enter a question.")
    if vector_store is None:
        vector_store = open_vector_store(persist_directory)

    matches = vector_store.similarity_search(question, k=k)
    if not matches:
        return "I couldn't find relevant information in the loaded documents.", [], []

    context_parts = []
    for index, doc in enumerate(matches, start=1):
        source = Path(str(doc.metadata.get("source", "Unknown source"))).name
        page = doc.metadata.get("page")
        if isinstance(page, int) and source.lower().endswith(".pdf"):
            page_display = page + 1
        else:
            page_display = page if page is not None else "unknown"
        context_parts.append(
            f"[Source {index}: {source}, page {page_display}]\n{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    prompt = f"""You are a document question-answering assistant.
Answer the question using ONLY the context below.
If the context does not contain enough evidence, say: "I couldn't find that information in the provided documents."
Do not use outside knowledge to fill gaps.
Cite factual claims inline using the source labels exactly, for example [Source 1].
Do not invent source names, page numbers, or facts.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    response = llm.invoke(prompt)
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return answer, format_sources(matches), matches


def main():
    parser = argparse.ArgumentParser(description="Document Q&A assistant using RAG and ChromaDB.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build a knowledge base from PDF/TXT files.")
    ingest_parser.add_argument("--folder", default="sample_documents", help="Folder with PDF/TXT files.")
    ingest_parser.add_argument("--db", default=DEFAULT_DB_DIR, help="ChromaDB persistence folder.")

    ask_parser = subparsers.add_parser("ask", help="Ask a question about an indexed knowledge base.")
    ask_parser.add_argument("--question", required=True, help="Question to ask.")
    ask_parser.add_argument("--db", default=DEFAULT_DB_DIR, help="ChromaDB persistence folder.")

    args = parser.parse_args()
    if args.command == "ingest":
        folder = Path(args.folder)
        if not folder.exists():
            parser.error(f"Folder does not exist: {folder}")
        paths = [str(p) for p in sorted(folder.iterdir())
                 if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not paths:
            parser.error(f"No PDF or TXT files found in {folder}")
        _, doc_count, chunk_count = build_vector_store(paths, args.db)
        print(f"Knowledge base built. Loaded {doc_count} page/document(s) into {chunk_count} chunks.")
    else:
        answer, sources, _ = answer_question(args.question, persist_directory=args.db)
        print("\nANSWER\n======\n" + answer)
        if sources:
            print("\nSOURCES\n=======\n" + "\n".join(f"- {s}" for s in sources))


if __name__ == "__main__":
    main()
