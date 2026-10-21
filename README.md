# Document Q&A Assistant with RAG & Vector Embeddings

A Python Retrieval-Augmented Generation (RAG) application that lets you ask questions about your own PDF and text documents. It chunks the documents, creates vector embeddings, stores them in ChromaDB, retrieves relevant passages, and answers using the retrieved context with source/page citations.

## Features

- Upload PDF and TXT documents using a Streamlit interface.
- Load PDFs with page metadata, so answers can cite the source file and page.
- Split long documents using `RecursiveCharacterTextSplitter`.
- Generate embeddings with OpenAI's embedding model.
- Store and search vectors with ChromaDB.
- Answer only from retrieved context; say when the context does not contain the answer.
- Show retrieved source snippets and page numbers.
- Includes a CLI and basic tests for helper functions.

> Important: this starter project uses OpenAI API for embeddings and answer generation. API usage may incur costs. Never upload your `.env` file or API key to GitHub.

## Project structure

```text
document_qa_rag_assistant/
├── app.py
├── rag_pipeline.py
├── tests/
│   └── test_helpers.py
├── sample_documents/
│   └── sample_notes.txt
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 1. Requirements

- Python 3.10+
- An OpenAI API key with access to embeddings and chat completions

## 2. Install dependencies

Create and activate a virtual environment (recommended):

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

## 3. Configure the API key

Copy `.env.example` to `.env` and add your key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Do not commit `.env`. The `.gitignore` file excludes it.

## 4. Run the web app

From the project root:

```bash
streamlit run app.py
```

1. Enter your API key in `.env`.
2. Upload one or more PDF or TXT documents in the sidebar.
3. Click **Build / Refresh Knowledge Base**.
4. Type a question and click **Ask**.
5. Review the answer and source citations.

The app rebuilds the local ChromaDB collection when you build the knowledge base. Uploaded files are saved under `data/uploads/`; the vector database is stored under `data/chroma_db/`. Both are ignored by Git to avoid accidentally committing private documents or generated data.

## 5. Use the command line (optional)

Build the index from a folder containing PDF/TXT files:

```bash
python rag_pipeline.py ingest --folder sample_documents
```

Ask a question:

```bash
python rag_pipeline.py ask --question "What is retrieval-augmented generation?"
```

The first command creates the local ChromaDB index. Run the ask command after ingestion.

## 6. Run tests

```bash
python -m unittest discover -s tests -v
```

These tests check utility behavior and do not call the API.

## How the RAG pipeline works

1. **Load:** read PDF pages and TXT documents.
2. **Chunk:** split documents into smaller, overlapping passages.
3. **Embed:** convert chunks into vectors using an embedding model.
4. **Store:** persist vectors in ChromaDB.
5. **Retrieve:** find the chunks most similar to the user's question.
6. **Generate:** provide only retrieved context to the chat model.
7. **Cite:** display source filenames and page numbers from chunk metadata.

## Example

The included `sample_documents/sample_notes.txt` can be indexed with the CLI. Example question:

> What are the main stages of a RAG pipeline?

The answer should be grounded in the sample document and cite the text file as its source.

## GitHub submission

1. Create an empty GitHub repository, for example `document-qa-rag-assistant`.
2. Extract this ZIP and open a terminal in the project folder.
3. Run:

```bash
git init
git add .
git commit -m "Build document Q&A RAG assistant"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/document-qa-rag-assistant.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

Before submitting, test the app locally, verify source citations, and make sure `.env`, private documents, and the generated vector database are not committed.

## Limitations and future improvements

- Scanned/image-only PDFs need OCR before text can be extracted.
- Answers depend on the quality and relevance of retrieved chunks.
- API access and usage costs depend on your OpenAI account.
- Useful extensions: add OCR, login, document deletion, multi-user storage, and a local embedding/LLM option.
