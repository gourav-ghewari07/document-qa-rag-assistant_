"""Unit tests for helper functions that do not require an API key."""
import unittest
from langchain_core.documents import Document
from rag_pipeline import split_documents, format_sources


class RAGHelperTests(unittest.TestCase):
    def test_split_documents_returns_chunks(self):
        docs = [Document(page_content=("RAG retrieves relevant context. " * 80),
                         metadata={"source": "notes.txt", "page": 1})]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.page_content.strip() for chunk in chunks))
        self.assertTrue(all(chunk.metadata["source"] == "notes.txt" for chunk in chunks))

    def test_pdf_page_citation_is_human_friendly(self):
        docs = [Document(page_content="Some text", metadata={"source": "guide.pdf", "page": 0})]
        self.assertEqual(format_sources(docs), ["guide.pdf, page 1"])

    def test_text_page_citation(self):
        docs = [Document(page_content="Some text", metadata={"source": "notes.txt", "page": 1})]
        self.assertEqual(format_sources(docs), ["notes.txt, page 1"])

    def test_chunk_overlap_must_be_smaller_than_chunk_size(self):
        docs = [Document(page_content="Some text", metadata={"source": "notes.txt"})]
        with self.assertRaises(ValueError):
            split_documents(docs, chunk_size=100, chunk_overlap=100)


if __name__ == "__main__":
    unittest.main()
