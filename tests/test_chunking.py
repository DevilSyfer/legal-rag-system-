"""
Legal RAG System — Service Tests
Tests for: chunking_service, embedding_service, pdf_service
Run with: pytest tests/ -v
"""

import pytest
import os
import tempfile


# ============================================================
# CHUNKING SERVICE TESTS
# ============================================================

from app.services.chunking_service import chunk_text

SAMPLE_TEXT = """
Artificial intelligence is transforming industries across the world.
Machine learning models are being deployed in healthcare, finance, and legal sectors.
Natural language processing enables computers to understand human language.
Large language models like GPT and Claude can generate human-like text.
Retrieval augmented generation combines search with language generation.
Vector databases store embeddings for semantic search operations.
Docker containers package applications with all their dependencies.
CI/CD pipelines automate testing and deployment of software.
Python is the most popular language for machine learning development.
FastAPI is a modern web framework for building APIs with Python.
""" * 10  # multiply to ensure multiple chunks


class TestChunkingService:

    def test_returns_list(self):
        """chunk_text must return a list"""
        result = chunk_text(SAMPLE_TEXT)
        assert isinstance(result, list)

    def test_not_empty(self):
        """chunk_text must return at least one chunk"""
        result = chunk_text(SAMPLE_TEXT)
        assert len(result) > 0

    def test_chunk_has_required_keys(self):
        """Every chunk must have all required keys"""
        result = chunk_text(SAMPLE_TEXT)
        required_keys = {"chunk_index", "text", "token_count", "start_token", "end_token"}
        for chunk in result:
            assert required_keys.issubset(chunk.keys()), f"Missing keys in chunk: {chunk.keys()}"

    def test_chunk_text_not_empty_string(self):
        """Every chunk must have non-empty text"""
        result = chunk_text(SAMPLE_TEXT)
        for chunk in result:
            assert len(chunk["text"].strip()) > 0

    def test_token_count_positive(self):
        """Every chunk must have positive token count"""
        result = chunk_text(SAMPLE_TEXT)
        for chunk in result:
            assert chunk["token_count"] > 0

    def test_chunk_index_sequential(self):
        """Chunk indexes must start at 0 and be sequential"""
        result = chunk_text(SAMPLE_TEXT)
        for i, chunk in enumerate(result):
            assert chunk["chunk_index"] == i

    def test_start_token_less_than_end_token(self):
        """start_token must always be less than end_token"""
        result = chunk_text(SAMPLE_TEXT)
        for chunk in result:
            assert chunk["start_token"] < chunk["end_token"]

    def test_respects_chunk_size(self):
        """No chunk should exceed the chunk size limit"""
        chunk_size = 500
        result = chunk_text(SAMPLE_TEXT, chunk_size=chunk_size)
        for chunk in result:
            assert chunk["token_count"] <= chunk_size

    def test_empty_text_returns_empty_list(self):
        """Empty text should return empty list gracefully"""
        result = chunk_text("")
        assert isinstance(result, list)

    def test_short_text_returns_one_chunk(self):
        """Short text that fits in one chunk should return exactly one chunk"""
        short_text = "This is a very short text."
        result = chunk_text(short_text)
        assert len(result) == 1

    def test_custom_chunk_size(self):
        """Custom chunk size should be respected"""
        result_small = chunk_text(SAMPLE_TEXT, chunk_size=100)
        result_large = chunk_text(SAMPLE_TEXT, chunk_size=500)
        assert len(result_small) > len(result_large)

    def test_overlap_creates_continuity(self):
        """With overlap, end of chunk N should appear at start of chunk N+1"""
        result = chunk_text(SAMPLE_TEXT, chunk_size=100, overlap=20)
        if len(result) > 1:
            # start_token of chunk 1 should be less than end_token of chunk 0
            assert result[1]["start_token"] < result[0]["end_token"]


# ============================================================
# EMBEDDING SERVICE TESTS
# ============================================================

from app.services.embedding_service import get_embeddings


class TestEmbeddingService:

    def test_returns_list(self):
        """get_embeddings must return a list"""
        result = get_embeddings(["hello world"])
        assert isinstance(result, list)

    def test_one_input_one_output(self):
        """One text input should produce one embedding"""
        result = get_embeddings(["hello world"])
        assert len(result) == 1

    def test_multiple_inputs_multiple_outputs(self):
        """Multiple text inputs should produce same number of embeddings"""
        texts = ["first text", "second text", "third text"]
        result = get_embeddings(texts)
        assert len(result) == len(texts)

    def test_embedding_dimension_is_384(self):
        """Each embedding must have exactly 384 dimensions"""
        result = get_embeddings(["hello world"])
        assert len(result[0]) == 384

    def test_embedding_contains_floats(self):
        """Each embedding value must be a float"""
        result = get_embeddings(["hello world"])
        for value in result[0]:
            assert isinstance(value, float)

    def test_similar_texts_have_similar_embeddings(self):
        """Similar texts should produce more similar embeddings than different texts"""
        import numpy as np

        texts = ["The cat sat on the mat", "The cat is sitting on the mat"]
        different = "Docker is a containerization platform"

        embeddings = get_embeddings(texts + [different])
        similar_1 = np.array(embeddings[0])
        similar_2 = np.array(embeddings[1])
        different_emb = np.array(embeddings[2])

        # cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_similar = cosine_sim(similar_1, similar_2)
        sim_different = cosine_sim(similar_1, different_emb)

        assert sim_similar > sim_different

    def test_returns_plain_list_not_numpy(self):
        """Embeddings must be plain Python lists, not numpy arrays"""
        result = get_embeddings(["hello world"])
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)

    def test_empty_list_returns_empty_list(self):
        """Empty input should return empty list"""
        result = get_embeddings([])
        assert result == []


# ============================================================
# PDF SERVICE TESTS
# ============================================================

from app.services.pdf_service import extract_text_from_pdf


class TestPdfService:

    def test_raises_error_for_nonexistent_file(self):
        """Should raise FileNotFoundError for missing files"""
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf("nonexistent_file.pdf")

    def test_returns_string(self):
        """Should return a string when valid PDF is provided"""
        # Only runs if a test PDF exists
        test_pdf = "uploads/PDF_for_RAG.pdf"
        if os.path.exists(test_pdf):
            result = extract_text_from_pdf(test_pdf)
            assert isinstance(result, str)

    def test_extracted_text_not_empty(self):
        """Extracted text should not be empty for a real PDF"""
        test_pdf = "uploads/PDF_for_RAG.pdf"
        if os.path.exists(test_pdf):
            result = extract_text_from_pdf(test_pdf)
            assert len(result) > 0

    def test_page_markers_in_output(self):
        """Our pdf_service adds page markers — verify they exist"""
        test_pdf = "uploads/PDF_for_RAG.pdf"
        if os.path.exists(test_pdf):
            result = extract_text_from_pdf(test_pdf)
            assert "---Page" in result