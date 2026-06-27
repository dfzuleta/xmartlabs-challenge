"""
Additional integration tests to improve coverage
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from src.config import BASIC_CORPUS


class TestIntegration:
    """Integration tests that test multiple components together"""

    @patch("src.agents.hf_pipeline")
    def test_rag_agent_with_basic_corpus(self, mock_agent_pipeline):
        """Test RAGAgent with VectorStoreRAGPipeline integration.

        VectorStoreRAGPipeline no longer loads Falcon — only the agent does.
        """
        from src.agents import RAGAgent
        from src.rag import VectorStoreRAGPipeline

        mock_gen = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_gen.tokenizer = mock_tokenizer
        mock_gen.return_value = [{"generated_text": "Aviation response"}]
        mock_agent_pipeline.return_value = mock_gen

        mock_vector_store = Mock()
        mock_vector_store.search.return_value = (
            [BASIC_CORPUS[0]],
            [0.9],
            [{"source": "basic_corpus"}],
        )

        rag_pipeline = VectorStoreRAGPipeline(mock_vector_store)
        agent = RAGAgent(rag_pipeline)

        agent.observe("What are the forces of flight?", "user")
        response = agent.act()

        assert len(response) > 0
        assert isinstance(response, str)

    @patch("src.vector_store.faiss")
    @patch("src.vector_store.SentenceTransformer")
    def test_vector_store_edge_cases(self, mock_sentence_transformer, mock_faiss):
        """Test vector store with edge cases"""
        from src.vector_store import VectorStore
        import numpy as np

        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        store = VectorStore()

        # Test empty query search
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_index.search.return_value = (np.array([[0.5]]), np.array([[0]]))

        # Add some documents first
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings

        test_chunks = [
            {"text": "test1", "metadata": {}},
            {"text": "test2", "metadata": {}},
        ]
        store.add_documents(test_chunks)

        # Test search with empty query
        texts, scores, metadata = store.search("", k=1)
        assert texts == []
        assert scores == []
        assert metadata == []

        # Test search with whitespace query
        texts, scores, metadata = store.search("   ", k=1)
        assert texts == []
        assert scores == []
        assert metadata == []

    def test_rag_pipeline_handles_vector_store_error(self):
        """Test RAG pipeline gracefully handles vector store search errors."""
        from src.rag import VectorStoreRAGPipeline, NO_INFO

        mock_vector_store = Mock()
        mock_vector_store.search.side_effect = Exception("FAISS index not ready")

        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        with pytest.raises(Exception, match="FAISS index not ready"):
            pipeline.run("What is a steep turn?")

    def test_text_processing_robustness(self):
        """Test text processing with various input types.

        The new _clean_text preserves paragraph structure: paragraphs are separated
        by \\n\\n, and within each paragraph line-wrapping whitespace is collapsed.
        """
        from src.document_processor import DocumentProcessor

        processor = DocumentProcessor()

        # Single-paragraph inputs: leading/trailing spaces removed, double spaces collapsed
        single_paragraph_cases = [
            "normal text",
            "  leading spaces",
            "trailing spaces  ",
            "  both  ",
            "multiple   spaces   between   words",
        ]
        for text in single_paragraph_cases:
            cleaned = processor._clean_text(text)
            assert isinstance(cleaned, str)
            assert not cleaned.startswith(" ")
            assert not cleaned.endswith(" ")
            assert "  " not in cleaned

        # Within a single paragraph, single newlines (line-wrap) become spaces
        cleaned = processor._clean_text("Line1\nLine2\nLine3")
        assert "\n" not in cleaned
        assert "Line1" in cleaned
        assert "Line3" in cleaned

        # Multi-paragraph inputs preserve the \\n\\n separator
        cleaned = processor._clean_text("Para one.\n\nPara two.")
        assert "Para one." in cleaned
        assert "Para two." in cleaned
        assert "\n\n" in cleaned
