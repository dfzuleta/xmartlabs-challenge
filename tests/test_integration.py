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
    @patch("src.rag.hf_pipeline")
    def test_rag_agent_with_basic_corpus(self, mock_rag_pipeline, mock_agent_pipeline):
        """Test RAGAgent with basic corpus integration"""
        from src.agents import RAGAgent
        from src.rag import VectorStoreRAGPipeline

        # Setup mocks for both pipelines
        mock_gen = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_gen.tokenizer = mock_tokenizer
        mock_gen.return_value = [{"generated_text": "Aviation response"}]

        mock_rag_pipeline.return_value = mock_gen
        mock_agent_pipeline.return_value = mock_gen

        # Create mock vector store
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = (
            [BASIC_CORPUS[0]],
            [0.9],
            [{"source": "basic_corpus"}],
        )

        # Create RAG pipeline and agent
        rag_pipeline = VectorStoreRAGPipeline(mock_vector_store)
        agent = RAGAgent(rag_pipeline)

        # Test the interaction
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

    @patch("src.rag.hf_pipeline")
    def test_rag_pipeline_error_recovery(self, mock_hf_pipeline):
        """Test RAG pipeline error recovery mechanisms"""
        from src.rag import VectorStoreRAGPipeline

        # Setup failing pipeline
        mock_gen = Mock()
        mock_gen.side_effect = Exception("Pipeline failure")
        mock_tokenizer = Mock()
        mock_gen.tokenizer = mock_tokenizer
        mock_hf_pipeline.return_value = mock_gen

        mock_vector_store = Mock()
        mock_vector_store.search.return_value = (["test content"], [0.9], [{}])

        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        # Test that errors are handled gracefully
        result = pipeline._generate_answer("test prompt")
        assert "Error generating response" in result

    def test_text_processing_robustness(self):
        """Test text processing with various input types"""
        from src.document_processor import DocumentProcessor

        processor = DocumentProcessor()

        # Test various whitespace scenarios
        test_cases = [
            "normal text",
            "  leading spaces",
            "trailing spaces  ",
            "  both  ",
            "\ttabs\t",
            "\nnewlines\n",
            "mixed\t\n  whitespace  \r\n",
            "multiple   spaces   between   words",
        ]

        for text in test_cases:
            cleaned = processor._clean_text(text)
            assert isinstance(cleaned, str)
            assert not cleaned.startswith(" ")
            assert not cleaned.endswith(" ")
            assert "  " not in cleaned  # No double spaces
            assert "\n" not in cleaned
            assert "\t" not in cleaned
