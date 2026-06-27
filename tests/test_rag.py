import pytest
from unittest.mock import Mock
from src.rag import BaseRAGPipeline, VectorStoreRAGPipeline, NO_INFO, RELEVANCE_THRESHOLD
from src.config import BASIC_CORPUS


class TestBaseRAGPipeline:
    """Test BaseRAGPipeline abstract class"""

    def test_init(self):
        class ConcreteRAGPipeline(BaseRAGPipeline):
            def retrieve(self, question: str):
                return []

            def run(self, question: str):
                return [], []

        docs = ["doc1", "doc2", "doc3"]
        pipeline = ConcreteRAGPipeline(docs, top_k=3)

        assert pipeline.docs == docs
        assert pipeline.top_k == 3

    def test_init_default_top_k(self):
        class ConcreteRAGPipeline(BaseRAGPipeline):
            def retrieve(self, question: str):
                return []

            def run(self, question: str):
                return [], []

        docs = ["doc1", "doc2"]
        pipeline = ConcreteRAGPipeline(docs)

        assert pipeline.docs == docs
        assert pipeline.top_k == 5  # Default from config


class TestVectorStoreRAGPipeline:
    """Test VectorStoreRAGPipeline — Falcon-free, score-threshold pipeline"""

    def test_init(self):
        mock_vector_store = Mock()
        pipeline = VectorStoreRAGPipeline(mock_vector_store, top_k=3)

        assert pipeline.vector_store == mock_vector_store
        assert pipeline.top_k == 3
        assert pipeline.docs == []

    def test_init_default_top_k(self):
        mock_vector_store = Mock()
        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        assert pipeline.top_k == 5

    def test_retrieve_returns_chunks_above_threshold(self):
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = (
            ["Steep turns require 45-degree bank.", "Back pressure maintains altitude."],
            [0.80, 0.72],
            [{}, {}],
        )
        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        result = pipeline.retrieve("What is a steep turn?")

        mock_vector_store.search.assert_called_once_with("What is a steep turn?", k=5)
        assert result == ["Steep turns require 45-degree bank.", "Back pressure maintains altitude."]

    def test_retrieve_returns_no_info_below_threshold(self):
        """Out-of-domain: all chunk scores below RELEVANCE_THRESHOLD → NO_INFO."""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = (
            ["Some aviation chunk"],
            [0.20],  # below 0.4
            [{}],
        )
        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        result = pipeline.retrieve("What is the age of the universe?")

        assert result == [NO_INFO]

    def test_retrieve_returns_no_info_below_exact_threshold(self):
        """Score strictly below threshold is rejected."""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = (
            ["Some chunk"],
            [RELEVANCE_THRESHOLD - 0.001],
            [{}],
        )
        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        result = pipeline.retrieve("test question")

        assert result == [NO_INFO]

    def test_retrieve_returns_no_info_empty_results(self):
        """Empty FAISS results → NO_INFO."""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = ([], [], [])
        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        result = pipeline.retrieve("any question")

        assert result == [NO_INFO]

    def test_retrieve_empty_question(self):
        """Empty or whitespace question skips FAISS search entirely."""
        mock_vector_store = Mock()
        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        assert pipeline.retrieve("") == [NO_INFO]
        assert pipeline.retrieve("   ") == [NO_INFO]
        mock_vector_store.search.assert_not_called()

    def test_generate_answer_joins_all_chunks(self):
        """generate_answer must join all chunks — regression for the single-chunk bug."""
        mock_vector_store = Mock()
        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        snippets = ["Chunk one about steep turns.", "Chunk two about bank angle."]
        result = pipeline.generate_answer(snippets)

        assert "Chunk one about steep turns." in result
        assert "Chunk two about bank angle." in result
        assert "Based on the available information" in result

    def test_generate_answer_no_info_passthrough(self):
        """When snippet is NO_INFO, generate_answer returns NO_INFO directly."""
        mock_vector_store = Mock()
        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        result = pipeline.generate_answer([NO_INFO])

        assert result == NO_INFO

    def test_generate_answer_empty_list(self):
        mock_vector_store = Mock()
        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        result = pipeline.generate_answer([])

        assert result == NO_INFO

    def test_run_aviation_question_returns_answer_and_sources(self):
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = (
            ["Steep turns are performed at 45-degree bank."],
            [0.75],
            [{}],
        )
        pipeline = VectorStoreRAGPipeline(mock_vector_store, top_k=1)

        snippets, sources = pipeline.run("What is a steep turn?")

        assert len(snippets) == 1
        assert "Steep turns are performed" in snippets[0]
        assert sources == [0]

    def test_run_out_of_domain_returns_no_info_and_empty_sources(self):
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = (
            ["Aviation chunk"],
            [0.15],
            [{}],
        )
        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        snippets, sources = pipeline.run("What is the chemical formula for water?")

        assert snippets == [NO_INFO]
        assert sources == []

    def test_run_multiple_chunks_all_included_in_answer(self):
        """All retrieved chunks must appear in the answer — regression for the single-chunk bug."""
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = (
            ["chunk A", "chunk B", "chunk C", "chunk D", "chunk E"],
            [0.80, 0.75, 0.70, 0.65, 0.60],
            [{}, {}, {}, {}, {}],
        )
        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        snippets, sources = pipeline.run("Tell me about aviation maneuvers")

        answer = snippets[0]
        for chunk in ["chunk A", "chunk B", "chunk C", "chunk D", "chunk E"]:
            assert chunk in answer
