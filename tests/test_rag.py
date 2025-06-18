import pytest
from unittest.mock import Mock, patch, MagicMock
from src.rag import BaseRAGPipeline, VectorStoreRAGPipeline
from src.config import BASIC_CORPUS


class TestBaseRAGPipeline:
    """Test BaseRAGPipeline abstract class"""

    def test_init(self):
        """Test BaseRAGPipeline initialization"""

        # Create a concrete implementation for testing
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
        """Test BaseRAGPipeline initialization with default top_k"""

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
    """Test VectorStoreRAGPipeline functionality"""

    def test_init(self):
        """Test VectorStoreRAGPipeline initialization"""
        mock_vector_store = Mock()
        mock_vector_store.documents = ["doc1", "doc2"]

        with patch("src.rag.hf_pipeline") as mock_hf_pipeline:
            mock_gen = Mock()
            mock_tokenizer = Mock()
            mock_gen.tokenizer = mock_tokenizer
            mock_hf_pipeline.return_value = mock_gen

            pipeline = VectorStoreRAGPipeline(mock_vector_store, top_k=3)

            assert pipeline.vector_store == mock_vector_store
            assert pipeline.top_k == 3
            assert (
                pipeline.docs == []
            )  # VectorStoreRAGPipeline initializes with empty docs
            assert pipeline._gen == mock_gen
            assert pipeline._tokenizer == mock_tokenizer

    def test_init_default_top_k(self):
        """Test VectorStoreRAGPipeline initialization with default top_k"""
        mock_vector_store = Mock()
        mock_vector_store.documents = ["doc1"]

        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        assert pipeline.top_k == 5  # Default from config

    def test_retrieve_basic(self):
        """Test basic retrieve functionality"""
        mock_vector_store = Mock()
        mock_vector_store.documents = BASIC_CORPUS.copy()
        mock_vector_store.search.return_value = (
            ["aircraft engines text", "flight forces text"],
            [0.9, 0.8],
            [{"source": "doc1"}, {"source": "doc2"}],
        )

        pipeline = VectorStoreRAGPipeline(mock_vector_store, top_k=2)

        results = pipeline.retrieve("aircraft engines")

        mock_vector_store.search.assert_called_once_with("aircraft engines", k=2)
        assert len(results) == 2
        assert results[0] == "aircraft engines text"
        assert results[1] == "flight forces text"

    def test_retrieve_empty_results(self):
        """Test retrieve with empty search results"""
        mock_vector_store = Mock()
        mock_vector_store.documents = ["doc1"]
        mock_vector_store.search.return_value = ([], [], [])

        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        results = pipeline.retrieve("nonexistent query")

        assert results == ["I do not know"]

    def test_retrieve_empty_question(self):
        """Test retrieve with empty question"""
        mock_vector_store = Mock()

        with patch("src.rag.hf_pipeline"):
            pipeline = VectorStoreRAGPipeline(mock_vector_store)

        results = pipeline.retrieve("")
        assert results == ["I do not know"]

        results = pipeline.retrieve("   ")
        assert results == ["I do not know"]

    def test_run_with_vector_search(self):
        """Test run method with vector search results"""
        mock_vector_store = Mock()
        mock_vector_store.documents = BASIC_CORPUS.copy()
        mock_vector_store.search.return_value = (
            ["The four forces of flight are lift, weight, thrust, and drag"],
            [0.95],
            [{"source": "aviation_doc.pdf", "page": "1"}],
        )

        pipeline = VectorStoreRAGPipeline(mock_vector_store, top_k=1)

        snippets, sources = pipeline.run("What are the four forces of flight?")

        assert len(snippets) == 1
        assert "four forces of flight" in snippets[0].lower()
        assert len(sources) == 1

    def test_run_fallback_to_basic_corpus(self):
        """Test run method fallback to basic corpus when no vector results"""
        mock_vector_store = Mock()
        mock_vector_store.documents = BASIC_CORPUS.copy()
        mock_vector_store.search.return_value = ([], [], [])

        pipeline = VectorStoreRAGPipeline(mock_vector_store, top_k=2)

        snippets, sources = pipeline.run("aviation")

        # Should fall back to basic corpus search
        assert len(snippets) > 0
        assert snippets != ["I do not know"]

    def test_run_no_relevant_content(self):
        """Test run method when no relevant content found"""
        mock_vector_store = Mock()
        mock_vector_store.documents = BASIC_CORPUS.copy()
        mock_vector_store.search.return_value = ([], [], [])

        pipeline = VectorStoreRAGPipeline(mock_vector_store, top_k=1)

        snippets, sources = pipeline.run("completely unrelated query xyz123")

        # Should generate an answer (not return exact "I do not know")
        assert len(snippets) == 1
        assert sources == []
        # The response should be from the LLM generation
        assert len(snippets[0]) > 0

    @patch("src.rag.hf_pipeline")
    def test_init_with_generation_pipeline(self, mock_hf_pipeline):
        """Test VectorStoreRAGPipeline initialization with generation pipeline"""
        mock_vector_store = Mock()
        mock_vector_store.documents = ["doc1", "doc2"]

        # Mock the pipeline and tokenizer
        mock_gen = Mock()
        mock_tokenizer = Mock()
        mock_gen.tokenizer = mock_tokenizer
        mock_hf_pipeline.return_value = mock_gen

        pipeline = VectorStoreRAGPipeline(mock_vector_store, top_k=3)

        assert pipeline.vector_store == mock_vector_store
        assert pipeline.top_k == 3
        assert pipeline._gen == mock_gen
        assert pipeline._tokenizer == mock_tokenizer
        mock_hf_pipeline.assert_called_once()

    @patch("src.rag.hf_pipeline")
    def test_init_generation_pipeline_failure(self, mock_hf_pipeline):
        """Test VectorStoreRAGPipeline initialization when generation pipeline fails"""
        mock_vector_store = Mock()
        mock_vector_store.documents = ["doc1"]

        mock_hf_pipeline.side_effect = Exception("Model load failed")

        # Should handle the exception gracefully by raising it
        with pytest.raises(Exception, match="Model load failed"):
            VectorStoreRAGPipeline(mock_vector_store)

    def test_generate_answer_method(self):
        """Test _generate_answer functionality"""
        mock_vector_store = Mock()

        with patch("src.rag.hf_pipeline") as mock_hf_pipeline:
            mock_gen = Mock()
            mock_gen.return_value = [{"generated_text": "Generated response"}]
            mock_tokenizer = Mock()
            mock_gen.tokenizer = mock_tokenizer
            mock_hf_pipeline.return_value = mock_gen

            pipeline = VectorStoreRAGPipeline(mock_vector_store)

            result = pipeline._generate_answer("test prompt")

            assert result == "Generated response"
            mock_gen.assert_called_once_with("test prompt")

    def test_generate_answer_no_generator(self):
        """Test _generate_answer when generator is not available"""
        mock_vector_store = Mock()

        with patch("src.rag.hf_pipeline") as mock_hf_pipeline:
            mock_gen = Mock()
            mock_tokenizer = Mock()
            mock_gen.tokenizer = mock_tokenizer
            mock_hf_pipeline.return_value = mock_gen

            pipeline = VectorStoreRAGPipeline(mock_vector_store)

            # Mock the _gen to None using setattr to bypass type checking
            with patch.object(pipeline, "_gen", None):
                with pytest.raises(ValueError, match="LLM not initialized"):
                    pipeline._generate_answer("test prompt")

    @patch("src.rag.hf_pipeline")
    def test_generate_answer_with_snippets(self, mock_hf_pipeline):
        """Test generate_answer with valid snippets"""
        mock_vector_store = Mock()

        # Setup generation pipeline mock
        mock_gen = Mock()
        mock_tokenizer = Mock()
        mock_gen.tokenizer = mock_tokenizer
        mock_hf_pipeline.return_value = mock_gen

        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        snippets = ["Aircraft engines include turbojets and turbofans"]
        result = pipeline.generate_answer("What are aircraft engines?", snippets)

        # Should return the snippet-based response
        assert "Aircraft engines include turbojets and turbofans" in result
        assert "Based on the available information" in result

    @patch("src.rag.hf_pipeline")
    def test_generate_answer_i_do_not_know(self, mock_hf_pipeline):
        """Test generate_answer with 'I do not know' snippet"""
        mock_vector_store = Mock()

        # Setup generation pipeline mock
        mock_gen = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_gen.tokenizer = mock_tokenizer
        mock_gen.return_value = [
            {
                "generated_text": "I don't have enough information to answer this question."
            }
        ]
        mock_hf_pipeline.return_value = mock_gen

        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        snippets = ["I do not know"]
        result = pipeline.generate_answer("What is quantum computing?", snippets)

        # Should use LLM generation for unknown topics
        assert len(result) > 0

    def test_build_prompt(self):
        """Test _build_prompt functionality"""
        mock_vector_store = Mock()

        with patch("src.rag.hf_pipeline") as mock_hf_pipeline:
            mock_gen = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
            mock_gen.tokenizer = mock_tokenizer
            mock_hf_pipeline.return_value = mock_gen

            pipeline = VectorStoreRAGPipeline(mock_vector_store)

            prompt = pipeline._build_prompt("Test content")

            assert prompt == "formatted prompt"
            mock_tokenizer.apply_chat_template.assert_called_once()

    def test_build_prompt_no_tokenizer(self):
        """Test _build_prompt when tokenizer is not available"""
        mock_vector_store = Mock()

        with patch("src.rag.hf_pipeline") as mock_hf_pipeline:
            mock_gen = Mock()
            mock_tokenizer = Mock()
            mock_gen.tokenizer = mock_tokenizer
            mock_hf_pipeline.return_value = mock_gen

            pipeline = VectorStoreRAGPipeline(mock_vector_store)

            # Mock the _tokenizer to None using setattr to bypass type checking
            with patch.object(pipeline, "_tokenizer", None):
                with pytest.raises(ValueError, match="LLM not initialized"):
                    pipeline._build_prompt("Test content")

    def test_generate_answer_unexpected_format(self):
        """Test _generate_answer with unexpected output format"""
        mock_vector_store = Mock()

        with patch("src.rag.hf_pipeline") as mock_hf_pipeline:
            mock_gen = Mock()
            mock_gen.return_value = [{"unexpected_key": "value"}]
            mock_tokenizer = Mock()
            mock_gen.tokenizer = mock_tokenizer
            mock_hf_pipeline.return_value = mock_gen

            pipeline = VectorStoreRAGPipeline(mock_vector_store)

            result = pipeline._generate_answer("test prompt")

            assert "Unexpected output format" in result

    def test_generate_answer_exception_handling(self):
        """Test _generate_answer with exception"""
        mock_vector_store = Mock()

        with patch("src.rag.hf_pipeline") as mock_hf_pipeline:
            mock_gen = Mock()
            mock_gen.side_effect = Exception("Generation error")
            mock_tokenizer = Mock()
            mock_gen.tokenizer = mock_tokenizer
            mock_hf_pipeline.return_value = mock_gen

            pipeline = VectorStoreRAGPipeline(mock_vector_store)

            result = pipeline._generate_answer("test prompt")

            assert "Error generating response" in result
            assert "Generation error" in result

    @patch("src.rag.hf_pipeline")
    def test_generate_answer_with_answer_cleaning(self, mock_hf_pipeline):
        """Test generate_answer with answer post-processing"""
        mock_vector_store = Mock()

        # Setup generation pipeline mock
        mock_gen = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_gen.tokenizer = mock_tokenizer

        # Test with Answer: prefix in response
        mock_gen.return_value = [
            {"generated_text": "Some context Answer: This is the cleaned answer"}
        ]
        mock_hf_pipeline.return_value = mock_gen

        pipeline = VectorStoreRAGPipeline(mock_vector_store)

        snippets = ["I do not know"]
        result = pipeline.generate_answer("What is quantum computing?", snippets)

        # Should clean the answer by removing everything before "Answer:"
        assert "This is the cleaned answer" in result
        assert "Some context Answer:" not in result
