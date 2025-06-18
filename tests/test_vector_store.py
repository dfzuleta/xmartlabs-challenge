"""
Tests for the VectorStore class
"""

import pytest
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

from src.vector_store import VectorStore


class TestVectorStore:
    """Test VectorStore functionality"""

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    def test_init(self, mock_faiss, mock_sentence_transformer):
        """Test VectorStore initialization"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        store = VectorStore()

        assert store.embedding_model_name == "all-MiniLM-L6-v2"
        assert store.store_path == "vector_store"
        assert store.model == mock_model
        assert store.index is None
        assert store.documents == []
        assert store.metadata == []
        assert store.embeddings is None

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    def test_init_with_custom_params(self, mock_faiss, mock_sentence_transformer):
        """Test VectorStore initialization with custom parameters"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        store = VectorStore(embedding_model="custom-model", store_path="/custom/path")

        assert store.embedding_model_name == "custom-model"
        assert store.store_path == "/custom/path"
        mock_sentence_transformer.assert_called_with("custom-model")

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    def test_add_documents_empty_chunks(self, mock_faiss, mock_sentence_transformer):
        """Test adding empty chunks raises error"""
        store = VectorStore()

        with pytest.raises(ValueError, match="No chunks provided"):
            store.add_documents([])

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    def test_add_documents_new_index(
        self, mock_faiss, mock_sentence_transformer, test_chunks
    ):
        """Test adding documents creates new index"""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        # Mock embeddings
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        mock_model.encode.return_value = embeddings

        # Mock FAISS
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()

        store = VectorStore()
        store.add_documents(test_chunks)

        # Verify calls
        mock_model.encode.assert_called_once()
        mock_faiss.IndexFlatIP.assert_called_once_with(3)  # embedding dimension
        mock_faiss.normalize_L2.assert_called()
        mock_index.add.assert_called_once()

        # Verify state
        assert store.index == mock_index
        assert len(store.documents) == 3
        assert len(store.metadata) == 3
        assert store.embeddings is not None

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    def test_add_documents_extend_existing(
        self, mock_faiss, mock_sentence_transformer, test_chunks
    ):
        """Test adding documents to existing index"""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        # Mock embeddings
        existing_embeddings = np.array([[0.1, 0.2, 0.3]])
        new_embeddings = np.array([[0.4, 0.5, 0.6]])
        mock_model.encode.return_value = new_embeddings

        # Mock FAISS
        mock_index = Mock()
        mock_faiss.normalize_L2 = Mock()

        store = VectorStore()
        # Setup existing state
        store.index = mock_index
        store.embeddings = existing_embeddings
        store.documents = ["existing doc"]
        store.metadata = [{"existing": "metadata"}]

        # Add new document
        new_chunk = [test_chunks[0]]
        store.add_documents(new_chunk)

        # Verify calls
        mock_index.add.assert_called_once()

        # Verify state extended
        assert len(store.documents) == 2
        assert len(store.metadata) == 2
        assert store.embeddings.shape[0] == 2

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    def test_search_empty_query(self, mock_faiss, mock_sentence_transformer):
        """Test search with empty query returns empty results"""
        store = VectorStore()

        results = store.search("")
        assert results == ([], [], [])

        results = store.search("   ")
        assert results == ([], [], [])

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    def test_search_no_documents(self, mock_faiss, mock_sentence_transformer):
        """Test search with no documents returns empty results"""
        store = VectorStore()

        results = store.search("test query")
        assert results == ([], [], [])

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    def test_search_success(self, mock_faiss, mock_sentence_transformer, test_chunks):
        """Test successful search"""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        # Mock query embedding
        query_embedding = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = query_embedding

        # Mock FAISS index
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8]]),  # scores
            np.array([[0, 1]]),  # indices
        )
        mock_faiss.normalize_L2 = Mock()

        store = VectorStore()
        store.index = mock_index
        store.documents = ["doc1", "doc2", "doc3"]
        store.metadata = test_chunks

        texts, scores, metadata = store.search("test query", k=2)

        # Verify results
        assert len(texts) == 2
        assert len(scores) == 2
        assert len(metadata) == 2
        assert texts == ["doc1", "doc2"]
        assert scores == [0.9, 0.8]

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    def test_search_with_threshold(
        self, mock_faiss, mock_sentence_transformer, test_chunks
    ):
        """Test search with score threshold"""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        query_embedding = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = query_embedding

        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.3]]),  # scores
            np.array([[0, 1]]),  # indices
        )
        mock_faiss.normalize_L2 = Mock()

        store = VectorStore()
        store.index = mock_index
        store.documents = ["doc1", "doc2"]
        store.metadata = test_chunks[:2]

        texts, scores, metadata = store.search("test query", k=2, score_threshold=0.5)

        # Only first result should pass threshold
        assert len(texts) == 1
        assert scores[0] == 0.9

    def test_get_stats_empty(self):
        """Test get_stats with empty store"""
        with patch("src.vector_store.SentenceTransformer"):
            store = VectorStore()
            stats = store.get_stats()

            assert stats["num_documents"] == 0
            assert stats["num_chunks"] == 0
            assert stats["index_size"] == 0

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    def test_get_stats_with_data(
        self, mock_faiss, mock_sentence_transformer, test_chunks
    ):
        """Test get_stats with data"""
        mock_index = Mock()
        mock_index.ntotal = 100

        store = VectorStore()
        store.index = mock_index
        store.documents = ["doc1", "doc2"]
        store.metadata = test_chunks[:2]

        stats = store.get_stats()

        assert stats["num_documents"] == 2
        assert stats["num_chunks"] == 2
        assert stats["index_size"] == 100

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    @patch("builtins.open", create=True)
    @patch("src.vector_store.pickle")
    @patch("src.vector_store.np")
    @patch("src.vector_store.json")
    @patch("src.vector_store.os")
    def test_save(
        self,
        mock_os,
        mock_json,
        mock_np,
        mock_pickle,
        mock_open,
        mock_faiss,
        mock_sentence_transformer,
    ):
        """Test saving vector store"""
        # Setup mocks
        mock_os.makedirs = Mock()
        mock_os.path.join = Mock(side_effect=lambda *args: "/".join(args))
        mock_faiss.write_index = Mock()

        mock_index = Mock()
        store = VectorStore()
        store.index = mock_index
        store.documents = ["doc1", "doc2"]
        store.metadata = [{"meta": "1"}, {"meta": "2"}]
        store.embeddings = np.array([[1, 2], [3, 4]])
        store.store_path = "/test/path"

        store.save()

        # Verify calls
        mock_os.makedirs.assert_called_with("/test/path", exist_ok=True)
        mock_faiss.write_index.assert_called_once()
        mock_np.save.assert_called_once()

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    @patch("src.vector_store.os")
    def test_load_no_store_path(self, mock_os, mock_faiss, mock_sentence_transformer):
        """Test load returns False when store path doesn't exist"""
        mock_os.path.exists.return_value = False

        store = VectorStore()
        result = store.load()

        assert result is False

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    @patch("src.vector_store.os")
    @patch("builtins.open", create=True)
    @patch("src.vector_store.pickle")
    @patch("src.vector_store.np")
    @patch("src.vector_store.json")
    def test_load_success(
        self,
        mock_json,
        mock_np,
        mock_pickle,
        mock_open,
        mock_os,
        mock_faiss,
        mock_sentence_transformer,
    ):
        """Test successful load"""
        # Setup mocks
        mock_os.path.exists.return_value = True
        mock_os.path.join = Mock(side_effect=lambda *args: "/".join(args))

        mock_json.load.return_value = {"embedding_model": "test", "num_documents": 2}
        mock_faiss.read_index.return_value = Mock()
        mock_pickle.load.side_effect = [
            ["doc1", "doc2"],
            [{"meta": "1"}, {"meta": "2"}],
        ]
        mock_np.load.return_value = np.array([[1, 2], [3, 4]])

        store = VectorStore()
        result = store.load()

        assert result is True
        assert len(store.documents) == 2
        assert len(store.metadata) == 2

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    @patch("src.vector_store.os.path.exists")
    def test_load_missing_files(
        self, mock_exists, mock_faiss, mock_sentence_transformer
    ):
        """Test load when some files are missing"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        # Mock file existence - simulate some files missing
        def side_effect(path):
            if "documents.pkl" in path:
                return False  # Documents file missing
            return True

        mock_exists.side_effect = side_effect

        store = VectorStore()

        # Should handle missing files gracefully
        store.load()

        # Verify that it handles the missing file case
        assert store.documents == []

    @patch("src.vector_store.SentenceTransformer")
    @patch("src.vector_store.faiss")
    def test_search_with_threshold_filtering(
        self, mock_faiss, mock_sentence_transformer
    ):
        """Test search with score threshold filtering"""
        import numpy as np

        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        # Setup FAISS mock
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()

        store = VectorStore()

        # Add test documents
        test_chunks = [
            {"text": "high relevance", "metadata": {"id": 1}},
            {"text": "low relevance", "metadata": {"id": 2}},
        ]

        # Mock embeddings
        embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = embeddings

        store.add_documents(test_chunks)

        # Mock search results with different scores
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_index.search.return_value = (
            np.array([[0.9, 0.3]]),  # High score, low score
            np.array([[0, 1]]),  # Indices
        )

        # Test with score threshold
        texts, scores, metadata = store.search("test", k=2, score_threshold=0.5)

        # Should only return results above threshold
        assert len(texts) == 1  # Only high score result
        assert texts[0] == "high relevance"
        assert scores[0] == 0.9
