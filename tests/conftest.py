"""
Test configuration and fixtures
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Test data
TEST_CHUNKS = [
    {
        "text": "Aircraft engines can be classified as reciprocating engines, turboprops, turbojets, or turbofans.",
        "page": "1",
        "chunk_id": "0",
        "source": "test_doc.pdf",
    },
    {
        "text": "The four fundamental forces of flight are lift, weight, thrust, and drag.",
        "page": "1",
        "chunk_id": "1",
        "source": "test_doc.pdf",
    },
    {
        "text": "Navigation systems include GPS, VOR, ADF, and ILS, providing pilots with accurate position guidance.",
        "page": "2",
        "chunk_id": "2",
        "source": "test_doc.pdf",
    },
]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def test_chunks():
    """Provide test document chunks"""
    return TEST_CHUNKS.copy()


@pytest.fixture
def mock_embedding_model():
    """Mock sentence transformer model"""
    mock_model = Mock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    return mock_model


@pytest.fixture
def mock_gen_model():
    """Mock generation model and tokenizer"""
    mock_gen = Mock()
    mock_tokenizer = Mock()

    # Mock tokenizer
    mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "This is a test response about aviation."
    mock_tokenizer.eos_token_id = 2

    # Mock generation pipeline
    mock_gen.return_value = [{"generated_token_ids": [1, 2, 3, 4, 5, 2]}]

    return mock_gen, mock_tokenizer
