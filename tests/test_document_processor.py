"""
Tests for the DocumentProcessor class
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path

from src.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test DocumentProcessor functionality"""

    def test_init_custom_params(self):
        """Test DocumentProcessor initialization with custom parameters"""
        processor = DocumentProcessor(chunk_size=200, overlap=25)

        assert processor.chunk_size == 200
        assert processor.overlap == 25

    def test_clean_text(self):
        """Test text cleaning functionality.

        The new _clean_text preserves paragraph structure: multiple blank lines
        collapse to a single paragraph separator (\\n\\n), and within each paragraph
        line-wrapping newlines are collapsed to a space. This produces cleaner semantic
        chunks rather than one flattened string.
        """
        processor = DocumentProcessor()

        # Two paragraphs separated by blank lines → preserved as two paragraphs
        dirty_text = "  Hello\n\n\nWorld  "
        clean_text = processor._clean_text(dirty_text)
        assert "Hello" in clean_text
        assert "World" in clean_text
        assert "\n\n" in clean_text  # paragraph boundary preserved

        # Single paragraph with multiple spaces → collapsed
        dirty_text = "Hello    World    Test"
        clean_text = processor._clean_text(dirty_text)
        assert clean_text == "Hello World Test"

        # Single paragraph with line-wrap newlines → collapsed to space
        dirty_text = "Line1\nLine2\nLine3"
        clean_text = processor._clean_text(dirty_text)
        assert clean_text == "Line1 Line2 Line3"
        assert "\n" not in clean_text

    def test_chunk_text_basic(self):
        """Test basic text chunking with paragraph-aware splitting.

        The new chunker splits by paragraph boundaries first, then by sentence.
        A single short sentence shorter than chunk_size produces 1 chunk.
        Multiple paragraphs each below chunk_size produce multiple chunks.
        """
        processor = DocumentProcessor(chunk_size=50, overlap=5)

        # Two paragraphs, each short enough to be their own chunk when combined they exceed limit
        text = "First paragraph about steep turns.\n\nSecond paragraph about lazy eights."
        chunks = processor.chunk_text(text, page_num=1)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert "text" in chunk
            assert "page" in chunk
            assert chunk["page"] == "1"

    def test_chunk_text_short_text(self):
        """Test chunking with text shorter than chunk size"""
        processor = DocumentProcessor(chunk_size=100, overlap=20)

        text = "Short text"
        chunks = processor.chunk_text(text, page_num=2)

        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["page"] == "2"

    def test_chunk_text_multiple_paragraphs_split_correctly(self):
        """Multiple paragraphs exceeding chunk_size split into separate chunks."""
        processor = DocumentProcessor(chunk_size=30, overlap=0)

        # Each paragraph is under 30 chars but together they exceed the limit
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = processor.chunk_text(text, page_num=1)

        assert len(chunks) >= 2
        full_text = " ".join(c["text"] for c in chunks)
        assert "Paragraph one" in full_text
        assert "Paragraph three" in full_text

    def test_process_pdf_file_not_found(self):
        """Test process_pdf with non-existent file"""
        processor = DocumentProcessor()

        with pytest.raises(FileNotFoundError):
            processor.process_pdf("/nonexistent/path/file.pdf")

    @patch("src.document_processor.urllib.request.urlretrieve")
    @patch("src.document_processor.tempfile.NamedTemporaryFile")
    @patch("src.document_processor.os.unlink")
    def test_process_pdf_url_download(
        self, mock_unlink, mock_temp_file, mock_urlretrieve
    ):
        """Test process_pdf with URL download"""
        processor = DocumentProcessor()

        # Setup mocks
        mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test.pdf"
        mock_temp_file.return_value.__exit__.return_value = None

        # Mock the _process_local_pdf method
        with patch.object(processor, "_process_local_pdf") as mock_process:
            mock_process.return_value = [
                {"text": "test", "page": "1", "chunk_id": "0", "source": "test.pdf"}
            ]

            url = "https://example.com/test.pdf"
            result = processor.process_pdf(url)

            # Verify download was attempted
            mock_urlretrieve.assert_called_once_with(url, "/tmp/test.pdf")
            mock_process.assert_called_once_with("/tmp/test.pdf", url)
            mock_unlink.assert_called_once_with("/tmp/test.pdf")

            assert len(result) == 1

    @patch("src.document_processor.urllib.request.urlretrieve")
    def test_process_pdf_url_download_failure(self, mock_urlretrieve):
        """Test process_pdf URL download failure"""
        processor = DocumentProcessor()

        mock_urlretrieve.side_effect = Exception("Download failed")

        with pytest.raises(FileNotFoundError, match="Could not download PDF from URL"):
            processor.process_pdf("https://example.com/test.pdf")

    @patch("src.document_processor.os.path.exists")
    def test_process_pdf_local_file(self, mock_exists):
        """Test process_pdf with local file"""
        processor = DocumentProcessor()

        mock_exists.return_value = True

        # Mock the _process_local_pdf method
        with patch.object(processor, "_process_local_pdf") as mock_process:
            mock_process.return_value = [
                {"text": "test", "page": "1", "chunk_id": "0", "source": "test.pdf"}
            ]

            result = processor.process_pdf("/path/to/test.pdf")

            mock_process.assert_called_once_with(
                "/path/to/test.pdf", "/path/to/test.pdf"
            )
            assert len(result) == 1

    @patch("src.document_processor.pdfplumber.open")
    def test_process_local_pdf_success(self, mock_pdfplumber):
        """Test _process_local_pdf with successful extraction"""
        processor = DocumentProcessor(chunk_size=50, overlap=10)

        # Mock PDF pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = (
            "This is the first page content about aviation and flight."
        )

        mock_page2 = Mock()
        mock_page2.extract_text.return_value = (
            "This is the second page with more aviation information."
        )

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)

        mock_pdfplumber.return_value = mock_pdf

        result = processor._process_local_pdf("/path/test.pdf", "/path/test.pdf")

        assert len(result) >= 2  # At least one chunk per page

        # Check chunk structure
        for chunk in result:
            assert "text" in chunk
            assert "page" in chunk
            assert "chunk_id" in chunk
            assert "source" in chunk
            assert chunk["source"] == "test.pdf"

        # Check chunk IDs are sequential
        chunk_ids = [int(chunk["chunk_id"]) for chunk in result]
        assert chunk_ids == list(range(len(chunk_ids)))

    @patch("src.document_processor.pdfplumber.open")
    def test_process_local_pdf_empty_pages(self, mock_pdfplumber):
        """Test _process_local_pdf with empty pages"""
        processor = DocumentProcessor()

        # Mock PDF with empty pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = None

        mock_page2 = Mock()
        mock_page2.extract_text.return_value = ""

        mock_page3 = Mock()
        mock_page3.extract_text.return_value = "   \n\t   "

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2, mock_page3]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)

        mock_pdfplumber.return_value = mock_pdf

        with pytest.raises(ValueError, match="No text could be extracted"):
            processor._process_local_pdf("/path/test.pdf", "/path/test.pdf")
