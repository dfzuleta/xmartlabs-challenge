import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber

logger = logging.getLogger(__name__)


class DocumentProcessor:

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        # Handle URL downloads
        if pdf_path.startswith(("http://", "https://")):
            logger.info(f"Downloading PDF from URL: {pdf_path}")
            try:
                # Download PDF to temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    urllib.request.urlretrieve(pdf_path, tmp_file.name)
                    local_pdf_path = tmp_file.name

                logger.info(f"Downloaded PDF to: {local_pdf_path}")

                try:
                    result = self._process_local_pdf(local_pdf_path, pdf_path)
                    return result
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(local_pdf_path)
                    except:
                        pass

            except Exception as e:
                logger.error(f"Failed to download PDF from URL: {e}")
                raise FileNotFoundError(f"Could not download PDF from URL: {pdf_path}")
        else:
            # Handle local file
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            return self._process_local_pdf(pdf_path, pdf_path)

    def _process_local_pdf(
        self, local_path: str, original_source: str
    ) -> List[Dict[str, str]]:
        logger.info(f"Processing PDF: {local_path}")
        all_chunks = []
        chunk_id = 0

        try:
            with pdfplumber.open(local_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text from the page
                    text = page.extract_text()

                    if text and text.strip():
                        text = self._clean_text(text)

                        page_chunks = self.chunk_text(text, page_num)

                        for chunk in page_chunks:
                            chunk["chunk_id"] = str(chunk_id)
                            chunk["source"] = (
                                os.path.basename(original_source)
                                if not original_source.startswith("http")
                                else "FAA_Airplane_Handbook.pdf"
                            )
                            all_chunks.append(chunk)
                            chunk_id += 1

        except Exception as e:
            raise RuntimeError(f"Error processing PDF {local_path}: {e}")

        if not all_chunks:
            raise ValueError(f"No text could be extracted from PDF: {local_path}")

        logger.info(f"Successfully extracted {len(all_chunks)} chunks from PDF")
        return all_chunks

    def chunk_text(self, text: str, page_num: int) -> List[Dict[str, str]]:
        if not text or not text.strip():
            return []

        chunks = []
        text = text.strip()

        # Simple character-based chunking with overlap
        start = 0
        local_chunk_id = 0

        while start < len(text):
            # Position
            end = start + self.chunk_size

            # If this isn't the last chunk, try to break at word boundary
            if end < len(text):
                # Look for last space within the chunk to avoid breaking words
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            # Extract chunk
            chunk_text = text[start:end].strip()

            if chunk_text:  # Only add non-empty chunks
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": str(page_num),
                        "local_chunk_id": str(local_chunk_id),
                    }
                )
                local_chunk_id += 1

            # Move start position with overlap
            start = end - self.overlap
            if start <= 0:
                start = end

            if start >= len(text):
                break

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and formatting
        """
        import re

        # Replace multiple whitespaces with single space
        text = re.sub(r"\s+", " ", text)

        # Remove extra newlines but preserve paragraph breaks
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        return {
            "filename": Path(pdf_path).name,
            "path": pdf_path,
            "status": "metadata extraction not implemented",
        }
