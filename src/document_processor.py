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
        import re
        if not text or not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_parts: List[str] = []
        current_size = 0
        local_chunk_id = 0

        for para in paragraphs:
            if current_size + len(para) > self.chunk_size and current_parts:
                chunks.append({
                    "text": "\n\n".join(current_parts),
                    "page": str(page_num),
                    "local_chunk_id": str(local_chunk_id),
                })
                local_chunk_id += 1
                # overlap: carry last paragraph into next chunk
                current_parts = [current_parts[-1]]
                current_size = len(current_parts[0])

            if len(para) > self.chunk_size:
                # paragraph too long — split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    if current_size + len(sentence) > self.chunk_size and current_parts:
                        chunks.append({
                            "text": "\n\n".join(current_parts),
                            "page": str(page_num),
                            "local_chunk_id": str(local_chunk_id),
                        })
                        local_chunk_id += 1
                        current_parts = []
                        current_size = 0
                    current_parts.append(sentence)
                    current_size += len(sentence)
            else:
                current_parts.append(para)
                current_size += len(para)

        if current_parts:
            chunks.append({
                "text": "\n\n".join(current_parts),
                "page": str(page_num),
                "local_chunk_id": str(local_chunk_id),
            })

        return chunks

    def _clean_text(self, text: str) -> str:
        import re

        # normalize multiple blank lines to a single paragraph separator
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        # within each paragraph collapse line-wrapping (single newlines) to space
        paragraphs = text.split("\n\n")
        cleaned = []
        for para in paragraphs:
            para = re.sub(r"\n", " ", para)
            para = re.sub(r" +", " ", para).strip()
            if para:
                cleaned.append(para)

        return "\n\n".join(cleaned)

    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        return {
            "filename": Path(pdf_path).name,
            "path": pdf_path,
            "status": "metadata extraction not implemented",
        }
