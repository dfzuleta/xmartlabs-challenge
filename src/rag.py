import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

from src.config import TOP_K

logger = logging.getLogger(__name__)


class BaseRAGPipeline(ABC):
    """Abstract base class for RAG pipelines."""

    def __init__(self, docs: List[str], top_k: int = TOP_K):
        self.docs = docs
        self.top_k = top_k

    @abstractmethod
    def retrieve(self, question: str) -> List[str]:
        """Retrieve relevant context for the question."""
        pass

    @abstractmethod
    def run(self, question: str) -> Tuple[List[str], List[int]]:
        """Run the retrieval, rank and return (snippets, source_indices)."""
        pass


NO_INFO = "I don't have information about that topic in the available documents."
RELEVANCE_THRESHOLD = 0.4


class VectorStoreRAGPipeline(BaseRAGPipeline):
    """Vector store-based RAG pipeline using FAISS for semantic search."""

    def __init__(self, vector_store, top_k: int = TOP_K):
        super().__init__(docs=[], top_k=top_k)
        self.vector_store = vector_store

    def retrieve(self, question: str) -> List[str]:
        if not question.strip():
            return [NO_INFO]

        texts, scores, _ = self.vector_store.search(question, k=self.top_k)

        if texts and max(scores) >= RELEVANCE_THRESHOLD:
            logger.debug(f"Found {len(texts)} relevant documents (max score: {max(scores):.3f})")
            return texts

        max_score = max(scores) if scores else 0
        logger.debug(f"No relevant documents above threshold (max score: {max_score:.3f})")
        return [NO_INFO]

    def generate_answer(self, snippets: list) -> str:
        if not snippets or snippets[0] == NO_INFO:
            return NO_INFO
        context = "\n\n".join(snippets)
        return f"Based on the available information:\n\n{context}"

    def run(self, question: str) -> Tuple[List[str], List[int]]:
        snippets = self.retrieve(question)
        sources = list(range(len(snippets))) if snippets[0] != NO_INFO else []
        answer = self.generate_answer(snippets)
        return [answer], sources
