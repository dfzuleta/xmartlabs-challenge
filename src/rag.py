import logging
import re
from abc import ABC, abstractmethod
from typing import List, Tuple

from transformers.pipelines import pipeline as hf_pipeline

from src.config import EARLY_STOPPING, GEN_MODEL, MAX_NEW_TOKENS, NUM_BEAMS, TOP_K

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


class VectorStoreRAGPipeline(BaseRAGPipeline):
    """Vector store-based RAG pipeline using FAISS for semantic search with LLM answer generation"""

    def __init__(self, vector_store, top_k: int = TOP_K):
        super().__init__(docs=[], top_k=top_k)
        self.vector_store = vector_store

        self._gen = hf_pipeline(
            "text-generation",
            model=GEN_MODEL,
            tokenizer=GEN_MODEL,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=EARLY_STOPPING,
        )
        self._tokenizer = self._gen.tokenizer

    def retrieve(self, question: str) -> List[str]:
        if not question.strip():
            logger.debug("Empty question provided, returning fallback")
            return ["I do not know"]

        logger.debug(f"Searching vector store for: {question}")
        texts, scores, _ = self.vector_store.search(question, k=self.top_k)

        if texts:
            logger.debug(f"Found {len(texts)} relevant documents")
            logger.debug(f"Sample {texts[0][:100]}...")
            return texts

        logger.debug("No relevant documents found")
        return ["I do not know"]

    def _build_prompt(self, content: str) -> str:
        if not self._tokenizer:
            raise ValueError("LLM not initialized. Cannot build prompt.")

        system_prompt = (
            "You are an expert assistant. Answer the user's question based on the provided context. "
            "If the context doesn't contain enough information to answer the question, "
            "say 'I don't have enough information to answer this question.'"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not isinstance(prompt, str):
            prompt = str(prompt)
        return prompt

    def _generate_answer(self, prompt: str) -> str:
        if not self._gen:
            raise ValueError("LLM not initialized. Cannot generate answer.")

        try:
            out = self._gen(prompt)
            if isinstance(out, list) and out and "generated_text" in out[0]:
                return out[0]["generated_text"].strip()
            raise ValueError("Unexpected output format from the generation pipeline.")
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_answer(self, question: str, snippets: list) -> str:
        logger.debug(f"Generating answer for question: {question}")
        if snippets and snippets[0] != "I do not know":
            logger.debug("Using direct snippet response")
            return f"Based on the available information:\n\n{snippets[0]}"

        context = "\n\n".join(snippets)
        content = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        logger.debug("Generating LLM response")
        prompt = self._build_prompt(content)
        answer = self._generate_answer(prompt)

        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        logger.debug(f"Generated answer length: {len(answer)}")
        return answer

    def run(self, question: str) -> Tuple[List[str], List[int]]:
        snippets = self.retrieve(question)
        sources = list(range(len(snippets))) if snippets[0] != "I do not know" else []
        answer = self.generate_answer(question, snippets)
        return [answer], sources
