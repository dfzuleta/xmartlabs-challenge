import logging
from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.rag import BaseRAGPipeline
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    question: str


@dataclass
class EvaluationResult:
    question: str
    answer: str
    chunks: List[str]
    retrieval_score: float
    relevance_score: float
    faithfulness_score: float


class RAGEvaluator:

    DATASET = [
        EvaluationSample("What is a lazy 8 maneuver?"),
        EvaluationSample("How do you perform a lazy 8 maneuver?"),
        EvaluationSample("What is a steep turn?"),
        EvaluationSample("How do you perform a steep turn?"),
        EvaluationSample("What is a steep spiral?"),
        EvaluationSample("How do you perform a steep spiral?"),
        EvaluationSample("How do you determine aircraft attitude in attitude flying?"),
        EvaluationSample("What is a common mistake beginner pilots make when maintaining straight and level flight, and why is it problematic?"),
        EvaluationSample("Why is the coordinated use of ailerons and rudder pedals important in straight and level flight, and what are the consequences of using them independently?"),
        EvaluationSample("At what age did George Washington die?"),
        EvaluationSample("What is the chemical composition of water?"),
    ]

    RETRIEVAL_THRESHOLD = 0.5
    RELEVANCE_THRESHOLD = 0.5
    FAITHFULNESS_THRESHOLD = 0.6

    def __init__(self, vector_store: VectorStore, rag_pipeline: BaseRAGPipeline):
        from src.agents import RAGAgent
        self.vector_store = vector_store
        self.rag_pipeline = rag_pipeline
        self._agent = RAGAgent(rag_pipeline=rag_pipeline)
        self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self._nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

    def _get_answer(self, question: str) -> str:
        """Generate answer using RAGAgent — same path the real user sees."""
        self._agent.history = []
        self._agent.observe(question, "user")
        response = self._agent.act()
        # strip the sources footer if present
        if "\n\nSources:" in response:
            response = response.split("\n\nSources:")[0]
        return response.strip()

    def _compute_semantic_similarity(self, query: str, texts: List[str], use_avg: bool = False) -> float:
        """Cosine similarity between query and a list of texts.
        use_avg=True penalizes noisy chunks (retrieval). use_avg=False checks if at least one chunk matches (faithfulness fallback)."""
        if not texts:
            return 0.0

        query_emb = self._embedding_model.encode([query], convert_to_numpy=True)
        texts_emb = self._embedding_model.encode(texts, convert_to_numpy=True)

        faiss.normalize_L2(query_emb)
        faiss.normalize_L2(texts_emb)

        similarities = np.dot(texts_emb, query_emb.T).flatten()
        return float(np.mean(similarities)) if use_avg else float(np.max(similarities))

    def _compute_nli_score(self, premise: str, hypothesis: str) -> float:
        """NLI entailment probability — how much does the hypothesis follow from the premise."""
        import torch
        logits = self._nli_model.predict([(premise, hypothesis)])
        # cross-encoder/nli-deberta-v3-small returns logits [contradiction, neutral, entailment]
        probs = torch.softmax(torch.tensor(logits[0]), dim=0)
        return float(probs[2])

    def _llm_judge(self, answer: str, chunks: List[str]) -> float:
        """Ask Falcon if the answer is faithful to the chunks using chat template. Returns 1.0 (YES) or 0.0 (NO)."""
        gen = getattr(self._agent, "_gen", None)
        if gen is None:
            return self._compute_semantic_similarity(answer, chunks)

        tokenizer = gen.tokenizer
        context = "\n\n".join(chunks[:3])
        messages = [
            {"role": "system", "content": "You are a fact-checking assistant. Reply with exactly one word: YES if the answer is supported by the context, NO if it is not."},
            {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}\n\nIs this answer faithful to the context? Reply YES or NO."},
        ]

        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not isinstance(prompt, str):
                prompt = str(prompt)
            out = gen(prompt, max_new_tokens=10)
            if isinstance(out, list) and out and "generated_text" in out[0]:
                generated = out[0]["generated_text"][len(prompt):].upper().strip()
                if generated.startswith("YES"):
                    return 1.0
                if generated.startswith("NO"):
                    return 0.0
        except Exception as e:
            logger.error(f"LLM judge failed: {e}")

        logger.warning("LLM judge returned ambiguous result, falling back to semantic similarity")
        return self._compute_semantic_similarity(answer, chunks)

    def evaluate_retrieval(self, question: str, chunks: List[str]) -> float:
        """Avg cosine similarity between the question and all retrieved chunks."""
        return self._compute_semantic_similarity(question, chunks, use_avg=True)

    def evaluate_relevance(self, question: str, answer: str) -> float:
        """NLI entailment: does the answer address what the question asked?"""
        return self._compute_nli_score(premise=question, hypothesis=answer)

    def evaluate_faithfulness(self, answer: str, chunks: List[str]) -> float:
        """LLM-as-judge: does the answer stay within what the chunks say?"""
        return self._llm_judge(answer, chunks)

    def run(self, samples: Optional[List[EvaluationSample]] = None) -> List[EvaluationResult]:
        from src.rag import RELEVANCE_THRESHOLD as PIPELINE_THRESHOLD
        samples = samples or self.DATASET
        results = []

        for sample in samples:
            print(f"  Evaluating: {sample.question[:60]}...")

            chunks, _, _ = self.vector_store.search(sample.question)
            retrieval_score = self.evaluate_retrieval(sample.question, chunks)

            if retrieval_score < PIPELINE_THRESHOLD:
                results.append(EvaluationResult(
                    question=sample.question,
                    answer="[No relevant documents found]",
                    chunks=chunks,
                    retrieval_score=retrieval_score,
                    relevance_score=0.0,
                    faithfulness_score=0.0,
                ))
                continue

            answer = self._get_answer(sample.question)
            results.append(
                EvaluationResult(
                    question=sample.question,
                    answer=answer,
                    chunks=chunks,
                    retrieval_score=retrieval_score,
                    relevance_score=self.evaluate_relevance(sample.question, answer),
                    faithfulness_score=self.evaluate_faithfulness(answer, chunks),
                )
            )

        return results

    def report(self, results: List[EvaluationResult]) -> None:
        print("\n" + "=" * 60)
        print("RAG EVALUATION REPORT")
        print("=" * 60)

        for r in results:
            ret_ok = r.retrieval_score >= self.RETRIEVAL_THRESHOLD
            rel_ok = r.relevance_score >= self.RELEVANCE_THRESHOLD
            fai_ok = r.faithfulness_score >= self.FAITHFULNESS_THRESHOLD

            print(f"\nQ: {r.question}")
            print(f"  Retrieval    {'[OK]' if ret_ok else '[FAIL]'}  {r.retrieval_score:.3f}  (semantic sim: question<->chunks)")
            print(f"  Relevance    {'[OK]' if rel_ok else '[FAIL]'}  {r.relevance_score:.3f}  (NLI: question->answer)")
            print(f"  Faithfulness {'[OK]' if fai_ok else '[FAIL]'}  {r.faithfulness_score:.3f}  (LLM judge: answer<->chunks)")

        print("\n" + "-" * 60)
        avg_ret = float(np.mean([r.retrieval_score for r in results]))
        avg_rel = float(np.mean([r.relevance_score for r in results]))
        avg_fai = float(np.mean([r.faithfulness_score for r in results]))
        print(f"  Avg Retrieval:     {avg_ret:.3f}")
        print(f"  Avg Relevance:     {avg_rel:.3f}")
        print(f"  Avg Faithfulness:  {avg_fai:.3f}")
        print("=" * 60 + "\n")
