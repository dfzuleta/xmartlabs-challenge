import logging
from typing import List, Tuple, Literal

from transformers.pipelines import pipeline as hf_pipeline

from src.config import EARLY_STOPPING, GEN_MODEL, MAX_NEW_TOKENS, NUM_BEAMS
from src.rag import BaseRAGPipeline

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base agent"""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.history: List[Tuple[str, str]] = []
        self._gen = hf_pipeline(
            "text-generation",
            model=GEN_MODEL,
            tokenizer=GEN_MODEL,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=EARLY_STOPPING,
        )
        self._tokenizer = self._gen.tokenizer

    def observe(self, message: str, role: Literal["user", "assistant"] = "user") -> None:
        self.history.append((message, role))

    def _build_prompt(self, content: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not isinstance(prompt, str):
            prompt = str(prompt)
        return prompt

    def _generate(self, prompt: str) -> str:
        try:
            logger.debug(f"Generating response for prompt length: {len(prompt)}")
            out = self._gen(prompt)
            if isinstance(out, list) and out and "generated_text" in out[0]:
                response = out[0]["generated_text"].strip()
                logger.debug(f"Generated response length: {len(response)}")
                return response
            raise ValueError("Unexpected output format from the generation pipeline.")
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"


class RAGAgent(BaseAgent):
    SYSTEM_PROMPT = (
        "You are an advanced AI aviation expert with expertise in all areas of flight, aircraft systems, and aviation regulations. "
        "You engage in thoughtful conversations about aviation, using chain-of-thought reasoning. "
        "For each question, first analyze the provided context, think step by step, then provide a clear answer. "
        "Consider previous conversation context when relevant. Be accurate, educational, and engaging."
    )

    CONTEXT_TEMPLATE = (
        "Previous conversation context:\n{chat_history}\n\n"
        "Current question: {question}\n\n"
        "Relevant context from documents:\n{snippets}\n\n"
        "Answer:"
    )

    def __init__(self, rag_pipeline: BaseRAGPipeline):
        super().__init__(self.SYSTEM_PROMPT)
        self.rag = rag_pipeline

    def _format_chat_history(self) -> str:
        if not self.history:
            return "No previous conversation."

        recent_history = self.history[-3:]
        formatted = []
        for i, (m, o) in enumerate(recent_history, 1):
            msg_clean = m.replace("<|assistant|>", "").replace("<|user|>", "").strip()
            formatted.append(f"<|{o}|> {msg_clean}")

        return "\n".join(formatted)

    def act(self) -> str:
        if not self.history:
            raise ValueError("No query in history to answer.")

        question = self.history[-1][0]
        logger.debug(f"Processing question: {question}")

        content, sources = self.rag.run(question)
        content_text = (
            "\n\n".join(content) if content else "No specific context available."
        )
        logger.debug(f"Retrieved {len(sources)} sources for context")

        chat_context = self._format_chat_history()

        context_content = self.CONTEXT_TEMPLATE.format(
            chat_history=chat_context, question=question, snippets=content_text
        )
        cot_prompt = self._build_prompt(context_content)
        final_answer = self._generate(cot_prompt)
        final_answer = self.pretty_print_answer(final_answer)

        self.observe(final_answer, "assistant")
        logger.debug(
            f"Added response to chat history, total exchanges: {len(self.history)}"
        )

        if sources and content_text != "No specific context available.":
            source_text = (
                f"\n\n📚 Sources: Based on {len(sources)} relevant document sections"
            )
            return final_answer + source_text

        return final_answer

    def pretty_print_answer(self, answer: str) -> str:
        if "<|assistant|>" in answer:
            return answer.split("<|assistant|>")[-1].strip()
        elif "Final answer" in answer:
            return answer.split("Final answer")[-1].strip()
        elif "Refined answer" in answer:
            return answer.split("Refined answer")[-1].strip()
        else:
            return answer
