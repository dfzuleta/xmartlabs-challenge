import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ServerCapabilities, TextContent, Tool

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import RAGAgent
from src.config import BASIC_CORPUS
from src.rag import BaseRAGPipeline

logger = logging.getLogger(__name__)
debug_mode = os.getenv("MCP_DEBUG", "").lower() in ("true", "1", "yes")
if debug_mode:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
else:
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    logging.getLogger("mcp.server").setLevel(logging.CRITICAL)
    logging.getLogger("mcp.server.lowlevel").setLevel(logging.CRITICAL)
    logging.getLogger("__main__").setLevel(logging.CRITICAL)
    logging.getLogger("src").setLevel(logging.CRITICAL)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DISABLE_TQDM"] = "true"


class RAGMCPServer:
    """MCP Server for RAG pipeline operations"""

    def __init__(
        self, vectorstore_path: Optional[str] = None, pdf_path: Optional[str] = None
    ):
        self.server = Server("rag-server")
        self.rag_pipeline = self._initialize_rag_pipeline(vectorstore_path, pdf_path)

        if self.rag_pipeline is None:
            raise ValueError("Failed to initialize RAG pipeline")

        self.chat_sessions = {}
        self.current_session_id = None
        self._setup_handlers()

    def _initialize_rag_pipeline(
        self, vectorstore_path: Optional[str], pdf_path: Optional[str]
    ) -> BaseRAGPipeline:
        if vectorstore_path:
            logger.info(f"Loading vector store from {vectorstore_path}")
            try:
                from src.rag import VectorStoreRAGPipeline
                from src.vector_store import VectorStore

                store = VectorStore(store_path=vectorstore_path)
                if store.load():
                    rag_pipeline = VectorStoreRAGPipeline(vector_store=store)
                    logger.info(
                        f"✅ Successfully loaded vector store with {store.get_stats()['num_documents']} documents (LLM enabled)"
                    )
                    return rag_pipeline
                else:
                    logger.warning(
                        "Failed to load vector store, falling back to basic corpus"
                    )
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                logger.info("Falling back to basic corpus")

        elif pdf_path:
            logger.info(f"Loading PDF from {pdf_path}")
            try:
                from src.document_processor import DocumentProcessor
                from src.rag import VectorStoreRAGPipeline
                from src.vector_store import VectorStore

                if not os.path.exists(pdf_path):
                    abs_path = os.path.abspath(pdf_path)
                    if os.path.exists(abs_path):
                        pdf_path = abs_path
                    else:
                        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

                logger.info(f"Processing PDF from path: {pdf_path}")
                processor = DocumentProcessor()
                chunks = processor.process_pdf(pdf_path)

                logger.info(f"Creating vector store from {len(chunks)} chunks...")
                store = VectorStore()
                store.add_documents(chunks)

                rag_pipeline = VectorStoreRAGPipeline(vector_store=store)
                logger.info("PDF successfully processed and indexed (LLM enabled)")
                return rag_pipeline
            except Exception as e:
                logger.error(f"Error processing PDF: {e}")
                logger.info("Falling back to basic corpus")

        logger.info("Using default basic corpus")
        try:
            from src.rag import VectorStoreRAGPipeline
            from src.vector_store import VectorStore

            store = VectorStore()

            basic_chunks = []
            for i, text in enumerate(BASIC_CORPUS):
                basic_chunks.append(
                    {
                        "text": text,
                        "page": "1",
                        "chunk_id": str(i),
                        "source": "basic_corpus",
                    }
                )

            store.add_documents(basic_chunks)
            rag_pipeline = VectorStoreRAGPipeline(vector_store=store)
            logger.info("Successfully initialized with basic corpus")
            return rag_pipeline
        except Exception as e:
            logger.error(f"Error creating basic RAG pipeline: {e}")
            raise ValueError(f"Failed to initialize any RAG pipeline: {e}")

    def _setup_handlers(self):
        self.server.list_tools()(self._handle_list_tools)
        self.server.call_tool()(self._handle_call_tool)

    async def _handle_list_tools(self):
        return [
            Tool(
                name="search_aviation",
                description="Search for aviation and flight information using RAG",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The aviation question to search for",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (1-10)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="start_chat_session",
                description="Start a new interactive chat session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID, if not provided a new one will be generated",
                        },
                    },
                },
            ),
            Tool(
                name="chat_message",
                description="Send a message in an ongoing chat session with context awareness",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "The chat session ID",
                        },
                        "message": {
                            "type": "string",
                            "description": "The user's message/question",
                        },
                    },
                    "required": ["session_id", "message"],
                },
            ),
            Tool(
                name="get_chat_history",
                description="Get the conversation history for a chat session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "The chat session ID",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            Tool(
                name="health_check",
                description="Check the health status of the RAG server",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    async def _handle_call_tool(self, name: str, arguments: Dict[str, Any]):
        try:
            logger.debug(f"Tool called: {name} with args: {arguments}")

            if name == "start_chat_session":
                import uuid

                session_id = arguments.get("session_id") or str(uuid.uuid4())

                agent = RAGAgent(rag_pipeline=self.rag_pipeline)
                self.chat_sessions[session_id] = {
                    "agent": agent,
                    "created_at": asyncio.get_event_loop().time(),
                }
                self.current_session_id = session_id

                logger.debug(f"Started new chat session with RAGAgent: {session_id}")
                return [
                    TextContent(
                        type="text", text=f"Chat session started with ID: {session_id}"
                    )
                ]

            elif name == "chat_message":
                session_id = arguments.get("session_id", "")
                message = arguments.get("message", "").strip()

                if not session_id:
                    return [TextContent(type="text", text="Session ID is required")]

                if not message:
                    return [TextContent(type="text", text="Message cannot be empty")]

                if session_id not in self.chat_sessions:
                    return [
                        TextContent(
                            type="text",
                            text=f"Chat session {session_id} not found. Please start a new session first.",
                        )
                    ]

                session_data = self.chat_sessions[session_id]
                agent = session_data["agent"]

                try:
                    agent.observe(message, "user")
                    response = agent.act()

                    logger.debug(
                        f"Chat response generated using RAGAgent for session {session_id}"
                    )
                    return [TextContent(type="text", text=response)]

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error while processing your question: {str(e)}"
                    logger.error(f"Error generating response with RAGAgent: {e}")
                    return [TextContent(type="text", text=error_msg)]

            elif name == "get_chat_history":
                session_id = arguments.get("session_id", "")

                if not session_id:
                    return [TextContent(type="text", text="Session ID is required")]

                if session_id not in self.chat_sessions:
                    return [
                        TextContent(
                            type="text", text=f"Chat session {session_id} not found"
                        )
                    ]

                session_data = self.chat_sessions[session_id]
                agent = session_data["agent"]

                if not agent.history:
                    return [
                        TextContent(type="text", text="No conversation history yet")
                    ]

                history_text = f"📚 Chat History for Session {session_id}:\n"
                history_text += f"✈️ Agent: Captain Pilot\n\n"

                for i, (content, role) in enumerate(agent.history, 1):
                    role_emoji = "🧑" if role == "user" else "🤖"
                    display_content = (
                        content[:150] + "..." if len(content) > 150 else content
                    )
                    history_text += (
                        f"{i}. {role_emoji} {role.title()}: {display_content}\n\n"
                    )

                return [TextContent(type="text", text=history_text)]

            elif name == "search_aviation":
                query = arguments.get("query", "").strip()
                top_k = arguments.get("top_k", 5)

                if not query:
                    return [TextContent(type="text", text="Query cannot be empty")]

                top_k = max(1, min(10, int(top_k)))

                snippets, sources = self.rag_pipeline.run(query)

                if not snippets or snippets == ["I do not know"]:
                    result_text = (
                        f"No relevant aviation information found for: '{query}'"
                    )
                else:
                    result_text = f"Search results for '{query}':\n\n"
                    for i, snippet in enumerate(snippets[:top_k], 1):
                        result_text += f"{i}. {snippet[:200]}{'...' if len(snippet) > 200 else ''}\n\n"

                logger.info(
                    f"Search completed for query: '{query}' - {len(snippets)} results"
                )
                return [TextContent(type="text", text=result_text)]

            elif name == "generate_answer":
                question = arguments.get("question", "").strip()

                if not question:
                    return [TextContent(type="text", text="Question cannot be empty")]

                try:
                    answer = self.rag_pipeline.generate_answer(question)  # type: ignore
                    logger.debug(f"Answer generated for question: '{question}'")
                    return [TextContent(type="text", text=answer)]

                except Exception as e:
                    logger.error(f"Error generating answer: {e}")
                    return [
                        TextContent(
                            type="text", text=f"Error generating answer: {str(e)}"
                        )
                    ]

            elif name == "health_check":
                try:
                    test_snippets, test_sources = self.rag_pipeline.run("test query")
                    test_successful = test_snippets is not None
                except Exception as e:
                    test_successful = False
                    logger.warning(f"Health check test failed: {e}")

                pipeline_type = type(self.rag_pipeline).__name__
                corpus_size = getattr(self.rag_pipeline, "docs", None)
                corpus_size = len(corpus_size) if corpus_size else "Unknown"

                status = {
                    "status": "healthy" if test_successful else "degraded",
                    "agent_type": "RAGAgent",
                    "rag_pipeline": pipeline_type,
                    "corpus_size": corpus_size,
                    "test_query_successful": test_successful,
                    "active_sessions": len(self.chat_sessions),
                }

                status_text = (
                    "🟢 Aviation RAG MCP Server Health Check (with RAGAgent)\n\n"
                )
                for key, value in status.items():
                    status_text += f"• {key}: {value}\n"

                logger.info("Health check completed successfully")
                return [TextContent(type="text", text=status_text)]

            else:
                logger.warning(f"Unknown tool requested: {name}")
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Tool execution error: {str(e)}")]

    def cleanup(self):
        """Clean up server resources"""
        try:
            for session_data in self.chat_sessions.values():
                agent = session_data.get("agent")
                if agent and hasattr(agent, "_gen"):
                    del agent._gen
                    agent._gen = None
                if agent and hasattr(agent, "_tokenizer"):
                    del agent._tokenizer
                    agent._tokenizer = None

            self.chat_sessions.clear()

            if self.rag_pipeline and hasattr(self.rag_pipeline, "_gen"):
                delattr(self.rag_pipeline, "_gen")
            if self.rag_pipeline and hasattr(self.rag_pipeline, "_tokenizer"):
                delattr(self.rag_pipeline, "_tokenizer")

            import gc

            gc.collect()

            logger.info("Server resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting RAG MCP Server...")

        try:
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="rag-server",
                        server_version="1.0.0",
                        capabilities=ServerCapabilities(),
                    ),
                    raise_exceptions=True,
                )
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
        finally:
            self.cleanup()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG MCP Server")
    parser.add_argument("--vectorstore", help="Path to vector store directory")
    parser.add_argument("--pdf", help="Path to PDF file to process")
    args = parser.parse_args()

    server = RAGMCPServer(vectorstore_path=args.vectorstore, pdf_path=args.pdf)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
