import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from src.config import PDF_URL

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    if debug:
        level = logging.DEBUG
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Configure root logger for debug mode
        logging.basicConfig(
            level=level, format=format_str, datefmt="%Y-%m-%d %H:%M:%S", force=True
        )

        # Enable full logging in debug mode
        logging.getLogger("src").setLevel(logging.DEBUG)
        logging.getLogger("__main__").setLevel(logging.DEBUG)
        logging.getLogger("mcp").setLevel(logging.INFO)

    else:
        # NORMAL MODE: Only show user-facing messages, suppress all internal logs
        level = logging.CRITICAL  # Only show critical errors
        format_str = "%(message)s"

        # Configure root logger to suppress almost everything
        logging.basicConfig(level=level, format=format_str, force=True)

        # Suppress all internal logging in normal mode
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger("transformers").setLevel(logging.CRITICAL)
        logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)
        logging.getLogger("faiss").setLevel(logging.CRITICAL)
        logging.getLogger("torch").setLevel(logging.CRITICAL)
        logging.getLogger("mcp").setLevel(logging.CRITICAL)
        logging.getLogger("mcp.server").setLevel(logging.CRITICAL)
        logging.getLogger("mcp.server.lowlevel").setLevel(logging.CRITICAL)
        logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.CRITICAL)
        logging.getLogger("src").setLevel(logging.CRITICAL)
        logging.getLogger("__main__").setLevel(logging.CRITICAL)

        # Disable progress bars in normal mode
        import os

        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Disable tqdm progress bars
        os.environ["DISABLE_TQDM"] = "true"


def run_evaluation():
    try:
        from src.evaluation import RAGEvaluator

        logger.info("TODO: Implement evaluation suite")
    except ImportError:
        logger.info("TODO: Implement evaluation suite")


def process_pdf(pdf_path: str, vectorstore_path: str = "vector_store"):
    try:
        from src.document_processor import DocumentProcessor
        from src.vector_store import VectorStore

        logger.info("Processing PDF document...")
        processor = DocumentProcessor()
        chunks = processor.process_pdf(pdf_path)

        logger.info(f"Extracted {len(chunks)} chunks from PDF")

        logger.info("Building vector store...")
        store = VectorStore(store_path=vectorstore_path)
        store.add_documents(chunks)
        store.save()

        logger.info("Vector store created successfully!")
        logger.info(f"Stats: {store.get_stats()}")

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise e


def run_interactive_chat(vectorstore_path: Optional[str] = None):

    async def chat_main():
        try:
            from src.mcp_client import RAGMCPChatUI, RAGMCPClient

            client = RAGMCPClient(vectorstore_path=vectorstore_path)

            async with client:
                chat_ui = RAGMCPChatUI(client)
                await chat_ui.run_interactive_chat()
        except KeyboardInterrupt:
            logger.debug("Chat interrupted by user")
        except Exception as e:
            logger.error(f"Error in chat application: {e}")
            raise e

    asyncio.run(chat_main())


def main():
    parser = argparse.ArgumentParser(description="GenAI Challenge CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat", help="Start interactive chat with AdvancedRAGAgent"
    )
    chat_parser.add_argument(
        "--vectorstore",
        dest="vectorstore_path",
        help="Path to vector store directory",
        default="vector_store",
    )

    # Evaluation command
    subparsers.add_parser("eval", help="Run the evaluation suite")

    # PDF processing commands
    pdf_parser = subparsers.add_parser(
        "process-pdf", help="Process PDF and create vector store"
    )
    pdf_parser.add_argument(
        "pdf_path",
        nargs="?",
        default=PDF_URL,
        help="Path to the PDF file to process (default: value from config)",
    )
    pdf_parser.add_argument(
        "-o",
        "--output",
        dest="vectorstore_path",
        default="vector_store",
        help="Output directory for the vector store (default: vector_store)",
    )

    args = parser.parse_args()

    # Setup logging based on debug flag
    setup_logging(debug=args.debug)

    if args.debug:
        logger.info("Debug mode enabled")

    if args.command == "chat":
        run_interactive_chat(getattr(args, "vectorstore_path", None))
    elif args.command == "eval":
        run_evaluation()
    elif args.command == "process-pdf":
        process_pdf(
            pdf_path=getattr(args, "pdf_path"),
            vectorstore_path=getattr(args, "vectorstore_path"),
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
