import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Setup module logger
logger = logging.getLogger(__name__)


class RAGMCPClient:

    def __init__(
        self,
        server_script_path: str | None = None,
        vectorstore_path: str | None = None,
        pdf_path: str | None = None,
    ):
        self.server_script_path = server_script_path or str(
            Path(__file__).parent / "mcp_server.py"
        )
        self.vectorstore_path = vectorstore_path
        self.pdf_path = pdf_path
        self._session = None
        self._read_stream = None
        self._write_stream = None
        self._stdio_client = None

    async def __aenter__(self):
        # Build server command with configuration
        server_args = ["-m", "src.mcp_server"]
        if self.vectorstore_path:
            server_args.extend(["--vectorstore", self.vectorstore_path])
        elif self.pdf_path:
            server_args.extend(["--pdf", self.pdf_path])

        # Prepare environment variables for the server subprocess
        env = os.environ.copy()
        # Pass debug mode to server via environment variable
        if logger.isEnabledFor(logging.DEBUG):
            env["MCP_DEBUG"] = "true"
        else:
            env["MCP_DEBUG"] = "false"

        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command=sys.executable,
            args=server_args,
            env=env,  # Pass debug environment to subprocess
            cwd=str(
                Path(__file__).parent.parent
            ),  # Set working directory to project root
        )

        logger.info(f"Connecting to MCP server via module: src.mcp_server")
        logger.info(f"Server args: {server_args}")

        try:
            # Establish stdio connection
            self._stdio_client = stdio_client(server_params)
            self._read_stream, self._write_stream = (
                await self._stdio_client.__aenter__()
            )

            # Create and initialize session
            self._session = ClientSession(self._read_stream, self._write_stream)
            await self._session.__aenter__()

            logger.info("MCP session established")

            # Initialize the session
            await self._session.initialize()
            logger.info("MCP protocol initialized")

            # Wait longer for server to be ready, especially for vector store loading
            wait_time = 5 if (self.vectorstore_path or self.pdf_path) else 2
            await asyncio.sleep(wait_time)

            # List available tools
            tools = await self._session.list_tools()
            logger.info(f"Available tools: {[tool.name for tool in tools.tools]}")

            return self

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            # Clean up any partial connections
            await self._cleanup()
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()

    async def _cleanup(self):
        logger.info("Cleaning up MCP client resources...")

        # Close session first
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
                logger.info("MCP session closed")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            finally:
                self._session = None

        # Close stdio client (this should terminate the subprocess)
        if self._stdio_client:
            try:
                await self._stdio_client.__aexit__(None, None, None)
                logger.info("MCP server connection closed")
            except Exception as e:
                logger.warning(f"Error closing stdio client: {e}")
            finally:
                self._stdio_client = None
                self._read_stream = None
                self._write_stream = None

        await asyncio.sleep(0.5)  # Brief delay to ensure cleanup operations complete
        logger.debug("Resource cleanup completed")

    async def search_aviation(self, query: str, top_k: int = 5):
        if not self._session:
            raise RuntimeError("Client not initialized")

        try:
            # Validate inputs
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")

            top_k_validated = max(1, min(10, int(top_k)))

            logger.info(
                f"Calling search_aviation tool with query: '{query}' (top_k={top_k_validated})"
            )

            # Call the tool
            result = await self._session.call_tool(
                name="search_aviation",
                arguments={"query": query.strip(), "top_k": top_k_validated},
            )

            if result.isError:
                logger.error(f"Tool call failed: {result.content}")
                return {"error": True, "message": str(result.content)}

            # Extract text content from result
            response_text = ""
            for content in result.content:
                if isinstance(content, types.TextContent):
                    response_text += content.text

            logger.info(f"Search completed successfully")
            return {
                "error": False,
                "query": query,
                "top_k": top_k_validated,
                "response": response_text,
            }

        except Exception as e:
            logger.error(f"Error calling search_aviation: {e}")
            return {"error": True, "message": str(e)}

    async def health_check(self):
        if not self._session:
            raise RuntimeError("Client not initialized.")

        try:
            logger.info("Performing health check")

            result = await self._session.call_tool(name="health_check", arguments={})

            if result.isError:
                logger.error(f"Health check failed: {result.content}")
                return {"error": True, "message": str(result.content)}

            # Extract text content from result
            response_text = ""
            for content in result.content:
                if isinstance(content, types.TextContent):
                    response_text += content.text

            logger.info(f"Health check completed")
            return {"error": False, "status": response_text}

        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {"error": True, "message": str(e)}

    async def start_chat_session(self, session_id: Optional[str] = None):
        if not self._session:
            raise RuntimeError("Client not initialized.")

        try:
            logger.info("Starting new chat session")

            args = {}
            if session_id:
                args["session_id"] = session_id

            result = await self._session.call_tool(
                name="start_chat_session", arguments=args
            )

            if result.isError:
                logger.error(f"Failed to start chat session: {result.content}")
                return {"error": True, "message": str(result.content)}

            # Extract response
            response_text = ""
            for content in result.content:
                if isinstance(content, types.TextContent):
                    response_text += content.text

            # Extract session ID from response
            session_match = re.search(
                r"session started with ID: ([a-f0-9\-]+)", response_text
            )
            extracted_session_id = (
                session_match.group(1) if session_match else session_id
            )

            logger.info(f"Chat session started: {extracted_session_id}")
            return {
                "error": False,
                "session_id": extracted_session_id,
                "message": response_text,
            }

        except Exception as e:
            logger.error(f"Error starting chat session: {e}")
            return {"error": True, "message": str(e)}

    async def send_chat_message(self, session_id: str, message: str):
        if not self._session:
            raise RuntimeError("Client not initialized.")

        try:
            if not message or not message.strip():
                raise ValueError("Message cannot be empty")

            logger.info(f"Sending chat message to session {session_id}: '{message}'")

            result = await self._session.call_tool(
                name="chat_message",
                arguments={"session_id": session_id, "message": message.strip()},
            )

            if result.isError:
                logger.error(f"Chat message failed: {result.content}")
                return {"error": True, "message": str(result.content)}

            # Extract response
            response_text = ""
            for content in result.content:
                if isinstance(content, types.TextContent):
                    response_text += content.text

            logger.info(f"Chat message processed successfully")
            return {
                "error": False,
                "session_id": session_id,
                "user_message": message,
                "response": response_text,
            }

        except Exception as e:
            logger.error(f"Error sending chat message: {e}")
            return {"error": True, "message": str(e)}

    async def get_chat_history(self, session_id: str):
        if not self._session:
            raise RuntimeError("Client not initialized.")

        try:
            logger.info(f"Getting chat history for session {session_id}")

            result = await self._session.call_tool(
                name="get_chat_history", arguments={"session_id": session_id}
            )

            if result.isError:
                logger.error(f"Failed to get chat history: {result.content}")
                return {"error": True, "message": str(result.content)}

            # Extract response
            response_text = ""
            for content in result.content:
                if isinstance(content, types.TextContent):
                    response_text += content.text

            logger.info(f"Chat history retrieved successfully")
            return {"error": False, "session_id": session_id, "history": response_text}

        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return {"error": True, "message": str(e)}


class RAGMCPChatUI:

    def __init__(self, client: RAGMCPClient):
        self.client = client
        self.session_id = None

    async def run_interactive_chat(self):
        print("✈️ Welcome to the Aviation RAG Chat!")
        print("Ask me anything about aviation, or type 'quit' to exit.")
        print(
            "Type 'health' to check server status, 'history' to view conversation history."
        )
        print("-" * 50)

        try:
            # Start a new chat session
            logger.debug("Starting chat session...")
            if logger.isEnabledFor(logging.DEBUG):
                print("🔧 Starting chat session...")

            result = await self.client.start_chat_session()

            if result["error"]:
                logger.error(f"Failed to start chat session: {result['message']}")
                print(f"❌ Failed to start chat session: {result['message']}")
                return

            self.session_id = result["session_id"]
            logger.debug(f"Chat session started: {self.session_id}")
            if logger.isEnabledFor(logging.DEBUG):
                print(f"✅ Chat session started: {self.session_id}")

            # Main chat loop
            while True:
                try:
                    # Get user input
                    user_input = input("\n🔍 Your question: ").strip()

                    if not user_input:
                        continue

                    if user_input.lower() in ["quit", "exit", "q", "bye"]:
                        print("👋 Goodbye!")
                        break

                    if user_input.lower() == "health":
                        if logger.isEnabledFor(logging.DEBUG):
                            print("🔧 Checking server health...")
                        health_result = await self.client.health_check()

                        if health_result["error"]:
                            logger.error(
                                f"Health check failed: {health_result['message']}"
                            )
                            print(f"❌ Health check failed: {health_result['message']}")
                        else:
                            print(f"✅ Server status: {health_result['status']}")
                        continue

                    if user_input.lower() == "history":
                        if logger.isEnabledFor(logging.DEBUG):
                            print("📜 Getting conversation history...")
                        history_result = await self.client.get_chat_history(
                            self.session_id
                        )

                        if history_result["error"]:
                            logger.error(
                                f"Failed to get history: {history_result['message']}"
                            )
                            print(
                                f"❌ Failed to get history: {history_result['message']}"
                            )
                        else:
                            print(f"\n{history_result['history']}")
                        continue

                    # Send message to chat session
                    logger.debug(f"Sending message: {user_input}")
                    print("🤔 Thinking...")
                    chat_result = await self.client.send_chat_message(
                        self.session_id, user_input
                    )

                    if chat_result["error"]:
                        logger.error(f"Chat error: {chat_result['message']}")
                        print(f"❌ Error: {chat_result['message']}")
                    else:
                        logger.debug("Response received successfully")
                        print("\n📚 Response:")
                        print("-" * 40)
                        print(chat_result["response"])
                        print("-" * 40)

                except KeyboardInterrupt:
                    logger.debug("Chat interrupted by user")
                    print("\n👋 Chat interrupted. Goodbye!")
                    return  # Let the context manager handle cleanup
                except Exception as e:
                    logger.error(f"Error in chat loop: {e}")
                    print(f"❌ An error occurred: {e}")
                    print("Please try again or type 'quit' to exit.")

        except Exception as e:
            logger.error(f"Error in chat session: {e}")
            print(f"❌ Failed to start chat session: {e}")
            # Fallback quit prompt
            try:
                print("Would you like to exit? (y/n)")
                response = input().strip().lower()
                if response in ["y", "yes", ""]:
                    print("👋 Goodbye!")
                    return
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                return
        finally:
            logger.debug("Chat session ending")
            print("✅ Chat session ended. Thank you for chatting!")
