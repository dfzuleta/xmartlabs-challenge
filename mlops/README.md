# MLOps Scripts for GenAI Challenge

Automation scripts for building and running the app

## Quick Start

```bash
#See all commands
make help
```

## Makefile Commands

**Individual Steps:**
- `make build-vector-store` - Process PDFs and create vector store
- `make build-docker` - Build Docker image  
- `make run-chat` - Run interactive chat in container
- `make run-chat DEBUG=1` - Run with debug output

**Development:**
- `make test` - Run unit tests
- `make clean` - Clean build artifacts
- `make docker-clean` - Remove Docker images/containers

**Custom Processing:**
- `make process-pdf PDF=/path/to/file.pdf` - Process specific PDF file

## What Each Step Does

**build-vector-store:**
- Processes PDF documents using the main.py direct URL functionality
- Creates embeddings and builds a FAISS search index
- Falls back to processing any PDFs in the `data/` directory

**build-docker:**
- Builds Docker image using the project's Dockerfile
- Creates a containerized environment with all dependencies

**run-chat:**
- Runs the chat interface in a Docker container
- Mounts vector store and data directories for access
- Supports interactive mode and debug output

## Prerequisites

- Docker installed and running
- Make utility (standard on macOS/Linux)
- Internet connection for downloading dependencies
