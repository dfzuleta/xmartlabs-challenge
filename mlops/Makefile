# Configuration
PROJECT_NAME := genai-challenge
IMAGE_NAME := $(PROJECT_NAME)
IMAGE_TAG := latest
FULL_IMAGE_NAME := $(IMAGE_NAME):$(IMAGE_TAG)
CONTAINER_NAME := $(PROJECT_NAME)-chat
VECTOR_STORE_PATH := ../vector_store
DATA_PATH := ../data
PROJECT_ROOT := ..

# Default target
.DEFAULT_GOAL := help

# Phony targets
.PHONY: help setup clean build-vector-store build-docker run-chat test lint format check-docker install-deps all pipeline

## Help target
help:
	@echo ""
	@echo "  make build-vector-store    # Build vector store only"
	@echo "  make build-docker          # Build Docker image only"
	@echo "  make run-chat             # Run interactive chat"
	@echo "  make run-chat DEBUG=1     # Run with debug mode"
	@echo ""


install-deps:
	pip install -r $(PROJECT_ROOT)/requirements.txt

build-vector-store: ## Build vector store from PDFs using direct URL
	@echo "Building vector store..."
	chmod +x ./build_vector_store.sh;
	./build_vector_store.sh;
	@echo "Vector store built successfully"

build-docker: ## Build Docker image
	@echo "Building Docker image..."
	chmod +x ./build_docker.sh;
	./build_docker.sh --name $(IMAGE_NAME) --tag $(IMAGE_TAG);
	@echo "Docker image built successfully"


run-chat: ## Run interactive chat in Docker container
	@if [ "$(DEBUG)" = "1" ]; then \
		DEBUG_FLAG="--debug"; \
	else \
		DEBUG_FLAG=""; \
	fi; \
	docker run -it --rm \
		--name $(CONTAINER_NAME) \
		$(FULL_IMAGE_NAME) \
		python -m src.main $$DEBUG_FLAG chat; \

## Development and Testing
test: ## Run tests
	cd $(PROJECT_ROOT) && python -m pytest tests/ -v

docker-clean: ## Clean Docker images and containers
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
	-docker rmi $(FULL_IMAGE_NAME) 2>/dev/null || true

docker-logs: ## Show Docker container logs
	docker logs $(CONTAINER_NAME)

## Utility Targets
clean: ## Clean build artifacts and cache
	cd $(PROJECT_ROOT) && find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	cd $(PROJECT_ROOT) && find . -type f -name "*.pyc" -delete 2>/dev/null || true
	cd $(PROJECT_ROOT) && find . -type f -name "*.pyo" -delete 2>/dev/null || true
	cd $(PROJECT_ROOT) && find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	cd $(PROJECT_ROOT) && rm -rf .pytest_cache/ .coverage .mypy_cache/ dist/ build/

## Custom PDF Processing
process-pdf: ## Process a specific PDF file (usage: make process-pdf PDF=/path/to/file.pdf)
	@if [ -z "$(PDF)" ]; then \
		echo "PDF path required. Usage: make process-pdf PDF=/path/to/file.pdf"; \
		exit 1; \
	fi
	cd $(PROJECT_ROOT) && python -m src.main process-pdf "$(PDF)" -o vector_store

## Pipeline
pipeline: build-vector-store build-docker run-chat ## Run complete MLOps pipeline
