#!/bin/bash

set -e

# Default values
IMAGE_NAME="genai-challenge"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building Docker image: $FULL_IMAGE_NAME"

# Change to project root directory
cd "$(dirname "$0")/.."

# Build Docker image
docker build -t "$FULL_IMAGE_NAME" -f mlops/Dockerfile .
echo "Docker image built successfully: $FULL_IMAGE_NAME"

# Test the image
echo "Testing Docker image..."
docker run --rm "$FULL_IMAGE_NAME" python -c "import src.main; print('Image test successful')"

echo "Build complete!"
