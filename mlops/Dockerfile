FROM python:3.10-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Set HF directories
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/vector_store

# Pre-download models to cache them in the image
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
from transformers import pipeline; \
from src.config import EMBEDDING_MODEL, GEN_MODEL; \
print('Downloading sentence transformer model:', EMBEDDING_MODEL); \
SentenceTransformer(EMBEDDING_MODEL); \
print('Downloading generation model:', GEN_MODEL); \
pipeline('text-generation', model=GEN_MODEL); \
print('Models cached successfully!'); \
"


CMD ["python", "-m", "src.main", "chat"]
