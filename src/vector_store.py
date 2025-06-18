import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for document embeddings"""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        store_path: str = "vector_store",
    ):
        self.embedding_model_name = embedding_model
        self.store_path = store_path

        self.model = SentenceTransformer(embedding_model)

        self.index = None
        self.documents = []
        self.metadata = []
        self.embeddings = None

    def add_documents(self, chunks: List[Dict[str, str]]):
        if not chunks:
            raise ValueError("No chunks provided to add to vector store")

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        new_embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )

        if self.index is None:
            dimension = new_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(
                dimension
            )

            faiss.normalize_L2(new_embeddings)
            self.index.add(new_embeddings.astype(np.float32))  # type: ignore

            self.embeddings = new_embeddings
            self.documents = texts
            self.metadata = chunks
        else:
            faiss.normalize_L2(new_embeddings)
            self.index.add(new_embeddings.astype(np.float32))  # type: ignore

            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            else:
                self.embeddings = new_embeddings
            self.documents.extend(texts)
            self.metadata.extend(chunks)

        logger.info(f"Added {len(chunks)} chunks to vector store")

    def search(
        self, query: str, k: int = 5, score_threshold: float = 0.0
    ) -> Tuple[List[str], List[float], List[Dict]]:
        if not query.strip():
            return [], [], []

        if self.index is None or len(self.documents) == 0:
            return [], [], []

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        search_k = min(k, len(self.documents))
        scores, indices = self.index.search(
            query_embedding.astype(np.float32), search_k
        )  # type: ignore

        results_texts = []
        results_scores = []
        results_metadata = []

        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score >= score_threshold:
                results_texts.append(self.documents[idx])
                results_scores.append(float(score))
                results_metadata.append(self.metadata[idx])

        return results_texts, results_scores, results_metadata

    def save(self):
        try:
            os.makedirs(self.store_path, exist_ok=True)

            # Save FAISS index
            if self.index is not None:
                faiss.write_index(
                    self.index, os.path.join(self.store_path, "faiss.index")
                )

            # Save metadata and documents
            with open(os.path.join(self.store_path, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)

            with open(os.path.join(self.store_path, "metadata.pkl"), "wb") as f:
                pickle.dump(self.metadata, f)

            # Save embeddings
            if self.embeddings is not None:
                np.save(
                    os.path.join(self.store_path, "embeddings.npy"), self.embeddings
                )

            # Save config
            config = {
                "embedding_model": self.embedding_model_name,
                "num_documents": len(self.documents),
            }
            with open(os.path.join(self.store_path, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"Vector store saved to {self.store_path}")

        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

    def load(self) -> bool:
        try:
            if not os.path.exists(self.store_path):
                return False

            # Load config
            config_path = os.path.join(self.store_path, "config.json")
            if not os.path.exists(config_path):
                return False

            with open(config_path, "r") as f:
                config = json.load(f)

            # Load FAISS index
            index_path = os.path.join(self.store_path, "faiss.index")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)

            # Load documents and metadata
            docs_path = os.path.join(self.store_path, "documents.pkl")
            meta_path = os.path.join(self.store_path, "metadata.pkl")

            if os.path.exists(docs_path) and os.path.exists(meta_path):
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)

            # Load embeddings
            emb_path = os.path.join(self.store_path, "embeddings.npy")
            if os.path.exists(emb_path):
                self.embeddings = np.load(emb_path)

            logger.info(f"Vector store loaded from {self.store_path}")
            logger.info(f"Loaded {len(self.documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        return {
            "num_documents": len(self.documents),
            "num_chunks": len(self.metadata),
            "index_size": self.index.ntotal if self.index else 0,
        }
