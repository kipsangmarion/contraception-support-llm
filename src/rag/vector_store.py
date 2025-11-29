"""
FAISS vector store for efficient similarity search.
Handles indexing, saving, loading, and retrieval of document embeddings.
"""

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger


class FAISSVectorStore:
    """FAISS-based vector store for document retrieval."""

    def __init__(self, dimension: int, index_type: str = "Flat"):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Dimension of embeddings
            index_type: Type of FAISS index ("Flat", "IVF", "HNSW")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.chunks = []  # Store original chunk data

        self._create_index()

    def _create_index(self):
        """Create FAISS index based on type."""
        logger.info(f"Creating FAISS index (type: {self.index_type}, dimension: {self.dimension})")

        if self.index_type == "Flat":
            # Simple flat index (exact search, good for small datasets)
            self.index = faiss.IndexFlatL2(self.dimension)

        elif self.index_type == "IVF":
            # Inverted file index (faster for large datasets)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)

        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World (fast approximate search)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        logger.info("FAISS index created")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, any]]
    ):
        """
        Add embeddings and corresponding chunks to the index.

        Args:
            embeddings: NumPy array of embeddings (shape: [n, dimension])
            chunks: List of chunk dictionaries
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        logger.info(f"Adding {len(embeddings)} embeddings to index")

        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')

        # Train index if necessary (for IVF)
        if self.index_type == "IVF" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store chunks
        self.chunks.extend(chunks)

        logger.info(f"Total vectors in index: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (L2 distance)

        Returns:
            List of dictionaries containing chunks and scores
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # Ensure query is 2D array and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        # Convert to list of results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            # Skip if index is invalid
            if idx == -1 or idx >= len(self.chunks):
                continue

            # Skip if score is below threshold
            if score_threshold is not None and dist > score_threshold:
                continue

            result = {
                'chunk': self.chunks[idx],
                'score': float(dist),
                'rank': i + 1
            }
            results.append(result)

        logger.debug(f"Found {len(results)} results")
        return results

    def save(self, index_path: str, chunks_path: str):
        """
        Save index and chunks to disk.

        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks (JSON)
        """
        index_path = Path(index_path)
        chunks_path = Path(chunks_path)

        # Create directories
        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, str(index_path))

        # Save chunks (remove embeddings to save space)
        logger.info(f"Saving chunks to {chunks_path}")
        chunks_to_save = []
        for chunk in self.chunks:
            chunk_copy = chunk.copy()
            if 'embedding' in chunk_copy:
                del chunk_copy['embedding']  # Remove embedding to save space
            chunks_to_save.append(chunk_copy)

        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_to_save, f, indent=2)

        logger.info("Index and chunks saved successfully")

    def load(self, index_path: str, chunks_path: str):
        """
        Load index and chunks from disk.

        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to chunks JSON file
        """
        index_path = Path(index_path)
        chunks_path = Path(chunks_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(str(index_path))

        # Load chunks
        logger.info(f"Loading chunks from {chunks_path}")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks")

    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the vector store."""
        return {
            'total_vectors': self.index.ntotal,
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'index_type': self.index_type
        }


def build_faiss_index(
    chunks: List[Dict[str, any]],
    embeddings: np.ndarray,
    dimension: int,
    index_type: str = "Flat"
) -> FAISSVectorStore:
    """
    Build FAISS index from chunks and embeddings.

    Args:
        chunks: List of chunk dictionaries
        embeddings: NumPy array of embeddings
        dimension: Embedding dimension
        index_type: Type of FAISS index

    Returns:
        FAISSVectorStore instance
    """
    logger.info("Building FAISS index")

    vector_store = FAISSVectorStore(dimension=dimension, index_type=index_type)
    vector_store.add_embeddings(embeddings, chunks)

    return vector_store
