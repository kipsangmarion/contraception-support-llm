"""
Simple RAG Retriever for Experiment 3

Uses sentence-transformers embeddings (all-MiniLM-L6-v2) to retrieve
relevant passages from WHO/CDC contraception guidelines.

Based on actual Exp3 implementation that produced the results.
"""

from typing import Dict, List, Any
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger


class SimpleRAGRetriever:
    """
    Simplified RAG retriever matching Experiment 3 implementation.

    Retrieves top-k relevant passages from preprocessed WHO/CDC documents.
    """

    def __init__(
        self,
        documents_path: str = "data/processed/rag_documents.json",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG retriever.

        Args:
            documents_path: Path to preprocessed documents JSON
            embedding_model: Sentence transformer model name
        """
        self.documents_path = Path(documents_path)
        self.embedding_model_name = embedding_model

        # Load sentence transformer
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)

        # Load documents and embeddings
        self.documents = []
        self.embeddings = None

        if self.documents_path.exists():
            self._load_documents()
        else:
            logger.warning(f"Documents not found at {documents_path}")
            logger.warning("RAG retrieval will return empty results")

    def _load_documents(self):
        """Load preprocessed documents and compute embeddings."""
        logger.info(f"Loading documents from {self.documents_path}")

        with open(self.documents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.documents = data.get('documents', [])
        logger.info(f"Loaded {len(self.documents)} document chunks")

        # Compute embeddings if not cached
        if 'embeddings' in data:
            self.embeddings = np.array(data['embeddings'])
            logger.info("Loaded cached embeddings")
        else:
            logger.info("Computing embeddings...")
            texts = [doc['content'] for doc in self.documents]
            self.embeddings = self.encoder.encode(texts, show_progress_bar=True)
            logger.info(f"Computed embeddings: {self.embeddings.shape}")

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User's question
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with retrieved documents and metadata
        """
        if not self.documents or self.embeddings is None:
            logger.warning("No documents loaded, returning empty results")
            return {
                'documents': [],
                'query': query,
                'top_k': top_k
            }

        # Encode query
        query_embedding = self.encoder.encode([query])[0]

        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Retrieve documents
        retrieved_docs = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['score'] = float(similarities[idx])
            retrieved_docs.append(doc)

        logger.debug(f"Retrieved {len(retrieved_docs)} documents with scores: {[d['score'] for d in retrieved_docs]}")

        return {
            'documents': retrieved_docs,
            'query': query,
            'top_k': top_k
        }


# Alias for compatibility with unified pipeline
class RAGRetriever(SimpleRAGRetriever):
    """Alias for SimpleRAGRetriever for compatibility."""
    pass
