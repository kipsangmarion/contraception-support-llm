"""
RAG Retriever Component

Handles document retrieval from vector store.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from src.rag.vector_store import FAISSVectorStore
from src.rag.embeddings import EmbeddingGenerator


class RAGRetriever:
    """
    Retrieves relevant documents from vector store based on query.

    Features:
    - Query embedding generation
    - FAISS-based similarity search
    - Relevance filtering
    - Metadata preservation
    """

    def __init__(
        self,
        vector_store_path: str,
        embeddings_config: Dict,
        relevance_threshold: float = 0.3
    ):
        """
        Initialize retriever.

        Args:
            vector_store_path: Path to saved FAISS index
            embeddings_config: Configuration for embedding generator
            relevance_threshold: Minimum similarity score (0-1)
        """
        self.vector_store_path = Path(vector_store_path)
        self.relevance_threshold = relevance_threshold

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            model_name=embeddings_config.get('model_name', 'all-MiniLM-L6-v2'),
            provider=embeddings_config.get('provider', 'sentence-transformers')
        )

        # Load vector store
        self.vector_store = self._load_vector_store()

        logger.info(f"RAGRetriever initialized with {len(self.vector_store.chunks)} chunks")

    def _load_vector_store(self) -> FAISSVectorStore:
        """
        Load FAISS vector store from disk.

        Returns:
            Loaded FAISSVectorStore instance
        """
        if not self.vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.vector_store_path}. "
                "Please run preprocess_documents.py first."
            )

        # Get embedding dimension from generator
        dimension = self.embedding_generator.dimension

        # Create vector store and load
        vector_store = FAISSVectorStore(dimension=dimension)
        vector_store.load(str(self.vector_store_path))

        logger.info(f"Loaded vector store from {self.vector_store_path}")
        return vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_threshold: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User's question
            top_k: Number of documents to retrieve
            filter_threshold: Whether to filter by relevance threshold

        Returns:
            List of documents with text, metadata, and scores
        """
        logger.debug(f"Retrieving documents for query: {query[:100]}...")

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # Filter by relevance threshold
        if filter_threshold:
            results = [
                doc for doc in results
                if doc['score'] >= self.relevance_threshold
            ]

        logger.debug(f"Retrieved {len(results)} documents above threshold {self.relevance_threshold}")

        return results

    def get_context(
        self,
        query: str,
        top_k: int = 5,
        max_length: Optional[int] = None
    ) -> str:
        """
        Get concatenated context from retrieved documents.

        Args:
            query: User's question
            top_k: Number of documents to retrieve
            max_length: Maximum context length (characters)

        Returns:
            Concatenated context string
        """
        # Retrieve documents
        documents = self.retrieve(query=query, top_k=top_k)

        if not documents:
            logger.warning("No relevant documents found above threshold")
            return ""

        # Build context with citations
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Extract metadata
            source = doc['metadata'].get('source', 'Unknown')
            page = doc['metadata'].get('page_number', 'N/A')

            # Format context entry
            citation = f"[Source {i}: {source}, Page {page}]"
            text = doc['text']

            context_parts.append(f"{citation}\n{text}\n")

        # Concatenate context
        context = "\n".join(context_parts)

        # Truncate if needed
        if max_length and len(context) > max_length:
            context = context[:max_length] + "\n[... context truncated ...]"
            logger.debug(f"Context truncated to {max_length} characters")

        logger.debug(f"Built context with {len(documents)} documents, {len(context)} chars")

        return context

    def get_sources(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Get source citations for a query.

        Args:
            query: User's question
            top_k: Number of documents to retrieve

        Returns:
            List of source metadata
        """
        documents = self.retrieve(query=query, top_k=top_k)

        # Extract unique sources
        sources = []
        seen_sources = set()

        for doc in documents:
            source = doc['metadata'].get('source', 'Unknown')
            page = doc['metadata'].get('page_number', 'N/A')
            source_key = f"{source}:{page}"

            if source_key not in seen_sources:
                sources.append({
                    'source': source,
                    'page': page,
                    'relevance_score': doc['score']
                })
                seen_sources.add(source_key)

        return sources


class HybridRetriever(RAGRetriever):
    """
    Advanced retriever with keyword + semantic search.

    Combines:
    - Dense vector search (semantic)
    - Keyword matching (lexical)
    """

    def __init__(
        self,
        vector_store_path: str,
        embeddings_config: Dict,
        relevance_threshold: float = 0.3,
        keyword_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store_path: Path to saved FAISS index
            embeddings_config: Configuration for embedding generator
            relevance_threshold: Minimum similarity score
            keyword_weight: Weight for keyword matching (0-1)
        """
        super().__init__(vector_store_path, embeddings_config, relevance_threshold)
        self.keyword_weight = keyword_weight
        self.semantic_weight = 1.0 - keyword_weight

    def _keyword_score(self, query: str, text: str) -> float:
        """
        Calculate keyword-based relevance score.

        Args:
            query: User's question
            text: Document text

        Returns:
            Keyword match score (0-1)
        """
        # Simple keyword matching (can be enhanced with TF-IDF)
        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())

        if not query_terms:
            return 0.0

        # Jaccard similarity
        intersection = query_terms & text_terms
        union = query_terms | text_terms

        return len(intersection) / len(union) if union else 0.0

    def retrieve(
        self,
        query: str,
        top_k: int = 10,  # Retrieve more for re-ranking
        filter_threshold: bool = True
    ) -> List[Dict]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: User's question
            top_k: Number of documents to retrieve initially
            filter_threshold: Whether to filter by threshold

        Returns:
            Re-ranked documents
        """
        # Get semantic results
        semantic_results = super().retrieve(
            query=query,
            top_k=top_k,
            filter_threshold=False  # Don't filter yet
        )

        # Calculate hybrid scores
        for doc in semantic_results:
            semantic_score = doc['score']
            keyword_score = self._keyword_score(query, doc['text'])

            # Combine scores
            doc['semantic_score'] = semantic_score
            doc['keyword_score'] = keyword_score
            doc['score'] = (
                self.semantic_weight * semantic_score +
                self.keyword_weight * keyword_score
            )

        # Re-rank by hybrid score
        semantic_results.sort(key=lambda x: x['score'], reverse=True)

        # Filter by threshold
        if filter_threshold:
            semantic_results = [
                doc for doc in semantic_results
                if doc['score'] >= self.relevance_threshold
            ]

        # Return top-k after re-ranking
        return semantic_results[:top_k // 2]  # Return fewer after re-ranking


# Example usage
if __name__ == "__main__":
    from src.utils.logger import setup_logger

    # Setup logging
    logger = setup_logger()

    # Example configuration
    embeddings_config = {
        'model_name': 'all-MiniLM-L6-v2',
        'provider': 'sentence-transformers'
    }

    # Initialize retriever
    retriever = RAGRetriever(
        vector_store_path="data/processed/vector_store",
        embeddings_config=embeddings_config,
        relevance_threshold=0.3
    )

    # Example query
    query = "What are the side effects of DMPA injection?"

    # Retrieve documents
    documents = retriever.retrieve(query, top_k=3)

    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(documents)} documents:\n")

    for i, doc in enumerate(documents, 1):
        print(f"{i}. Score: {doc['score']:.3f}")
        print(f"   Source: {doc['metadata'].get('source', 'Unknown')}")
        print(f"   Text: {doc['text'][:200]}...")
        print()

    # Get formatted context
    context = retriever.get_context(query, top_k=3)
    print("\nFormatted Context:")
    print(context[:500] + "..." if len(context) > 500 else context)
