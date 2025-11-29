"""
Embedding generation module for text chunks.
Supports both sentence-transformers and OpenAI embeddings.
"""

import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger
import os


class EmbeddingGenerator:
    """Generates embeddings for text chunks."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        provider: str = "sentence-transformers",
        batch_size: int = 32
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: Name of the embedding model
            provider: "sentence-transformers" or "openai"
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.provider = provider
        self.batch_size = batch_size
        self.model = None
        self.dimension = None

        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        logger.info(f"Loading embedding model: {self.model_name} (provider: {self.provider})")

        if self.provider == "sentence-transformers":
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

        elif self.provider == "openai":
            try:
                import openai
                self.model = openai
                # OpenAI text-embedding-3-small has dimension 1536
                self.dimension = 1536 if "3-small" in self.model_name else 1536
                logger.info(f"OpenAI embeddings configured. Dimension: {self.dimension}")
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings
            show_progress: Show progress bar

        Returns:
            NumPy array of embeddings (shape: [n_texts, embedding_dim])
        """
        if not texts:
            return np.array([])

        logger.info(f"Generating embeddings for {len(texts)} texts")

        if self.provider == "sentence-transformers":
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )

        elif self.provider == "openai":
            embeddings = self._generate_openai_embeddings(texts)

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def _generate_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            response = client.embeddings.create(
                model=self.model_name,
                input=batch
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

            if (i + self.batch_size) % 100 == 0:
                logger.info(f"Processed {i + self.batch_size}/{len(texts)} texts")

        return np.array(embeddings)

    def embed_chunks(
        self,
        chunks: List[Dict[str, any]],
        text_key: str = "text"
    ) -> List[Dict[str, any]]:
        """
        Add embeddings to chunk dictionaries.

        Args:
            chunks: List of chunk dictionaries
            text_key: Key in dictionary containing text to embed

        Returns:
            List of chunks with 'embedding' field added
        """
        logger.info(f"Embedding {len(chunks)} chunks")

        # Extract texts
        texts = [chunk[text_key] for chunk in chunks]

        # Generate embeddings
        embeddings = self.generate_embeddings(texts)

        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]

        logger.info("Embeddings added to all chunks")
        return chunks

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.dimension


def load_embedding_model(config: Dict[str, any]) -> EmbeddingGenerator:
    """
    Load embedding model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        EmbeddingGenerator instance
    """
    embedding_config = config.get('models', {}).get('embedding', {})

    provider = embedding_config.get('provider', 'sentence-transformers')
    model_name = embedding_config.get('model_name', 'all-MiniLM-L6-v2')

    logger.info(f"Loading embedding model: {model_name} ({provider})")

    return EmbeddingGenerator(
        model_name=model_name,
        provider=provider
    )
