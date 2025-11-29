"""
Complete RAG Pipeline

Combines retrieval and generation for end-to-end query handling.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from src.rag.retriever import RAGRetriever, HybridRetriever
from src.rag.generator import RAGGenerator, MultilingualGenerator


class RAGPipeline:
    """
    End-to-end RAG system for contraception counseling.

    Features:
    - Document retrieval
    - Response generation
    - Multi-language support
    - Conversation history
    - Error handling
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        use_hybrid_retrieval: bool = False,
        use_multilingual: bool = True
    ):
        """
        Initialize RAG pipeline.

        Args:
            config_path: Path to configuration file
            use_hybrid_retrieval: Use hybrid retriever (keyword + semantic)
            use_multilingual: Use multilingual generator with auto-detection
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize retriever
        retriever_class = HybridRetriever if use_hybrid_retrieval else RAGRetriever
        self.retriever = retriever_class(
            vector_store_path=self.config['data']['processed_dir'] + "/vector_store",
            embeddings_config=self.config['models']['embeddings'],
            relevance_threshold=self.config['rag'].get('relevance_threshold', 0.3)
        )

        # Initialize generator
        generator_class = MultilingualGenerator if use_multilingual else RAGGenerator
        self.generator = generator_class(
            llm_config=self.config['models']['llm']
        )

        # Conversation storage
        self.conversations = {}  # session_id -> conversation history

        logger.info(f"RAGPipeline initialized (hybrid={use_hybrid_retrieval}, multilingual={use_multilingual})")

    def query(
        self,
        question: str,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 5,
        include_sources: bool = True,
        **kwargs
    ) -> Dict:
        """
        Process a user query end-to-end.

        Args:
            question: User's question
            language: Response language (auto-detected if None and using multilingual)
            session_id: Session ID for conversation tracking
            top_k: Number of documents to retrieve
            include_sources: Include source citations in response
            **kwargs: Additional arguments for generator

        Returns:
            Dictionary with response, sources, and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")

        try:
            # Step 1: Retrieve relevant context
            logger.debug("Step 1: Retrieving context")
            context = self.retriever.get_context(
                query=question,
                top_k=top_k,
                max_length=kwargs.get('max_context_length', None)
            )

            # Check if context is empty
            if not context:
                logger.warning("No relevant context found")
                # Use safety fallback
                return {
                    'response': self.generator._get_safety_fallback(language or 'english'),
                    'sources': [],
                    'language': language or 'english',
                    'metadata': {
                        'no_context': True,
                        'query': question
                    }
                }

            # Step 2: Get conversation history
            logger.debug("Step 2: Loading conversation history")
            conversation_history = None
            if session_id:
                conversation_history = self.conversations.get(session_id, [])

            # Step 3: Generate response
            logger.debug("Step 3: Generating response")
            result = self.generator.generate(
                query=question,
                context=context,
                language=language,
                conversation_history=conversation_history,
                **kwargs
            )

            # Step 4: Get sources
            if include_sources:
                logger.debug("Step 4: Extracting sources")
                sources = self.retriever.get_sources(query=question, top_k=top_k)
                result['sources'] = sources
            else:
                result['sources'] = []

            # Step 5: Update conversation history
            if session_id:
                self._update_conversation(
                    session_id=session_id,
                    query=question,
                    response=result['response'],
                    language=result['language']
                )

            # Add query to metadata
            result['metadata']['query'] = question
            result['metadata']['session_id'] = session_id

            logger.info(f"Query processed successfully ({result['language']})")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                'response': self.generator._get_safety_fallback(language or 'english'),
                'sources': [],
                'language': language or 'english',
                'metadata': {
                    'error': str(e),
                    'query': question
                }
            }

    def _update_conversation(
        self,
        session_id: str,
        query: str,
        response: str,
        language: str
    ):
        """
        Update conversation history.

        Args:
            session_id: Session identifier
            query: User's query
            response: System's response
            language: Response language
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        self.conversations[session_id].append({
            'query': query,
            'response': response,
            'language': language
        })

        # Keep only last N turns
        max_history = self.config['rag'].get('max_conversation_history', 10)
        if len(self.conversations[session_id]) > max_history:
            self.conversations[session_id] = self.conversations[session_id][-max_history:]

        logger.debug(f"Updated conversation for session {session_id}")

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of conversation turns
        """
        return self.conversations.get(session_id, [])

    def clear_conversation(self, session_id: str):
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.conversations:
            del self.conversations[session_id]
            logger.info(f"Cleared conversation for session {session_id}")

    def batch_query(
        self,
        questions: List[str],
        **kwargs
    ) -> List[Dict]:
        """
        Process multiple queries in batch.

        Args:
            questions: List of questions
            **kwargs: Arguments passed to query()

        Returns:
            List of results
        """
        logger.info(f"Processing batch of {len(questions)} queries")

        results = []
        for i, question in enumerate(questions, 1):
            logger.debug(f"Processing query {i}/{len(questions)}")
            result = self.query(question, **kwargs)
            results.append(result)

        logger.info(f"Batch processing complete: {len(results)} results")
        return results

    def evaluate_response(self, question: str, response: str) -> Dict:
        """
        Evaluate response quality.

        Args:
            question: Original question
            response: Generated response

        Returns:
            Evaluation metrics
        """
        return self.generator.validate_response(response, question)

    def get_statistics(self) -> Dict:
        """
        Get system statistics.

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_sessions': len(self.conversations),
            'total_conversations': sum(len(hist) for hist in self.conversations.values()),
            'retriever_chunks': len(self.retriever.vector_store.chunks) if hasattr(self.retriever.vector_store, 'chunks') else 0
        }

        return stats


class RAGPipelineWithMemory(RAGPipeline):
    """
    RAG Pipeline with persistent conversation memory.

    Extends RAGPipeline with:
    - Persistent storage of conversations
    - User profile tracking
    - Adherence monitoring context
    """

    def __init__(self, config_path: str = "configs/config.yaml", **kwargs):
        """
        Initialize pipeline with memory.

        Args:
            config_path: Configuration file path
            **kwargs: Additional arguments for RAGPipeline
        """
        super().__init__(config_path, **kwargs)

        # Storage for user profiles
        self.user_profiles = {}

        logger.info("RAGPipelineWithMemory initialized")

    def query_with_profile(
        self,
        question: str,
        user_profile: Dict,
        session_id: str,
        **kwargs
    ) -> Dict:
        """
        Query with user profile context.

        Args:
            question: User's question
            user_profile: User profile data
            session_id: Session identifier
            **kwargs: Additional arguments

        Returns:
            Response with personalized context
        """
        # Store/update user profile
        self.user_profiles[session_id] = user_profile

        # Extract relevant profile info
        language = user_profile.get('language_preference', 'english')
        prior_method = user_profile.get('prior_contraceptive_use')
        concerns = user_profile.get('primary_concerns', [])

        # Augment query with profile context (if relevant)
        # This helps the LLM provide personalized responses

        logger.debug(f"Processing query with profile context (lang={language})")

        # Call standard query with language
        result = self.query(
            question=question,
            language=language,
            session_id=session_id,
            **kwargs
        )

        # Add profile context to metadata
        result['metadata']['user_profile'] = {
            'prior_method': prior_method,
            'concerns': concerns
        }

        return result


# Example usage and testing
if __name__ == "__main__":
    from src.utils.logger import setup_logger

    # Setup logging
    logger = setup_logger()

    print("=" * 60)
    print("RAG Pipeline Demo")
    print("=" * 60)

    try:
        # Initialize pipeline
        print("\n1. Initializing RAG Pipeline...")
        pipeline = RAGPipeline(
            config_path="configs/config.yaml",
            use_hybrid_retrieval=False,
            use_multilingual=True
        )
        print("✓ Pipeline initialized")

        # Example queries
        queries = [
            ("What are the benefits of DMPA injection?", "english"),
            ("Quels sont les effets secondaires de la pilule?", "french"),
            ("How effective is the IUD?", "english")
        ]

        print("\n2. Processing example queries:")
        print("-" * 60)

        for i, (query, lang) in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            print(f"Language: {lang}")

            result = pipeline.query(
                question=query,
                language=lang,
                session_id="demo_session",
                top_k=3
            )

            print(f"\nResponse:\n{result['response'][:300]}...")
            print(f"\nSources: {len(result['sources'])} documents")
            if result['sources']:
                for source in result['sources'][:2]:
                    print(f"  - {source['source']} (page {source['page']})")

            print("-" * 60)

        # Show statistics
        print("\n3. Pipeline Statistics:")
        stats = pipeline.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n✓ Demo complete!")

    except FileNotFoundError as e:
        print(f"\n⚠ Warning: {e}")
        print("\nThis demo requires:")
        print("  1. Processed vector store (run preprocess_documents.py first)")
        print("  2. Ollama running with llama3.2 model")
        print("  3. Source PDFs in data/who/ and data/bcs/")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        logger.error("Demo failed", exc_info=True)
