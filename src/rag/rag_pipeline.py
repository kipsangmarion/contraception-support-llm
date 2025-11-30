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
from src.memory.memory_manager import MemoryManager


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
    RAG Pipeline with persistent conversation memory and user profiles.

    Extends RAGPipeline with:
    - Persistent storage of conversations
    - User profile tracking
    - Interest extraction
    - Session management
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        memory_dir: str = "data/memory",
        **kwargs
    ):
        """
        Initialize pipeline with memory.

        Args:
            config_path: Configuration file path
            memory_dir: Directory for memory storage
            **kwargs: Additional arguments for RAGPipeline
        """
        super().__init__(config_path, **kwargs)

        # Initialize memory manager
        self.memory = MemoryManager(
            conversation_dir=f"{memory_dir}/conversations",
            profile_dir=f"{memory_dir}/profiles"
        )

        logger.info("RAGPipelineWithMemory initialized with persistent storage")

    def query(
        self,
        question: str,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        top_k: int = 5,
        include_sources: bool = True,
        **kwargs
    ) -> Dict:
        """
        Process query with memory support.

        Args:
            question: User's question
            language: Response language
            session_id: Session identifier
            user_id: Optional user identifier for personalization
            top_k: Number of documents to retrieve
            include_sources: Include source citations
            **kwargs: Additional arguments

        Returns:
            Response with memory integration
        """
        # Start/resume session if needed
        if session_id and user_id:
            self.memory.start_session(
                session_id=session_id,
                user_id=user_id,
                language=language or 'english'
            )

        # Get memory context
        memory_context = None
        if session_id:
            memory_context = self.memory.get_context_for_query(
                session_id=session_id,
                user_id=user_id,
                max_turns=5
            )

            # Override language from user profile if available
            if not language and memory_context['user_preferences']:
                language = memory_context['user_preferences'].get('language', 'english')

        # Process query with parent class
        result = super().query(
            question=question,
            language=language,
            session_id=session_id,
            top_k=top_k,
            include_sources=include_sources,
            **kwargs
        )

        # Store interaction in memory
        if session_id:
            self.memory.add_interaction(
                session_id=session_id,
                query=question,
                response=result['response'],
                sources=result.get('sources', []),
                user_id=user_id,
                metadata=result.get('metadata', {})
            )

        # Add memory stats to metadata
        if memory_context:
            result['metadata']['memory'] = {
                'turns_in_history': len(memory_context['conversation_history']),
                'user_interests': memory_context['user_interests'][:5]  # Top 5
            }

        return result

    def query_with_profile(
        self,
        question: str,
        user_id: str,
        session_id: str,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Query with explicit user profile handling.

        Args:
            question: User's question
            user_id: User identifier
            session_id: Session identifier
            language: Optional language override
            **kwargs: Additional arguments

        Returns:
            Response with personalized context
        """
        # Get user profile
        profile = self.memory.user_profiles.get_profile(user_id)

        if not profile:
            # Create new profile
            profile = self.memory.user_profiles.create_profile(
                user_id=user_id,
                language=language or 'english'
            )

        # Use profile language if not specified
        if not language:
            language = profile['preferences']['language']

        # Process query
        result = self.query(
            question=question,
            language=language,
            session_id=session_id,
            user_id=user_id,
            **kwargs
        )

        # Add profile info to metadata
        result['metadata']['user_profile'] = {
            'session_count': profile['session_count'],
            'interests': profile['interests'][:5],
            'contraception_history': len(profile['contraception_history'])
        }

        return result

    def clear_session(self, session_id: str):
        """
        Clear session memory.

        Args:
            session_id: Session to clear
        """
        self.memory.clear_session(session_id)
        logger.info(f"Cleared session {session_id}")

    def get_user_summary(self, user_id: str) -> Dict:
        """
        Get user profile summary.

        Args:
            user_id: User identifier

        Returns:
            User summary
        """
        return self.memory.get_user_summary(user_id)

    def export_user_data(self, user_id: str) -> Dict:
        """
        Export user data (GDPR compliance).

        Args:
            user_id: User identifier

        Returns:
            Complete user data export
        """
        return self.memory.export_user_data(user_id)

    def delete_user_data(self, user_id: str):
        """
        Delete all user data (GDPR compliance).

        Args:
            user_id: User identifier
        """
        self.memory.delete_user_data(user_id)
        logger.info(f"Deleted all data for user {user_id}")

    def get_memory_statistics(self) -> Dict:
        """
        Get memory system statistics.

        Returns:
            Statistics dictionary
        """
        return self.memory.get_statistics()


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
