"""
Compliance-Aware Pipeline (No RAG)

Based on Experiment 2 results showing that compliance-aware prompting
outperforms RAG-based approaches.

Results:
- Claude Opus 4.5 + Exp2 prompting: 76.25% compliant, 0 critical issues
- RAG (Exp3) showed 35% degradation in compliance

This pipeline uses direct LLM generation with compliance-aware system prompts
instead of retrieving and constraining to context.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from src.pipeline.generator import ComplianceGenerator, MultilingualGenerator, SafetyValidator
from src.memory.memory_manager import MemoryManager


class CompliancePipeline:
    """
    Compliance-first pipeline for contraception counseling.

    Features:
    - Direct LLM generation (no RAG retrieval)
    - Compliance-aware prompting from Experiment 2
    - Multi-language support
    - Conversation history
    - Safety fallbacks
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        use_multilingual: bool = True,
        use_safety_validation: bool = True
    ):
        """
        Initialize compliance pipeline.

        Args:
            config_path: Path to configuration file
            use_multilingual: Use multilingual generator with auto-detection
            use_safety_validation: Enable safety validator (Experiment 4)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize generator (no retriever needed)
        generator_class = MultilingualGenerator if use_multilingual else ComplianceGenerator
        self.generator = generator_class(
            llm_config=self.config['models']['llm']
        )

        # Initialize safety validator (Experiment 4)
        self.safety_validator = SafetyValidator() if use_safety_validation else None
        self.use_safety_validation = use_safety_validation

        # Conversation storage
        self.conversations = {}  # session_id -> conversation history

        experiment = "Experiment 4 (with safety validation)" if use_safety_validation else "Experiment 2 (compliance prompting only)"
        logger.info(f"CompliancePipeline initialized (multilingual={use_multilingual}, safety_validation={use_safety_validation})")
        logger.info(f"Using {experiment}")

    def query(
        self,
        question: str,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 5,  # Kept for API compatibility but unused
        include_sources: bool = True,  # Kept for API compatibility
        **kwargs
    ) -> Dict:
        """
        Process a user query with compliance-aware generation.

        Args:
            question: User's question
            language: Response language (auto-detected if None and using multilingual)
            session_id: Session ID for conversation tracking
            top_k: Number of documents to retrieve (unused, kept for compatibility)
            include_sources: Include source citations (unused, kept for compatibility)
            **kwargs: Additional arguments for generator

        Returns:
            Dictionary with response, sources, and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")

        try:
            # Step 1: Get conversation history
            logger.debug("Step 1: Loading conversation history")
            conversation_history = None
            if session_id:
                conversation_history = self.conversations.get(session_id, [])

            # Step 2: Generate response (NO RETRIEVAL - direct LLM)
            # Empty context string since we're not using RAG
            logger.debug("Step 2: Generating compliance-aware response")
            result = self.generator.generate(
                query=question,
                context="",  # No retrieved context
                language=language,
                conversation_history=conversation_history,
                **kwargs
            )

            # No sources since we're not using retrieval
            result['sources'] = []

            # Step 3: Safety validation (Experiment 4)
            if self.use_safety_validation and self.safety_validator:
                logger.debug("Step 3: Running safety validation")
                validation_result = self.safety_validator.validate(
                    response=result['response'],
                    question=question,
                    language=result['language']
                )

                result['metadata']['safety_validation'] = validation_result

                # If high severity issues detected, use safer fallback
                if validation_result['severity'] == 'high':
                    logger.warning(f"High severity safety issues detected: {validation_result['issues']}")
                    result['response'] = validation_result.get('fallback_response', result['response'])
                    result['metadata']['safety_fallback_used'] = True
                elif validation_result['issues']:
                    logger.info(f"Medium severity issues logged: {validation_result['issues']}")
                    result['metadata']['safety_fallback_used'] = False

            # Step 4: Update conversation history
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
            result['metadata']['approach'] = 'compliance_aware_prompting_with_validation' if self.use_safety_validation else 'compliance_aware_prompting'
            result['metadata']['experiment'] = 'exp4_safety_validation' if self.use_safety_validation else 'exp2_compliance_only'

            logger.info(f"Generated response ({len(result['response'])} chars)")
            return result

        except Exception as e:
            logger.error(f"Error in compliance pipeline: {e}")
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
        Update conversation history for a session.

        Args:
            session_id: Session ID
            query: User query
            response: System response
            language: Response language
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        self.conversations[session_id].append({
            'query': query,
            'response': response,
            'language': language
        })

        # Keep only last 5 turns
        if len(self.conversations[session_id]) > 5:
            self.conversations[session_id] = self.conversations[session_id][-5:]

        logger.debug(f"Updated conversation {session_id} (now {len(self.conversations[session_id])} turns)")

    def clear_conversation(self, session_id: str):
        """
        Clear conversation history for a session.

        Args:
            session_id: Session ID
        """
        if session_id in self.conversations:
            del self.conversations[session_id]
            logger.info(f"Cleared conversation {session_id}")

    def get_stats(self) -> Dict:
        """
        Get pipeline statistics.

        Returns:
            Statistics dictionary
        """
        total_sessions = len(self.conversations)
        total_turns = sum(len(conv) for conv in self.conversations.values())

        return {
            'total_sessions': total_sessions,
            'total_turns': total_turns,
            'approach': 'compliance_aware_prompting',
            'experiment_basis': 'exp2',
            'model': self.config['models']['llm'].get('model_name', 'unknown')
        }


class CompliancePipelineWithMemory(CompliancePipeline):
    """
    Compliance Pipeline with persistent conversation memory and user profiles.

    Extends CompliancePipeline with:
    - Persistent storage of conversations
    - User profile tracking
    - Interest extraction
    - Session management
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        memory_dir: str = "data/memory",
        use_multilingual: bool = True,
        use_safety_validation: bool = True
    ):
        """
        Initialize compliance pipeline with memory.

        Args:
            config_path: Configuration file path
            memory_dir: Directory for memory storage
            use_multilingual: Use multilingual generator
            use_safety_validation: Enable safety validator (Experiment 4)
        """
        super().__init__(config_path, use_multilingual, use_safety_validation)

        # Initialize memory manager
        self.memory = MemoryManager(
            conversation_dir=f"{memory_dir}/conversations",
            profile_dir=f"{memory_dir}/profiles"
        )

        logger.info("CompliancePipelineWithMemory initialized with persistent storage")

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
            top_k: Number of documents (unused, for compatibility)
            include_sources: Include sources (unused, for compatibility)
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
            if not language and memory_context.get('user_preferences'):
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
                metadata={'language': result['language']}
            )

            # Update user profile
            if user_id:
                self.memory.update_user_profile(
                    user_id=user_id,
                    interaction_data={
                        'query': question,
                        'language': result['language']
                    }
                )

        return result

    def get_user_profile(self, user_id: str) -> Dict:
        """Get user profile."""
        return self.memory.get_user_profile(user_id)

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history from memory."""
        return self.memory.get_conversation(session_id)

    def export_user_data(self, user_id: str, output_path: str):
        """Export user data for GDPR compliance."""
        self.memory.export_user_data(user_id, output_path)

    def delete_user_data(self, user_id: str):
        """Delete user data for GDPR compliance."""
        self.memory.delete_user_data(user_id)


if __name__ == "__main__":
    from src.utils.logger import setup_logger

    # Setup logging
    logger = setup_logger()

    # Initialize pipeline
    pipeline = CompliancePipeline()

    # Example query
    question = "I'm breastfeeding. Can I use the implant?"

    # Process query
    result = pipeline.query(
        question=question,
        language='english',
        session_id='test_session_001'
    )

    print(f"\nQuery: {question}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata: {result['metadata']}")

    # Get stats
    stats = pipeline.get_stats()
    print(f"\nPipeline stats: {stats}")
