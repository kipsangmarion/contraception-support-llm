"""
Memory manager that orchestrates conversation memory and user profiles.
Provides unified interface for memory operations.
"""

from typing import Dict, List, Optional, Any
from loguru import logger

from .conversation_memory import ConversationMemory
from .user_profile import UserProfile


class MemoryManager:
    """Unified memory management for conversations and user profiles."""

    def __init__(
        self,
        conversation_dir: str = "data/memory/conversations",
        profile_dir: str = "data/memory/profiles"
    ):
        """
        Initialize memory manager.

        Args:
            conversation_dir: Directory for conversation storage
            profile_dir: Directory for profile storage
        """
        self.conversation_memory = ConversationMemory(storage_dir=conversation_dir)
        self.user_profiles = UserProfile(storage_dir=profile_dir)

        logger.info("Memory manager initialized")

    def start_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        language: str = "english"
    ) -> Dict:
        """
        Start a new conversation session.

        Args:
            session_id: Session identifier
            user_id: Optional user identifier for personalization
            language: Preferred language

        Returns:
            Session information
        """
        session_info = {
            'session_id': session_id,
            'user_id': user_id,
            'language': language
        }

        # If user_id provided, get or create profile
        if user_id:
            profile = self.user_profiles.get_profile(user_id)

            if not profile:
                profile = self.user_profiles.create_profile(
                    user_id=user_id,
                    language=language
                )

            # Increment session count
            self.user_profiles.increment_session_count(user_id)

            # Use user's preferred language if available
            session_info['language'] = profile['preferences']['language']

        logger.info(f"Started session {session_id}" + (f" for user {user_id}" if user_id else ""))
        return session_info

    def add_interaction(
        self,
        session_id: str,
        query: str,
        response: str,
        sources: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Record a conversation interaction.

        Args:
            session_id: Session identifier
            query: User's question
            response: System's response
            sources: Retrieved sources
            user_id: Optional user identifier
            metadata: Additional metadata
        """
        # Add to conversation memory
        self.conversation_memory.add_turn(
            session_id=session_id,
            query=query,
            response=response,
            sources=sources,
            metadata=metadata
        )

        # Update user profile if provided
        if user_id:
            # Extract and add interests from query
            self._extract_and_update_interests(user_id, query)

        logger.debug(f"Recorded interaction for session {session_id}")

    def get_context_for_query(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        max_turns: int = 5
    ) -> Dict[str, Any]:
        """
        Get context for current query including history and user profile.

        Args:
            session_id: Session identifier
            user_id: Optional user identifier
            max_turns: Maximum conversation turns to include

        Returns:
            Context dictionary
        """
        context = {
            'conversation_history': [],
            'user_preferences': {},
            'user_interests': [],
            'contraception_history': []
        }

        # Get conversation history
        history = self.conversation_memory.get_history(
            session_id=session_id,
            max_turns=max_turns
        )
        context['conversation_history'] = history

        # Get user profile if available
        if user_id:
            profile = self.user_profiles.get_profile(user_id)
            if profile:
                context['user_preferences'] = profile['preferences']
                context['user_interests'] = profile['interests']
                context['contraception_history'] = profile['contraception_history']

        return context

    def get_formatted_history(
        self,
        session_id: str,
        max_turns: int = 5
    ) -> str:
        """
        Get formatted conversation history for LLM context.

        Args:
            session_id: Session identifier
            max_turns: Maximum turns to include

        Returns:
            Formatted history string
        """
        return self.conversation_memory.get_formatted_history(
            session_id=session_id,
            max_turns=max_turns,
            include_sources=False
        )

    def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ):
        """
        Update user preferences.

        Args:
            user_id: User identifier
            preferences: Preference updates
        """
        self.user_profiles.update_preferences(user_id, preferences)
        logger.info(f"Updated preferences for user {user_id}")

    def clear_session(self, session_id: str):
        """
        Clear conversation history for session.

        Args:
            session_id: Session identifier
        """
        self.conversation_memory.clear_session(session_id)
        logger.info(f"Cleared session {session_id}")

    def delete_user_data(self, user_id: str):
        """
        Delete all user data (GDPR compliance).

        Args:
            user_id: User identifier
        """
        self.user_profiles.delete_profile(user_id)
        logger.info(f"Deleted all data for user {user_id}")

    def get_session_summary(self, session_id: str) -> Dict:
        """
        Get summary of conversation session.

        Args:
            session_id: Session identifier

        Returns:
            Session summary
        """
        return self.conversation_memory.get_session_summary(session_id)

    def get_user_summary(self, user_id: str) -> Dict:
        """
        Get summary of user profile.

        Args:
            user_id: User identifier

        Returns:
            User summary
        """
        return self.user_profiles.get_user_summary(user_id)

    def export_user_data(self, user_id: str) -> Dict:
        """
        Export all user data (GDPR data portability).

        Args:
            user_id: User identifier

        Returns:
            Complete user data export
        """
        # Get profile data
        profile_data = self.user_profiles.export_user_data(user_id)

        # Get all sessions for this user (simplified - would need session-user mapping)
        # For now, just return profile data
        export = {
            'user_data': profile_data,
            'note': 'Session data export requires session-user mapping implementation'
        }

        return export

    def cleanup_old_data(self, days: int = 30):
        """
        Clean up old conversation data.

        Args:
            days: Number of days to retain

        Returns:
            Number of sessions deleted
        """
        deleted = self.conversation_memory.cleanup_old_sessions(days=days)
        logger.info(f"Cleaned up {deleted} old sessions")
        return deleted

    def _extract_and_update_interests(self, user_id: str, query: str):
        """
        Extract contraception topics from query and update user interests.

        Args:
            user_id: User identifier
            query: User's query
        """
        query_lower = query.lower()

        # Define interest keywords
        interest_keywords = {
            'dmpa': ['dmpa', 'depot', 'medroxyprogesterone'],
            'iud': ['iud', 'intrauterine', 'copper iud', 'hormonal iud'],
            'pill': ['pill', 'oral contraceptive', 'combined pill'],
            'implant': ['implant', 'nexplanon', 'implanon'],
            'injection': ['injection', 'injectable'],
            'emergency': ['emergency contraception', 'morning after'],
            'condom': ['condom', 'barrier method'],
            'side_effects': ['side effect', 'adverse effect', 'complication'],
            'effectiveness': ['effective', 'effectiveness', 'failure rate'],
            'counseling': ['counsel', 'advice', 'guidance']
        }

        # Check for interests
        for interest, keywords in interest_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    self.user_profiles.add_interest(user_id, interest)
                    break

    def get_statistics(self) -> Dict:
        """
        Get overall memory system statistics.

        Returns:
            Statistics dictionary
        """
        total_sessions = len(self.conversation_memory.get_all_sessions())
        total_users = len(self.user_profiles.get_all_users())

        return {
            'total_sessions': total_sessions,
            'total_users': total_users,
            'active_sessions': len(self.conversation_memory.active_sessions),
            'active_profiles': len(self.user_profiles.active_profiles)
        }
