"""
Conversation memory module for storing and retrieving conversation history.
Supports multi-session tracking and conversation summarization.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


class ConversationMemory:
    """Manages conversation history for sessions."""

    def __init__(self, storage_dir: str = "data/memory/conversations"):
        """
        Initialize conversation memory.

        Args:
            storage_dir: Directory to store conversation data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for active sessions
        self.active_sessions: Dict[str, List[Dict]] = {}

        logger.info(f"Conversation memory initialized at {self.storage_dir}")

    def add_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add a conversation turn to session history.

        Args:
            session_id: Session identifier
            query: User's question
            response: System's response
            sources: Retrieved sources used
            metadata: Additional metadata
        """
        turn = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'sources': sources or [],
            'metadata': metadata or {}
        }

        # Add to in-memory cache
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []

        self.active_sessions[session_id].append(turn)

        # Persist to disk
        self._save_session(session_id)

        logger.debug(f"Added turn to session {session_id}")

    def get_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            max_turns: Maximum number of recent turns to return

        Returns:
            List of conversation turns
        """
        # Check in-memory cache first
        if session_id in self.active_sessions:
            history = self.active_sessions[session_id]
        else:
            # Load from disk
            history = self._load_session(session_id)
            self.active_sessions[session_id] = history

        # Return most recent turns if max_turns specified
        if max_turns:
            return history[-max_turns:]

        return history

    def get_formatted_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None,
        include_sources: bool = False
    ) -> str:
        """
        Get conversation history formatted as a string.

        Args:
            session_id: Session identifier
            max_turns: Maximum number of recent turns
            include_sources: Whether to include source citations

        Returns:
            Formatted conversation history
        """
        history = self.get_history(session_id, max_turns)

        if not history:
            return ""

        formatted = []
        for turn in history:
            formatted.append(f"User: {turn['query']}")
            formatted.append(f"Assistant: {turn['response']}")

            if include_sources and turn.get('sources'):
                sources_str = ", ".join([s.get('source', 'Unknown') for s in turn['sources']])
                formatted.append(f"Sources: {sources_str}")

            formatted.append("")  # Empty line between turns

        return "\n".join(formatted)

    def clear_session(self, session_id: str):
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier
        """
        # Remove from in-memory cache
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        # Delete from disk
        session_file = self.storage_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            logger.info(f"Cleared session {session_id}")

    def get_all_sessions(self) -> List[str]:
        """
        Get list of all session IDs.

        Returns:
            List of session identifiers
        """
        session_files = self.storage_dir.glob("*.json")
        return [f.stem for f in session_files]

    def get_session_summary(self, session_id: str) -> Dict:
        """
        Get summary statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Summary dictionary with stats
        """
        history = self.get_history(session_id)

        if not history:
            return {
                'session_id': session_id,
                'total_turns': 0,
                'start_time': None,
                'last_activity': None
            }

        return {
            'session_id': session_id,
            'total_turns': len(history),
            'start_time': history[0]['timestamp'],
            'last_activity': history[-1]['timestamp'],
            'languages': list(set(turn.get('metadata', {}).get('language', 'unknown')
                                for turn in history))
        }

    def summarize_conversation(
        self,
        session_id: str,
        max_length: int = 500
    ) -> str:
        """
        Generate a summary of the conversation.

        Args:
            session_id: Session identifier
            max_length: Maximum summary length

        Returns:
            Conversation summary
        """
        history = self.get_history(session_id)

        if not history:
            return "No conversation history."

        # Extract main topics from queries
        topics = []
        for turn in history:
            query = turn['query'].lower()
            # Simple keyword extraction (can be enhanced with NLP)
            if 'side effect' in query or 'side-effect' in query:
                topics.append('side effects')
            if 'dmpa' in query or 'injection' in query:
                topics.append('DMPA/injection')
            if 'iud' in query or 'copper' in query:
                topics.append('IUD')
            if 'pill' in query or 'oral' in query:
                topics.append('oral contraceptives')
            if 'implant' in query:
                topics.append('implants')

        unique_topics = list(set(topics))

        summary = f"Conversation with {len(history)} turns"
        if unique_topics:
            summary += f" about: {', '.join(unique_topics[:5])}"

        return summary[:max_length]

    def _save_session(self, session_id: str):
        """Save session to disk."""
        session_file = self.storage_dir / f"{session_id}.json"

        data = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'history': self.active_sessions.get(session_id, [])
        }

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _load_session(self, session_id: str) -> List[Dict]:
        """Load session from disk."""
        session_file = self.storage_dir / f"{session_id}.json"

        if not session_file.exists():
            return []

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('history', [])
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return []

    def cleanup_old_sessions(self, days: int = 30):
        """
        Delete sessions older than specified days.

        Args:
            days: Number of days to retain
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0

        for session_id in self.get_all_sessions():
            summary = self.get_session_summary(session_id)
            if summary['last_activity']:
                last_activity = datetime.fromisoformat(summary['last_activity'])
                if last_activity < cutoff:
                    self.clear_session(session_id)
                    deleted += 1

        logger.info(f"Cleaned up {deleted} old sessions")
        return deleted
