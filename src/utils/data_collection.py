"""
Optional data collection module for research purposes.
Implements privacy-preserving, opt-in data collection.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from loguru import logger


class DataCollector:
    """
    Privacy-preserving data collector for research.

    Features:
    - Anonymous user IDs (no PII)
    - Opt-in only
    - Separate storage for feedback
    - Easy to disable completely
    """

    def __init__(self, storage_dir: str = "data/collected", enabled: bool = False):
        """
        Initialize data collector.

        Args:
            storage_dir: Directory to store collected data
            enabled: Whether data collection is enabled (default: False for privacy)
        """
        self.storage_dir = Path(storage_dir)
        self.enabled = enabled

        if self.enabled:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Data collection enabled. Storing in {self.storage_dir}")
        else:
            logger.info("Data collection DISABLED (privacy mode)")

    def generate_anonymous_id(self) -> str:
        """Generate anonymous user ID (random UUID)."""
        return str(uuid.uuid4())

    def collect_interaction(
        self,
        query: str,
        response: str,
        session_id: str = None,
        user_opted_in: bool = False,
        metadata: Dict = None
    ) -> bool:
        """
        Collect a query-response interaction.

        Args:
            query: User's question
            response: System's response
            session_id: Optional session identifier
            user_opted_in: User must explicitly opt-in
            metadata: Optional metadata (language, etc.)

        Returns:
            True if collected, False if not
        """
        if not self.enabled or not user_opted_in:
            return False

        interaction = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id or self.generate_anonymous_id(),
            'query': query,
            'response': response,
            'metadata': metadata or {}
        }

        # Save to file
        filename = self.storage_dir / f"interaction_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(interaction, f, indent=2, ensure_ascii=False)

        logger.debug(f"Interaction collected: {filename}")
        return True

    def collect_feedback(
        self,
        session_id: str,
        rating: int,
        helpful: bool,
        comments: str = None,
        user_opted_in: bool = False
    ) -> bool:
        """
        Collect user feedback.

        Args:
            session_id: Session identifier
            rating: 1-5 star rating
            helpful: Whether response was helpful
            comments: Optional qualitative feedback
            user_opted_in: User must explicitly opt-in

        Returns:
            True if collected, False if not
        """
        if not self.enabled or not user_opted_in:
            return False

        feedback = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'rating': rating,
            'helpful': helpful,
            'comments': comments
        }

        # Save to separate feedback file
        feedback_file = self.storage_dir / "feedback.jsonl"
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback, ensure_ascii=False) + '\n')

        logger.debug("Feedback collected")
        return True

    def get_statistics(self) -> Dict:
        """
        Get statistics about collected data.

        Returns:
            Dictionary with stats
        """
        if not self.enabled:
            return {'enabled': False}

        interactions = list(self.storage_dir.glob("interaction_*.json"))

        stats = {
            'enabled': True,
            'total_interactions': len(interactions),
            'storage_dir': str(self.storage_dir)
        }

        # Count feedback entries
        feedback_file = self.storage_dir / "feedback.jsonl"
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                stats['total_feedback'] = sum(1 for _ in f)
        else:
            stats['total_feedback'] = 0

        return stats

    def export_dataset(self, output_file: str):
        """
        Export all collected data to a single file for analysis.

        Args:
            output_file: Path to output JSON file
        """
        if not self.enabled:
            logger.warning("Data collection is disabled")
            return

        interactions = []
        for file in sorted(self.storage_dir.glob("interaction_*.json")):
            with open(file, 'r', encoding='utf-8') as f:
                interactions.append(json.load(f))

        dataset = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_interactions': len(interactions)
            },
            'interactions': interactions
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        logger.info(f"Dataset exported to {output_file}")


class ConsentManager:
    """Manages user consent for data collection."""

    @staticmethod
    def get_consent_text() -> str:
        """Get consent form text."""
        return """
        DATA COLLECTION CONSENT

        This research study collects anonymous interaction data to improve
        contraception counseling systems.

        What we collect (if you opt-in):
        - Your questions and the system's responses
        - Timestamps
        - Optional feedback ratings

        What we DO NOT collect:
        - Your name or personal identifiers
        - Contact information
        - Location data
        - Any medical records

        Your data will be:
        - Stored anonymously
        - Used only for research purposes
        - Never sold or shared with third parties
        - Deleted upon request

        You can opt-out at any time.

        Do you consent to anonymous data collection? (yes/no)
        """

    @staticmethod
    def verify_consent(user_response: str) -> bool:
        """
        Verify user consent.

        Args:
            user_response: User's response to consent prompt

        Returns:
            True if user consents, False otherwise
        """
        positive_responses = ['yes', 'y', 'agree', 'accept', 'consent']
        return user_response.lower().strip() in positive_responses


# Example usage in API
"""
# In your FastAPI app:

from src.utils.data_collection import DataCollector

# Initialize (disabled by default for privacy)
collector = DataCollector(enabled=False)  # Set to True only if doing pilot study

@app.post("/counsel/query")
async def query(
    question: str,
    session_id: str = None,
    collect_data: bool = False  # User must opt-in
):
    # Generate response
    response = rag_system.query(question)

    # Optionally collect data (if user opted in)
    if collect_data:
        collector.collect_interaction(
            query=question,
            response=response,
            session_id=session_id,
            user_opted_in=True,
            metadata={'language': detect_language(question)}
        )

    return {'response': response}

@app.post("/feedback")
async def submit_feedback(
    session_id: str,
    rating: int,
    helpful: bool,
    comments: str = None,
    opted_in: bool = False
):
    collector.collect_feedback(
        session_id=session_id,
        rating=rating,
        helpful=helpful,
        comments=comments,
        user_opted_in=opted_in
    )
    return {'status': 'received'}
"""
