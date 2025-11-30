"""
User profile module for storing user preferences and context.
Supports personalized responses and preference tracking.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger


class UserProfile:
    """Manages user profiles and preferences."""

    def __init__(self, storage_dir: str = "data/memory/profiles"):
        """
        Initialize user profile manager.

        Args:
            storage_dir: Directory to store user profiles
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for active profiles
        self.active_profiles: Dict[str, Dict] = {}

        logger.info(f"User profile manager initialized at {self.storage_dir}")

    def create_profile(
        self,
        user_id: str,
        language: str = "english",
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create a new user profile.

        Args:
            user_id: User identifier
            language: Preferred language
            metadata: Additional metadata

        Returns:
            Created profile dictionary
        """
        profile = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'language': language,
            'preferences': {
                'language': language,
                'detail_level': 'medium',  # low, medium, high
                'include_sources': True
            },
            'demographics': {},
            'interests': [],
            'contraception_history': [],
            'session_count': 0,
            'metadata': metadata or {}
        }

        self.active_profiles[user_id] = profile
        self._save_profile(user_id)

        logger.info(f"Created profile for user {user_id}")
        return profile

    def get_profile(self, user_id: str) -> Optional[Dict]:
        """
        Get user profile.

        Args:
            user_id: User identifier

        Returns:
            Profile dictionary or None
        """
        # Check cache first
        if user_id in self.active_profiles:
            return self.active_profiles[user_id]

        # Load from disk
        profile = self._load_profile(user_id)
        if profile:
            self.active_profiles[user_id] = profile

        return profile

    def update_profile(
        self,
        user_id: str,
        updates: Dict[str, Any]
    ):
        """
        Update user profile.

        Args:
            user_id: User identifier
            updates: Dictionary of fields to update
        """
        profile = self.get_profile(user_id)

        if not profile:
            logger.warning(f"Profile not found for user {user_id}, creating new")
            profile = self.create_profile(user_id)

        # Update fields
        for key, value in updates.items():
            if key in profile:
                profile[key] = value
            elif key in profile['preferences']:
                profile['preferences'][key] = value
            elif key in profile['demographics']:
                profile['demographics'][key] = value

        profile['updated_at'] = datetime.now().isoformat()

        self.active_profiles[user_id] = profile
        self._save_profile(user_id)

        logger.debug(f"Updated profile for user {user_id}")

    def update_preferences(
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
        profile = self.get_profile(user_id)

        if not profile:
            profile = self.create_profile(user_id)

        profile['preferences'].update(preferences)
        profile['updated_at'] = datetime.now().isoformat()

        self.active_profiles[user_id] = profile
        self._save_profile(user_id)

        logger.debug(f"Updated preferences for user {user_id}")

    def add_interest(
        self,
        user_id: str,
        interest: str
    ):
        """
        Add a topic of interest to user profile.

        Args:
            user_id: User identifier
            interest: Interest/topic (e.g., "DMPA", "IUD", "side effects")
        """
        profile = self.get_profile(user_id)

        if not profile:
            profile = self.create_profile(user_id)

        if interest not in profile['interests']:
            profile['interests'].append(interest)
            profile['updated_at'] = datetime.now().isoformat()

            self.active_profiles[user_id] = profile
            self._save_profile(user_id)

            logger.debug(f"Added interest '{interest}' for user {user_id}")

    def add_contraception_history(
        self,
        user_id: str,
        method: str,
        notes: Optional[str] = None
    ):
        """
        Add contraception method to user history.

        Args:
            user_id: User identifier
            method: Contraception method
            notes: Additional notes
        """
        profile = self.get_profile(user_id)

        if not profile:
            profile = self.create_profile(user_id)

        entry = {
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'notes': notes
        }

        profile['contraception_history'].append(entry)
        profile['updated_at'] = datetime.now().isoformat()

        self.active_profiles[user_id] = profile
        self._save_profile(user_id)

        logger.debug(f"Added contraception history for user {user_id}")

    def increment_session_count(self, user_id: str):
        """
        Increment session count for user.

        Args:
            user_id: User identifier
        """
        profile = self.get_profile(user_id)

        if not profile:
            profile = self.create_profile(user_id)

        profile['session_count'] += 1
        profile['updated_at'] = datetime.now().isoformat()

        self.active_profiles[user_id] = profile
        self._save_profile(user_id)

    def get_preference(
        self,
        user_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get a specific user preference.

        Args:
            user_id: User identifier
            key: Preference key
            default: Default value if not found

        Returns:
            Preference value
        """
        profile = self.get_profile(user_id)

        if not profile:
            return default

        return profile.get('preferences', {}).get(key, default)

    def delete_profile(self, user_id: str):
        """
        Delete user profile (for privacy/GDPR compliance).

        Args:
            user_id: User identifier
        """
        # Remove from cache
        if user_id in self.active_profiles:
            del self.active_profiles[user_id]

        # Delete from disk
        profile_file = self.storage_dir / f"{user_id}.json"
        if profile_file.exists():
            profile_file.unlink()
            logger.info(f"Deleted profile for user {user_id}")

    def get_all_users(self) -> List[str]:
        """
        Get list of all user IDs.

        Returns:
            List of user identifiers
        """
        profile_files = self.storage_dir.glob("*.json")
        return [f.stem for f in profile_files]

    def get_user_summary(self, user_id: str) -> Dict:
        """
        Get summary of user profile.

        Args:
            user_id: User identifier

        Returns:
            Summary dictionary
        """
        profile = self.get_profile(user_id)

        if not profile:
            return {
                'user_id': user_id,
                'exists': False
            }

        return {
            'user_id': user_id,
            'exists': True,
            'created_at': profile['created_at'],
            'updated_at': profile['updated_at'],
            'language': profile['preferences']['language'],
            'session_count': profile['session_count'],
            'interests_count': len(profile['interests']),
            'history_count': len(profile['contraception_history'])
        }

    def _save_profile(self, user_id: str):
        """Save profile to disk."""
        profile_file = self.storage_dir / f"{user_id}.json"

        profile = self.active_profiles.get(user_id)
        if not profile:
            return

        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2)

    def _load_profile(self, user_id: str) -> Optional[Dict]:
        """Load profile from disk."""
        profile_file = self.storage_dir / f"{user_id}.json"

        if not profile_file.exists():
            return None

        try:
            with open(profile_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading profile {user_id}: {e}")
            return None

    def export_user_data(self, user_id: str) -> Optional[Dict]:
        """
        Export all user data (for GDPR data portability).

        Args:
            user_id: User identifier

        Returns:
            Complete user data dictionary
        """
        profile = self.get_profile(user_id)

        if not profile:
            return None

        export = {
            'profile': profile,
            'export_timestamp': datetime.now().isoformat(),
            'format_version': '1.0'
        }

        return export
