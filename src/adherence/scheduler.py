"""
Reminder Scheduler

Schedules and manages contraception adherence reminders across multiple
communication channels.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger


class ReminderScheduler:
    """Schedule and manage adherence reminders."""

    CHANNELS = [
        "SMS reminder",
        "Phone call",
        "WhatsApp message",
        "Community health worker visit"
    ]

    def __init__(self, storage_dir: str = "data/adherence"):
        """
        Initialize reminder scheduler.

        Args:
            storage_dir: Directory to store reminder records
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.reminders_file = self.storage_dir / "reminders.json"
        self._load_reminders()

    def _load_reminders(self):
        """Load existing reminders."""
        if self.reminders_file.exists():
            with open(self.reminders_file, 'r', encoding='utf-8') as f:
                self.reminders = json.load(f)
            logger.info(f"Loaded {len(self.reminders)} reminders")
        else:
            self.reminders = {}
            logger.info("No existing reminders found")

    def _save_reminders(self):
        """Save reminders to file."""
        with open(self.reminders_file, 'w', encoding='utf-8') as f:
            json.dump(self.reminders, f, indent=2, ensure_ascii=False)

    def schedule_reminder(
        self,
        user_id: str,
        reminder_date: str,
        channel: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Schedule a reminder for a user.

        Args:
            user_id: User identifier
            reminder_date: Date to send reminder (ISO format)
            channel: Communication channel
            message: Reminder message
            context: Optional context (method, days_until, etc.)

        Returns:
            Reminder ID
        """
        if channel not in self.CHANNELS:
            logger.warning(f"Unknown channel: {channel}, using SMS")
            channel = "SMS reminder"

        reminder_id = f"reminder_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        reminder = {
            "reminder_id": reminder_id,
            "user_id": user_id,
            "reminder_date": reminder_date,
            "channel": channel,
            "message": message,
            "context": context or {},
            "status": "scheduled",
            "created_at": datetime.now().isoformat(),
            "sent_at": None,
            "response": None
        }

        self.reminders[reminder_id] = reminder
        self._save_reminders()

        logger.info(f"Scheduled reminder {reminder_id} for user {user_id} via {channel}")
        return reminder_id

    def get_due_reminders(self) -> List[Dict[str, Any]]:
        """
        Get reminders that are due to be sent.

        Returns:
            List of due reminders
        """
        now = datetime.now()
        due_reminders = []

        for reminder_id, reminder in self.reminders.items():
            if reminder["status"] == "scheduled":
                reminder_date = datetime.fromisoformat(reminder["reminder_date"])

                if reminder_date <= now:
                    due_reminders.append(reminder)

        return due_reminders

    def mark_sent(self, reminder_id: str) -> bool:
        """
        Mark reminder as sent.

        Args:
            reminder_id: Reminder identifier

        Returns:
            Success status
        """
        if reminder_id not in self.reminders:
            logger.error(f"Reminder {reminder_id} not found")
            return False

        self.reminders[reminder_id]["status"] = "sent"
        self.reminders[reminder_id]["sent_at"] = datetime.now().isoformat()
        self._save_reminders()

        logger.info(f"Marked reminder {reminder_id} as sent")
        return True

    def record_response(
        self,
        reminder_id: str,
        responded: bool,
        response_time: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Record user response to reminder.

        Args:
            reminder_id: Reminder identifier
            responded: Whether user responded
            response_time: Time of response (ISO format)
            notes: Optional notes about response

        Returns:
            Success status
        """
        if reminder_id not in self.reminders:
            logger.error(f"Reminder {reminder_id} not found")
            return False

        if response_time is None:
            response_time = datetime.now().isoformat()

        self.reminders[reminder_id]["response"] = {
            "responded": responded,
            "response_time": response_time,
            "notes": notes
        }

        self.reminders[reminder_id]["status"] = "responded" if responded else "no_response"
        self._save_reminders()

        logger.info(f"Recorded response for reminder {reminder_id}: {'responded' if responded else 'no response'}")
        return True

    def get_user_reminders(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all reminders for a user.

        Args:
            user_id: User identifier

        Returns:
            List of reminders
        """
        user_reminders = []

        for reminder in self.reminders.values():
            if reminder["user_id"] == user_id:
                user_reminders.append(reminder)

        # Sort by reminder date
        user_reminders.sort(key=lambda r: r["reminder_date"], reverse=True)

        return user_reminders

    def calculate_response_rate(self, user_id: str) -> Optional[float]:
        """
        Calculate response rate for a user.

        Args:
            user_id: User identifier

        Returns:
            Response rate (0.0 - 1.0) or None
        """
        user_reminders = self.get_user_reminders(user_id)

        if not user_reminders:
            return None

        # Count sent reminders with responses
        sent_reminders = [r for r in user_reminders if r["status"] in ["sent", "responded", "no_response"]]

        if not sent_reminders:
            return None

        responded_count = sum(
            1 for r in sent_reminders
            if r.get("response") and r["response"].get("responded", False)
        )

        return responded_count / len(sent_reminders)

    def calculate_channel_effectiveness(
        self,
        user_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate response rates by channel.

        Args:
            user_id: Optional user to calculate for (None = all users)

        Returns:
            Dictionary of channel -> response rate
        """
        channel_stats = {channel: {"sent": 0, "responded": 0} for channel in self.CHANNELS}

        reminders_to_analyze = []

        if user_id:
            reminders_to_analyze = self.get_user_reminders(user_id)
        else:
            reminders_to_analyze = list(self.reminders.values())

        for reminder in reminders_to_analyze:
            if reminder["status"] in ["sent", "responded", "no_response"]:
                channel = reminder["channel"]
                channel_stats[channel]["sent"] += 1

                if reminder.get("response") and reminder["response"].get("responded", False):
                    channel_stats[channel]["responded"] += 1

        # Calculate rates
        channel_effectiveness = {}
        for channel, stats in channel_stats.items():
            if stats["sent"] > 0:
                channel_effectiveness[channel] = stats["responded"] / stats["sent"]
            else:
                channel_effectiveness[channel] = 0.0

        return channel_effectiveness

    def get_reminder_stats(self) -> Dict[str, Any]:
        """Get comprehensive reminder statistics."""
        total_reminders = len(self.reminders)
        scheduled = sum(1 for r in self.reminders.values() if r["status"] == "scheduled")
        sent = sum(1 for r in self.reminders.values() if r["status"] == "sent")
        responded = sum(1 for r in self.reminders.values() if r["status"] == "responded")
        no_response = sum(1 for r in self.reminders.values() if r["status"] == "no_response")

        overall_response_rate = responded / (responded + no_response) if (responded + no_response) > 0 else 0.0

        return {
            "total_reminders": total_reminders,
            "scheduled": scheduled,
            "sent": sent,
            "responded": responded,
            "no_response": no_response,
            "overall_response_rate": overall_response_rate,
            "channel_effectiveness": self.calculate_channel_effectiveness()
        }

    def send_reminder(self, reminder: Dict[str, Any]) -> bool:
        """
        Simulate sending a reminder (in production, integrate with actual SMS/etc.).

        Args:
            reminder: Reminder dictionary

        Returns:
            Success status
        """
        # In production, integrate with:
        # - Twilio for SMS
        # - WhatsApp Business API
        # - Phone system for calls
        # - CHW scheduling system

        logger.info(f"SENDING REMINDER: {reminder['channel']} to user {reminder['user_id']}")
        logger.info(f"Message: {reminder['message']}")

        # Mark as sent
        return self.mark_sent(reminder["reminder_id"])


# Example usage
if __name__ == "__main__":
    scheduler = ReminderScheduler()

    # Schedule a reminder
    reminder_id = scheduler.schedule_reminder(
        user_id="user_001",
        reminder_date=(datetime.now() + timedelta(days=7)).isoformat(),
        channel="SMS reminder",
        message="Your DMPA injection is due in 7 days. Please schedule an appointment.",
        context={"method": "DMPA", "days_until": 7}
    )

    print(f"Scheduled reminder: {reminder_id}")

    # Get due reminders
    due = scheduler.get_due_reminders()
    print(f"\nDue reminders: {len(due)}")

    # Get stats
    stats = scheduler.get_reminder_stats()
    print(f"\nReminder stats:")
    print(json.dumps(stats, indent=2))
