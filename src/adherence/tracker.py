"""
Adherence Tracker

Tracks user contraception adherence, calculates next appointment dates,
and monitors adherence rates over time.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger


class AdherenceTracker:
    """Track user contraception adherence."""

    # Method intervals in days
    METHOD_INTERVALS = {
        "DMPA": 90,  # DMPA injection every 3 months
        "implant": 1825,  # Implant lasts 5 years
        "IUD": 1825,  # IUD lasts 5+ years
        "pill": 1,  # Daily pill
        "patch": 7,  # Weekly patch
        "ring": 28,  # Monthly ring
        "condoms": 0  # Per use, no schedule
    }

    def __init__(self, storage_dir: str = "data/adherence"):
        """
        Initialize adherence tracker.

        Args:
            storage_dir: Directory to store adherence records
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.records_file = self.storage_dir / "adherence_records.json"
        self._load_records()

    def _load_records(self):
        """Load existing adherence records."""
        if self.records_file.exists():
            with open(self.records_file, 'r', encoding='utf-8') as f:
                self.records = json.load(f)
            logger.info(f"Loaded {len(self.records)} adherence records")
        else:
            self.records = {}
            logger.info("No existing adherence records found")

    def _save_records(self):
        """Save adherence records to file."""
        with open(self.records_file, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def start_tracking(
        self,
        user_id: str,
        method: str,
        start_date: Optional[str] = None,
        preferred_channel: str = "SMS reminder"
    ) -> Dict[str, Any]:
        """
        Start tracking adherence for a user.

        Args:
            user_id: User identifier
            method: Contraception method
            start_date: Start date (ISO format), defaults to today
            preferred_channel: Preferred communication channel

        Returns:
            Tracking record
        """
        if start_date is None:
            start_date = datetime.now().isoformat()

        if method not in self.METHOD_INTERVALS:
            logger.warning(f"Unknown method: {method}, using generic tracking")
            interval_days = 90  # Default


        else:
            interval_days = self.METHOD_INTERVALS[method]

        record = {
            "user_id": user_id,
            "method": method,
            "start_date": start_date,
            "interval_days": interval_days,
            "preferred_channel": preferred_channel,
            "history": [],
            "created_at": datetime.now().isoformat()
        }

        self.records[user_id] = record
        self._save_records()

        logger.info(f"Started tracking for user {user_id}: {method} (every {interval_days} days)")
        return record

    def record_appointment(
        self,
        user_id: str,
        appointment_date: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Record an appointment/injection/refill.

        Args:
            user_id: User identifier
            appointment_date: Appointment date (ISO format), defaults to today
            notes: Optional notes about appointment

        Returns:
            Success status
        """
        if user_id not in self.records:
            logger.error(f"User {user_id} not found in tracking")
            return False

        if appointment_date is None:
            appointment_date = datetime.now().isoformat()

        appointment = {
            "date": appointment_date,
            "notes": notes,
            "recorded_at": datetime.now().isoformat()
        }

        self.records[user_id]["history"].append(appointment)
        self._save_records()

        logger.info(f"Recorded appointment for user {user_id} on {appointment_date}")
        return True

    def get_next_appointment_date(self, user_id: str) -> Optional[str]:
        """
        Calculate next appointment date for user.

        Args:
            user_id: User identifier

        Returns:
            Next appointment date (ISO format) or None
        """
        if user_id not in self.records:
            return None

        record = self.records[user_id]

        # Get last appointment date
        if record["history"]:
            last_appointment = record["history"][-1]["date"]
        else:
            last_appointment = record["start_date"]

        # Calculate next date
        last_date = datetime.fromisoformat(last_appointment)
        interval = timedelta(days=record["interval_days"])
        next_date = last_date + interval

        return next_date.isoformat()

    def get_days_until_next(self, user_id: str) -> Optional[int]:
        """
        Get days until next appointment.

        Args:
            user_id: User identifier

        Returns:
            Days until next appointment or None
        """
        next_date_str = self.get_next_appointment_date(user_id)
        if not next_date_str:
            return None

        next_date = datetime.fromisoformat(next_date_str)
        days_until = (next_date - datetime.now()).days

        return days_until

    def is_overdue(self, user_id: str, grace_period_days: int = 7) -> bool:
        """
        Check if user is overdue for appointment.

        Args:
            user_id: User identifier
            grace_period_days: Grace period after due date

        Returns:
            True if overdue (accounting for grace period)
        """
        days_until = self.get_days_until_next(user_id)
        if days_until is None:
            return False

        return days_until < -grace_period_days

    def calculate_adherence_rate(self, user_id: str) -> Optional[float]:
        """
        Calculate adherence rate (on-time appointments / total expected).

        Args:
            user_id: User identifier

        Returns:
            Adherence rate (0.0 - 1.0) or None
        """
        if user_id not in self.records:
            return None

        record = self.records[user_id]

        if not record["history"] or record["interval_days"] == 0:
            return None

        # Calculate expected appointments
        start_date = datetime.fromisoformat(record["start_date"])
        days_elapsed = (datetime.now() - start_date).days
        expected_appointments = days_elapsed / record["interval_days"]

        if expected_appointments < 1:
            return 1.0  # Too early to evaluate

        # Actual appointments
        actual_appointments = len(record["history"])

        # Adherence rate
        adherence = min(actual_appointments / expected_appointments, 1.0)

        return adherence

    def get_user_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive adherence summary for user.

        Args:
            user_id: User identifier

        Returns:
            Summary dictionary or None
        """
        if user_id not in self.records:
            return None

        record = self.records[user_id]
        next_date = self.get_next_appointment_date(user_id)
        days_until = self.get_days_until_next(user_id)
        adherence_rate = self.calculate_adherence_rate(user_id)
        overdue = self.is_overdue(user_id)

        return {
            "user_id": user_id,
            "method": record["method"],
            "start_date": record["start_date"],
            "preferred_channel": record["preferred_channel"],
            "interval_days": record["interval_days"],
            "total_appointments": len(record["history"]),
            "next_appointment_date": next_date,
            "days_until_next": days_until,
            "adherence_rate": adherence_rate,
            "is_overdue": overdue,
            "last_appointment": record["history"][-1] if record["history"] else None
        }

    def get_users_needing_reminder(
        self,
        reminder_window_days: int = 14
    ) -> List[Dict[str, Any]]:
        """
        Get users who need reminders (within reminder window).

        Args:
            reminder_window_days: Days before appointment to send reminder

        Returns:
            List of user summaries needing reminders
        """
        users_needing_reminder = []

        for user_id in self.records:
            days_until = self.get_days_until_next(user_id)

            if days_until is not None and 0 <= days_until <= reminder_window_days:
                summary = self.get_user_summary(user_id)
                if summary:
                    users_needing_reminder.append(summary)

        return users_needing_reminder

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries for all tracked users."""
        summaries = []
        for user_id in self.records:
            summary = self.get_user_summary(user_id)
            if summary:
                summaries.append(summary)
        return summaries


# Example usage
if __name__ == "__main__":
    tracker = AdherenceTracker()

    # Start tracking for a user
    tracker.start_tracking(
        user_id="user_001",
        method="DMPA",
        preferred_channel="SMS reminder"
    )

    # Record an appointment
    tracker.record_appointment(
        user_id="user_001",
        notes="First DMPA injection"
    )

    # Get user summary
    summary = tracker.get_user_summary("user_001")
    print(json.dumps(summary, indent=2))

    # Get users needing reminders
    reminders = tracker.get_users_needing_reminder(reminder_window_days=14)
    print(f"\nUsers needing reminders: {len(reminders)}")
