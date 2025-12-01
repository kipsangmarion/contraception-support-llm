"""
Generate Multi-Session Scenarios for Experiment 4B

Creates synthetic user scenarios with multiple sessions over time to test
memory strategies (no memory, full memory, summarized memory).
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Set seed for reproducibility
random.seed(42)


class MultiSessionGenerator:
    """Generate multi-session user scenarios for memory strategy testing."""

    def __init__(self):
        self.methods = ["IUD", "implant", "DMPA injection", "pill", "condoms", "emergency contraception"]
        self.concerns = ["bleeding patterns", "fertility return", "side effects", "effectiveness",
                        "privacy", "partner concerns", "cost", "availability"]
        self.ages = list(range(18, 45))
        self.locations = ["urban", "rural", "peri-urban"]
        self.education_levels = ["primary", "secondary", "tertiary"]
        self.relationship_status = ["single", "married", "partnered"]

    def generate_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Generate a user profile."""
        return {
            "user_id": f"user_{user_id:03d}",
            "age": random.choice(self.ages),
            "location": random.choice(self.locations),
            "education": random.choice(self.education_levels),
            "relationship_status": random.choice(self.relationship_status),
            "primary_concern": random.choice(self.concerns),
            "preferred_method": random.choice(self.methods),
            "num_children": random.randint(0, 4)
        }

    def generate_session(self, session_num: int, user_profile: Dict[str, Any],
                        previous_topics: List[str], base_date: datetime) -> Dict[str, Any]:
        """Generate a single session with conversation turns."""

        # Calculate session date (sessions are 1-4 weeks apart)
        days_offset = sum([random.randint(7, 28) for _ in range(session_num)])
        session_date = base_date + timedelta(days=days_offset)

        turns = []
        num_turns = random.randint(3, 8)

        # First session: Introduction
        if session_num == 0:
            turns.append({
                "turn_number": 1,
                "user_message": f"Hi, I'm {user_profile['age']} years old and I'd like to learn about contraception options.",
                "is_recall_test": False,
                "establishes_fact": f"age: {user_profile['age']}"
            })

            turns.append({
                "turn_number": 2,
                "user_message": f"I'm particularly concerned about {user_profile['primary_concern']}.",
                "is_recall_test": False,
                "establishes_fact": f"primary_concern: {user_profile['primary_concern']}"
            })

            turns.append({
                "turn_number": 3,
                "user_message": f"I'm interested in {user_profile['preferred_method']}. What can you tell me about it?",
                "is_recall_test": False,
                "establishes_fact": f"preferred_method: {user_profile['preferred_method']}"
            })

            # Additional questions for first session
            first_session_questions = [
                "How effective is this method?",
                "What are the common side effects?",
                "How long does it take to start working?",
                "Is it reversible if I want to get pregnant later?",
                "How much does it cost?"
            ]

            for i, question in enumerate(first_session_questions[:num_turns - 3], start=4):
                turns.append({
                    "turn_number": i,
                    "user_message": question,
                    "is_recall_test": False
                })

        # Later sessions: Include recall tests
        else:
            # Start with a recall test
            recall_options = [
                {
                    "message": "Hi again! Do you remember what my main concern was?",
                    "expected": user_profile['primary_concern']
                },
                {
                    "message": f"Hello, it's me again. What method was I interested in last time?",
                    "expected": user_profile['preferred_method']
                },
                {
                    "message": "Hi! Can you remind me what we discussed in our previous sessions?",
                    "expected": f"{user_profile['preferred_method']}, {user_profile['primary_concern']}"
                }
            ]

            recall = random.choice(recall_options)
            turns.append({
                "turn_number": 1,
                "user_message": recall["message"],
                "is_recall_test": True,
                "expected_recall": recall["expected"]
            })

            # Follow-up questions based on session progression
            if session_num == 1:
                # Second session: follow-up questions
                follow_up_questions = [
                    "I've been thinking more about the side effects. Can you explain them again?",
                    "How soon can I start this method?",
                    "Do I need any medical tests before starting?",
                    "What happens if I miss a dose or appointment?",
                    "Can I switch methods later if this doesn't work for me?"
                ]
            elif session_num == 2:
                # Third session: practical concerns
                follow_up_questions = [
                    "I'm ready to start. What are the next steps?",
                    "Where can I get this method in my area?",
                    "Do I need my partner's consent?",
                    "How often will I need follow-up appointments?",
                    "What should I do if I experience side effects?"
                ]
            elif session_num == 3:
                # Fourth session: started using method
                follow_up_questions = [
                    "I started using the method. It's been going well so far.",
                    "I had some questions about the side effects I'm experiencing.",
                    "When should I come back for a check-up?",
                    "Is it normal to have irregular bleeding at first?",
                    "Can I exercise normally with this method?"
                ]
            else:
                # Fifth+ session: long-term questions
                follow_up_questions = [
                    "I've been using this for a while now. Everything is good.",
                    "I'm thinking about switching methods. What are my options?",
                    "When can I stop using contraception if I want to get pregnant?",
                    "Are there any long-term effects I should know about?",
                    "Do I need to continue with regular check-ups?"
                ]

            # Add follow-up questions
            for i, question in enumerate(random.sample(follow_up_questions, min(num_turns - 1, len(follow_up_questions))), start=2):
                turns.append({
                    "turn_number": i,
                    "user_message": question,
                    "is_recall_test": False
                })

            # Add a personalization test in later sessions
            if session_num >= 2 and len(turns) < num_turns:
                personalization_tests = [
                    {
                        "message": f"Given my situation with {user_profile['num_children']} children, what do you recommend?",
                        "expected": f"{user_profile['num_children']} children"
                    },
                    {
                        "message": f"Considering I live in a {user_profile['location']} area, how does that affect my options?",
                        "expected": user_profile['location']
                    }
                ]

                if len(turns) < num_turns:
                    pers_test = random.choice(personalization_tests)
                    turns.append({
                        "turn_number": len(turns) + 1,
                        "user_message": pers_test["message"],
                        "is_recall_test": True,
                        "expected_recall": pers_test["expected"]
                    })

        return {
            "session_number": session_num + 1,
            "date": session_date.strftime("%Y-%m-%d"),
            "num_turns": len(turns),
            "turns": turns
        }

    def generate_scenario(self, user_id: int, num_sessions: int) -> Dict[str, Any]:
        """Generate a complete multi-session scenario for one user."""

        user_profile = self.generate_user_profile(user_id)
        base_date = datetime(2024, 1, 1)

        sessions = []
        previous_topics = []

        for session_num in range(num_sessions):
            session = self.generate_session(session_num, user_profile, previous_topics, base_date)
            sessions.append(session)

            # Track topics discussed
            for turn in session['turns']:
                if not turn['is_recall_test']:
                    previous_topics.append(turn['user_message'])

        return {
            "user_id": user_profile["user_id"],
            "user_profile": user_profile,
            "num_sessions": len(sessions),
            "total_turns": sum(s['num_turns'] for s in sessions),
            "total_recall_tests": sum(
                sum(1 for t in s['turns'] if t.get('is_recall_test', False))
                for s in sessions
            ),
            "sessions": sessions
        }

    def generate_dataset(self, num_scenarios: int = 10, min_sessions: int = 3, max_sessions: int = 5) -> Dict[str, Any]:
        """Generate full dataset of multi-session scenarios."""
        scenarios = []

        for i in range(num_scenarios):
            num_sessions = random.randint(min_sessions, max_sessions)
            scenario = self.generate_scenario(i, num_sessions)
            scenarios.append(scenario)
            print(f"Generated scenario {i+1}/{num_scenarios}: User {scenario['user_id']}, {num_sessions} sessions, {scenario['total_turns']} turns")

        return {
            "metadata": {
                "num_scenarios": num_scenarios,
                "min_sessions": min_sessions,
                "max_sessions": max_sessions,
                "total_sessions": sum(s['num_sessions'] for s in scenarios),
                "total_turns": sum(s['total_turns'] for s in scenarios),
                "total_recall_tests": sum(s['total_recall_tests'] for s in scenarios),
                "purpose": "Test multi-session memory strategies (no memory, full memory, summarized memory)"
            },
            "scenarios": scenarios
        }


def main():
    """Generate multi-session scenarios dataset."""
    print("Generating Multi-Session Scenarios for Experiment 4B...")

    generator = MultiSessionGenerator()
    dataset = generator.generate_dataset(
        num_scenarios=10,
        min_sessions=3,
        max_sessions=5
    )

    # Save dataset
    output_path = Path("data/synthetic/multi_session_scenarios.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nDataset saved to: {output_path}")
    print(f"Generated {dataset['metadata']['num_scenarios']} scenarios")
    print(f"Total sessions: {dataset['metadata']['total_sessions']}")
    print(f"Total turns: {dataset['metadata']['total_turns']}")
    print(f"Total recall tests: {dataset['metadata']['total_recall_tests']}")

    # Show sample
    print(f"\nSample scenario:")
    sample = dataset['scenarios'][0]
    print(f"   User ID: {sample['user_id']}")
    print(f"   Sessions: {sample['num_sessions']}")
    print(f"   Total turns: {sample['total_turns']}")
    print(f"   Recall tests: {sample['total_recall_tests']}")
    print(f"   User profile: age {sample['user_profile']['age']}, {sample['user_profile']['preferred_method']}")


if __name__ == "__main__":
    main()
