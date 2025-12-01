"""
Generate Long-Session Conversations for Experiment 4A

Creates synthetic conversations with 20-40 turns to test memory consistency,
contradiction detection, and recall accuracy.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Set seed for reproducibility
random.seed(42)


class LongSessionGenerator:
    """Generate long conversation scenarios for memory testing."""

    def __init__(self):
        self.methods = ["IUD", "implant", "DMPA injection", "pill", "condoms"]
        self.concerns = ["bleeding", "fertility", "side effects", "effectiveness", "privacy", "partner concerns"]
        self.ages = list(range(18, 45))
        self.locations = ["urban", "rural", "peri-urban"]

    def generate_conversation(self, conv_id: int, num_turns: int) -> Dict[str, Any]:
        """Generate a single long conversation with recall tests."""

        # User facts to establish early
        user_age = random.choice(self.ages)
        user_method = random.choice(self.methods)
        user_concern = random.choice(self.concerns)
        user_location = random.choice(self.locations)
        user_children = random.randint(0, 4)

        user_facts = {
            "age": user_age,
            "preferred_method": user_method,
            "main_concern": user_concern,
            "location": user_location,
            "number_of_children": user_children
        }

        turns = []

        # Turn 1-5: Establish user facts
        turns.append({
            "turn_number": 1,
            "user_message": f"Hi, I'm {user_age} years old and interested in contraception options.",
            "is_recall_test": False,
            "establishes_fact": f"age: {user_age}"
        })

        turns.append({
            "turn_number": 2,
            "user_message": f"I'm particularly interested in learning about {user_method}.",
            "is_recall_test": False,
            "establishes_fact": f"preferred_method: {user_method}"
        })

        turns.append({
            "turn_number": 3,
            "user_message": f"My main concern is about {user_concern}. Can you tell me more?",
            "is_recall_test": False,
            "establishes_fact": f"main_concern: {user_concern}"
        })

        turns.append({
            "turn_number": 4,
            "user_message": f"I live in a {user_location} area. Is that important?",
            "is_recall_test": False,
            "establishes_fact": f"location: {user_location}"
        })

        turns.append({
            "turn_number": 5,
            "user_message": f"I have {user_children} children already.",
            "is_recall_test": False,
            "establishes_fact": f"number_of_children: {user_children}"
        })

        # Turn 6-15: General questions about contraception
        general_questions = [
            "How effective is this method compared to others?",
            "What are the most common side effects?",
            "How long does it take to work?",
            "Can I switch methods later if I want?",
            "Will this affect my menstrual cycle?",
            "How much does this typically cost?",
            "Do I need my partner's consent?",
            "How soon can I start this method?",
            "What if I want to get pregnant later?",
            "Are there any health risks I should know about?"
        ]

        for i, question in enumerate(general_questions[:min(10, num_turns - 15)], start=6):
            turns.append({
                "turn_number": i,
                "user_message": question,
                "is_recall_test": False
            })

        # Turn 16+: Recall tests scattered throughout
        current_turn = len(turns) + 1

        # Recall test 1: Age
        if current_turn <= num_turns - 8:
            turns.append({
                "turn_number": current_turn,
                "user_message": "By the way, what was my age that I mentioned earlier?",
                "is_recall_test": True,
                "expected_fact": str(user_age)
            })
            current_turn += 1

        # Filler question
        if current_turn <= num_turns - 7:
            turns.append({
                "turn_number": current_turn,
                "user_message": "Can you explain the insertion process?",
                "is_recall_test": False
            })
            current_turn += 1

        # Recall test 2: Method
        if current_turn <= num_turns - 6:
            turns.append({
                "turn_number": current_turn,
                "user_message": "What method did I say I was most interested in?",
                "is_recall_test": True,
                "expected_fact": user_method
            })
            current_turn += 1

        # More filler questions
        filler_questions = [
            "How often do I need follow-up appointments?",
            "Can I exercise normally with this method?",
            "Will this affect my weight?",
            "Is this method reversible?",
            "What should I do if I miss a dose or appointment?"
        ]

        for question in filler_questions[:min(5, num_turns - current_turn - 3)]:
            if current_turn <= num_turns - 3:
                turns.append({
                    "turn_number": current_turn,
                    "user_message": question,
                    "is_recall_test": False
                })
                current_turn += 1

        # Recall test 3: Concern
        if current_turn <= num_turns - 2:
            turns.append({
                "turn_number": current_turn,
                "user_message": f"What was the main concern I mentioned about {user_concern}?",
                "is_recall_test": True,
                "expected_fact": user_concern
            })
            current_turn += 1

        # Recall test 4: Location
        if current_turn <= num_turns - 1:
            turns.append({
                "turn_number": current_turn,
                "user_message": "Do you remember if I'm in an urban or rural area?",
                "is_recall_test": True,
                "expected_fact": user_location
            })
            current_turn += 1

        # Final comprehensive recall test
        if current_turn <= num_turns:
            turns.append({
                "turn_number": current_turn,
                "user_message": "Can you summarize what we've discussed about my situation?",
                "is_recall_test": True,
                "expected_fact": f"age {user_age}, {user_method}, {user_concern}"
            })

        # Ensure we have exactly num_turns
        while len(turns) < num_turns:
            turns.append({
                "turn_number": len(turns) + 1,
                "user_message": random.choice([
                    "Thank you for all this information.",
                    "This has been very helpful.",
                    "I appreciate your detailed explanations.",
                    "One more question - is there anything else I should know?"
                ]),
                "is_recall_test": False
            })

        return {
            "conversation_id": f"conv_{conv_id:03d}",
            "num_turns": len(turns),
            "user_facts": user_facts,
            "turns": turns[:num_turns]  # Trim to exact num_turns
        }

    def generate_dataset(self, num_conversations: int = 10, min_turns: int = 20, max_turns: int = 40) -> Dict[str, Any]:
        """Generate full dataset of long conversations."""
        conversations = []

        for i in range(num_conversations):
            num_turns = random.randint(min_turns, max_turns)
            conv = self.generate_conversation(i, num_turns)
            conversations.append(conv)
            print(f"Generated conversation {i+1}/{num_conversations}: {num_turns} turns")

        return {
            "metadata": {
                "num_conversations": num_conversations,
                "min_turns": min_turns,
                "max_turns": max_turns,
                "total_turns": sum(c["num_turns"] for c in conversations),
                "avg_turns": sum(c["num_turns"] for c in conversations) / num_conversations,
                "purpose": "Test long-session memory consistency and recall accuracy"
            },
            "conversations": conversations
        }


def main():
    """Generate long-session conversations dataset."""
    print("Generating Long-Session Conversations for Experiment 4A...")

    generator = LongSessionGenerator()
    dataset = generator.generate_dataset(
        num_conversations=10,
        min_turns=20,
        max_turns=40
    )

    # Save dataset
    output_path = Path("data/synthetic/long_session_conversations.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"📊 Generated {dataset['metadata']['num_conversations']} conversations")
    print(f"📊 Total turns: {dataset['metadata']['total_turns']}")
    print(f"📊 Average turns per conversation: {dataset['metadata']['avg_turns']:.1f}")

    # Show sample
    print(f"\n📝 Sample conversation:")
    sample = dataset['conversations'][0]
    print(f"   ID: {sample['conversation_id']}")
    print(f"   Turns: {sample['num_turns']}")
    print(f"   User facts: {sample['user_facts']}")
    print(f"   First turn: {sample['turns'][0]['user_message']}")
    print(f"   Recall tests: {sum(1 for t in sample['turns'] if t.get('is_recall_test', False))}")


if __name__ == "__main__":
    main()
