"""
Transform Adherence Dataset for Experiment 5

Transforms existing adherence_dataset.json into the format required for
the LinUCB RL experiment, adding response patterns and context features.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List

# Set seed for reproducibility
random.seed(42)


class AdherenceDataTransformer:
    """Transform adherence data for RL experiment."""

    def __init__(self):
        self.communication_channels = [
            "SMS reminder",
            "Phone call",
            "WhatsApp message",
            "Community health worker visit"
        ]

    def map_channel(self, original_channel: str) -> str:
        """Map original channel names to standard names."""
        channel_map = {
            "sms": "SMS reminder",
            "phone_call": "Phone call",
            "whatsapp": "WhatsApp message",
            "chw": "Community health worker visit",
            "community_health_worker": "Community health worker visit"
        }
        return channel_map.get(original_channel.lower(), "SMS reminder")

    def generate_response_pattern(self, base_probability: float, preferred_channel: str) -> Dict[str, float]:
        """Generate response probabilities for each communication channel."""

        response_pattern = {}

        for channel in self.communication_channels:
            if channel == preferred_channel:
                # Preferred channel has higher response rate
                prob = min(base_probability + random.uniform(0.1, 0.3), 0.95)
            else:
                # Other channels have lower response rate
                prob = max(base_probability - random.uniform(0.1, 0.3), 0.05)

            # Add some randomness
            prob = prob + random.uniform(-0.05, 0.05)
            prob = max(0.0, min(1.0, prob))  # Clamp to [0, 1]

            response_pattern[channel] = round(prob, 3)

        return response_pattern

    def transform_scenario(self, scenario: Dict[str, Any], scenario_id: int) -> Dict[str, Any]:
        """Transform a single scenario into RL format."""

        # Extract key information
        method = scenario.get('method', 'DMPA')
        days_since = scenario.get('days_since_last_injection', 0)
        preferred_channel = self.map_channel(scenario.get('preferred_channel', 'sms'))
        base_probability = scenario.get('response_probability', 0.5)
        previous_adherence = scenario.get('previous_adherence_rate', 0.5)

        # Calculate age (if not present, generate)
        age = random.randint(18, 45)

        # Calculate previous reminders based on days since injection
        # Assume reminders every 2 weeks for DMPA
        previous_reminders = int(days_since / 14) if method == "DMPA" else random.randint(0, 5)

        # Generate response pattern for all channels
        response_pattern = self.generate_response_pattern(base_probability, preferred_channel)

        return {
            "user_id": f"user_{scenario_id:04d}",
            "method": method,
            "days_since_injection": days_since,
            "age": age,
            "preferred_channel": preferred_channel,
            "past_response_rate": round(previous_adherence, 3),
            "previous_reminders": previous_reminders,
            "response_pattern": response_pattern
        }

    def transform_dataset(self, input_file: str, output_file: str, num_users: int = 1000):
        """Transform full adherence dataset."""

        print(f"Loading adherence dataset from: {input_file}")

        # Load original dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        print(f"Original dataset size: {len(original_data)} scenarios")

        # Transform scenarios
        transformed_users = []

        # Use all available scenarios, repeat if needed to reach num_users
        for i in range(num_users):
            source_scenario = original_data[i % len(original_data)]
            transformed = self.transform_scenario(source_scenario, i)
            transformed_users.append(transformed)

            if (i + 1) % 100 == 0:
                print(f"Transformed {i + 1}/{num_users} users...")

        # Calculate statistics
        num_dmpa = sum(1 for u in transformed_users if u['method'] == 'DMPA')
        avg_days_since = sum(u['days_since_injection'] for u in transformed_users) / len(transformed_users)
        avg_age = sum(u['age'] for u in transformed_users) / len(transformed_users)

        # Count preferred channels
        channel_counts = {}
        for user in transformed_users:
            channel = user['preferred_channel']
            channel_counts[channel] = channel_counts.get(channel, 0) + 1

        dataset = {
            "metadata": {
                "num_users": len(transformed_users),
                "num_dmpa_users": num_dmpa,
                "avg_days_since_injection": round(avg_days_since, 2),
                "avg_age": round(avg_age, 1),
                "channel_distribution": channel_counts,
                "communication_channels": self.communication_channels,
                "purpose": "LinUCB contextual bandit adherence optimization"
            },
            "users": transformed_users
        }

        # Save transformed dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"\nDataset saved to: {output_path}")
        print(f"Transformed {len(transformed_users)} users")
        print(f"DMPA users: {num_dmpa} ({num_dmpa/len(transformed_users)*100:.1f}%)")
        print(f"Average days since injection: {avg_days_since:.1f}")
        print(f"Average age: {avg_age:.1f}")
        print(f"Channel distribution:")
        for channel, count in channel_counts.items():
            print(f"   {channel}: {count} ({count/len(transformed_users)*100:.1f}%)")

        # Show sample
        print(f"\nSample user:")
        sample = transformed_users[0]
        print(f"   User ID: {sample['user_id']}")
        print(f"   Method: {sample['method']}")
        print(f"   Days since injection: {sample['days_since_injection']}")
        print(f"   Age: {sample['age']}")
        print(f"   Preferred channel: {sample['preferred_channel']}")
        print(f"   Past response rate: {sample['past_response_rate']}")
        print(f"   Response pattern:")
        for channel, prob in sample['response_pattern'].items():
            print(f"      {channel}: {prob:.3f}")

        return dataset


def main():
    """Transform adherence dataset."""
    print("Transforming Adherence Dataset for Experiment 5...\n")

    transformer = AdherenceDataTransformer()

    input_file = "data/synthetic/adherence_dataset.json"
    output_file = "data/synthetic/adherence_simulation.json"

    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        return

    dataset = transformer.transform_dataset(
        input_file=input_file,
        output_file=output_file,
        num_users=1000
    )

    print(f"\nTransformation complete!")


if __name__ == "__main__":
    main()
