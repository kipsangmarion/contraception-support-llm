"""
Generate synthetic datasets for experiments.
Creates user profiles, adherence scenarios, and evaluation questions.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.reproducibility import load_config, set_random_seeds
from src.utils.logger import setup_logger

logger = setup_logger()


class SyntheticDataGenerator:
    """Generates synthetic data for experiments."""

    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        self.synthetic_config = config.get('synthetic_data', {})

    def generate_user_profiles(self, n_profiles: int = None) -> List[Dict]:
        """
        Generate synthetic user profiles.

        Args:
            n_profiles: Number of profiles to generate

        Returns:
            List of user profile dictionaries
        """
        if n_profiles is None:
            n_profiles = self.synthetic_config.get('user_profiles', {}).get('n_profiles', 500)

        logger.info(f"Generating {n_profiles} user profiles")

        age_groups = self.synthetic_config['user_profiles']['age_groups']
        concerns = self.synthetic_config['user_profiles']['concerns']
        languages = self.synthetic_config['user_profiles'].get('languages', ['english', 'french', 'kinyarwanda'])

        profiles = []

        for i in range(n_profiles):
            profile = {
                'profile_id': f"user_{i:05d}",
                'age_group': random.choice(age_groups),
                'prior_contraceptive_use': random.choice([
                    None,
                    'pill',
                    'condoms',
                    'IUD',
                    'implant',
                    'injection',
                    'emergency_contraception'
                ]),
                'primary_concerns': random.sample(concerns, k=random.randint(1, 3)),
                'language_preference': random.choice(languages),
                'communication_channel': random.choice(['sms', 'whatsapp', 'app', 'phone_call']),
                'relationship_status': random.choice(['single', 'partnered', 'married']),
                'has_children': random.choice([True, False]),
                'education_level': random.choice(['primary', 'secondary', 'tertiary']),
                'urban_rural': random.choice(['urban', 'rural']),
            }
            profiles.append(profile)

        logger.info(f"Generated {len(profiles)} user profiles")
        return profiles

    def generate_adherence_dataset(self, n_samples: int = None) -> List[Dict]:
        """
        Generate synthetic adherence scenarios for RL experiments.

        Args:
            n_samples: Number of scenarios to generate

        Returns:
            List of adherence scenario dictionaries
        """
        if n_samples is None:
            n_samples = self.synthetic_config.get('adherence_dataset', {}).get('n_samples', 1000)

        logger.info(f"Generating {n_samples} adherence scenarios")

        methods = self.synthetic_config['adherence_dataset']['methods']
        channels = self.synthetic_config['adherence_dataset']['channels']

        scenarios = []

        for i in range(n_samples):
            method = random.choice(methods)

            # Different methods have different schedules
            if method == 'DMPA':
                injection_interval_days = 90  # 3 months
            elif method == 'pill':
                injection_interval_days = 1  # daily
            elif method == 'implant':
                injection_interval_days = 1825  # 5 years
            elif method == 'IUD':
                injection_interval_days = 1825  # 5 years
            else:
                injection_interval_days = 30

            # Random last injection time (0-180 days ago)
            days_since_last = random.randint(0, 180)

            # Calculate time until next dose
            days_until_next = injection_interval_days - days_since_last

            # Response behavior pattern
            behavior_pattern = random.choice([
                'compliant',      # Usually responds to reminders
                'inconsistent',   # Sometimes responds
                'non_responsive'  # Rarely responds
            ])

            # Response probability based on pattern
            if behavior_pattern == 'compliant':
                response_prob = random.uniform(0.7, 0.95)
            elif behavior_pattern == 'inconsistent':
                response_prob = random.uniform(0.3, 0.6)
            else:
                response_prob = random.uniform(0.0, 0.3)

            scenario = {
                'scenario_id': f"scenario_{i:05d}",
                'method': method,
                'injection_interval_days': injection_interval_days,
                'days_since_last_injection': days_since_last,
                'days_until_next': days_until_next,
                'preferred_channel': random.choice(channels),
                'behavior_pattern': behavior_pattern,
                'response_probability': round(response_prob, 3),
                'time_of_day_preference': random.choice(['morning', 'afternoon', 'evening']),
                'previous_adherence_rate': round(random.uniform(0.3, 1.0), 3)
            }
            scenarios.append(scenario)

        logger.info(f"Generated {len(scenarios)} adherence scenarios")
        return scenarios

    def generate_evaluation_questions(self, n_questions: int = 100) -> List[Dict]:
        """
        Generate evaluation questions based on contraception guidelines.

        Args:
            n_questions: Number of questions to generate

        Returns:
            List of question dictionaries with ground truth answers
        """
        logger.info(f"Generating {n_questions} evaluation questions")

        # Sample questions covering different topics
        question_templates = [
            # Effectiveness questions
            {
                'question': 'What is the typical-use effectiveness rate of the combined oral contraceptive pill?',
                'ground_truth': 'The typical-use effectiveness of the combined oral contraceptive pill is approximately 91%, meaning about 9 out of 100 women may become pregnant in the first year of use.',
                'category': 'effectiveness',
                'difficulty': 'easy'
            },
            {
                'question': 'How effective is the copper IUD for emergency contraception?',
                'ground_truth': 'The copper IUD is more than 99% effective when inserted within 5 days of unprotected intercourse for emergency contraception.',
                'category': 'effectiveness',
                'difficulty': 'medium'
            },
            # Side effects questions
            {
                'question': 'What are common side effects of DMPA (Depo-Provera) injection?',
                'ground_truth': 'Common side effects of DMPA include irregular bleeding or spotting, weight gain, headaches, and decreased bone density with long-term use. Most women experience changes in menstrual bleeding patterns.',
                'category': 'side_effects',
                'difficulty': 'easy'
            },
            {
                'question': 'Can hormonal contraceptives cause mood changes?',
                'ground_truth': 'Some women report mood changes with hormonal contraceptives, though research findings are mixed. If mood changes occur, they should be discussed with a healthcare provider.',
                'category': 'side_effects',
                'difficulty': 'medium'
            },
            # Eligibility questions
            {
                'question': 'Can a woman who is breastfeeding use the combined oral contraceptive pill?',
                'ground_truth': 'Combined oral contraceptives are generally not recommended for breastfeeding women in the first 6 months postpartum as they may affect milk supply. Progestin-only methods are preferred.',
                'category': 'eligibility',
                'difficulty': 'medium'
            },
            # Method-specific questions
            {
                'question': 'How often should DMPA injections be administered?',
                'ground_truth': 'DMPA injections should be administered every 12-13 weeks (approximately 3 months).',
                'category': 'administration',
                'difficulty': 'easy'
            },
            {
                'question': 'How long does it take for fertility to return after stopping the implant?',
                'ground_truth': 'Fertility typically returns quickly after implant removal, usually within 1-3 months.',
                'category': 'fertility_return',
                'difficulty': 'easy'
            },
            # Emergency contraception
            {
                'question': 'Within what timeframe should emergency contraceptive pills be taken after unprotected intercourse?',
                'ground_truth': 'Emergency contraceptive pills should be taken as soon as possible, ideally within 72-120 hours depending on the type, though they are most effective when taken within the first 24 hours.',
                'category': 'emergency_contraception',
                'difficulty': 'medium'
            },
        ]

        # Expand questions by varying them
        questions = []
        for i in range(n_questions):
            template = random.choice(question_templates)

            question = {
                'question_id': f"q_{i:05d}",
                'question': template['question'],
                'ground_truth': template['ground_truth'],
                'category': template['category'],
                'difficulty': template['difficulty'],
                'expects_safety_fallback': False  # Whether answer should include "consult healthcare provider"
            }
            questions.append(question)

        logger.info(f"Generated {len(questions)} evaluation questions")
        return questions


def main():
    """Generate all synthetic datasets."""

    logger.info("="*60)
    logger.info("Generating Synthetic Datasets")
    logger.info("="*60)

    # Set random seeds
    seeds = set_random_seeds()
    logger.info(f"Random seeds set: {seeds}")

    # Load configuration
    config = load_config()

    # Initialize generator
    generator = SyntheticDataGenerator(config)

    # Output paths
    output_dir = Path(config['paths']['data']['synthetic'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate user profiles
    profiles = generator.generate_user_profiles()
    profiles_path = output_dir / 'user_profiles.json'
    with open(profiles_path, 'w') as f:
        json.dump(profiles, f, indent=2)
    logger.info(f"Saved user profiles to {profiles_path}")

    # Generate adherence dataset
    adherence_data = generator.generate_adherence_dataset()
    adherence_path = output_dir / 'adherence_dataset.json'
    with open(adherence_path, 'w') as f:
        json.dump(adherence_data, f, indent=2)
    logger.info(f"Saved adherence dataset to {adherence_path}")

    # Generate evaluation questions
    eval_questions = generator.generate_evaluation_questions()
    eval_path = output_dir / 'eval_questions.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_questions, f, indent=2)
    logger.info(f"Saved evaluation questions to {eval_path}")

    logger.info("="*60)
    logger.info("Synthetic Data Generation Complete!")
    logger.info("="*60)
    logger.info(f"User profiles: {len(profiles)}")
    logger.info(f"Adherence scenarios: {len(adherence_data)}")
    logger.info(f"Evaluation questions: {len(eval_questions)}")


if __name__ == "__main__":
    main()
