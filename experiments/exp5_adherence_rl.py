"""
Experiment 5: Adaptive Adherence Support with Reinforcement Learning

This experiment compares LinUCB contextual bandit algorithm against baseline
strategies (fixed and random) for DMPA reinjection adherence support.

Metrics:
- Cumulative reward over 1,000 rounds
- Regret compared to oracle policy
- Convergence rate
- Strategy effectiveness comparison

Reference: outline.md Section 4, Experiment 5
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LinUCBBandit:
    """
    Linear Upper Confidence Bound (LinUCB) contextual bandit algorithm.

    Reference: Li et al. (2010) "A Contextual-Bandit Approach to Personalized News Article Recommendation"
    """

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        """
        Initialize LinUCB bandit.

        Args:
            n_arms: Number of actions/arms (e.g., communication strategies)
            n_features: Dimensionality of context features
            alpha: Exploration parameter (higher = more exploration)
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha

        # Initialize A (design matrix) and b (reward vector) for each arm
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select arm using UCB policy.

        Args:
            context: Feature vector for current context

        Returns:
            Selected arm index
        """
        p_values = []

        for arm in range(self.n_arms):
            # Compute theta (parameter estimate)
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv.dot(self.b[arm])

            # Compute UCB
            p = theta.T.dot(context) + self.alpha * np.sqrt(context.T.dot(A_inv).dot(context))
            p_values.append(p)

        return int(np.argmax(p_values))

    def update(self, arm: int, context: np.ndarray, reward: float):
        """
        Update arm parameters after observing reward.

        Args:
            arm: Selected arm index
            context: Feature vector used
            reward: Observed reward
        """
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context


class FixedStrategy:
    """Baseline: Always use the same strategy."""

    def __init__(self, n_arms: int, fixed_arm: int = 0):
        self.n_arms = n_arms
        self.fixed_arm = fixed_arm

    def select_arm(self, context: np.ndarray) -> int:
        return self.fixed_arm

    def update(self, arm: int, context: np.ndarray, reward: float):
        pass  # No learning


class RandomStrategy:
    """Baseline: Random strategy selection."""

    def __init__(self, n_arms: int):
        self.n_arms = n_arms

    def select_arm(self, context: np.ndarray) -> int:
        return np.random.randint(0, self.n_arms)

    def update(self, arm: int, context: np.ndarray, reward: float):
        pass  # No learning


class AdherenceRLExperiment:
    """Experiment comparing RL vs baseline adherence strategies."""

    def __init__(self):
        # Communication strategies
        self.strategies = [
            "SMS reminder",
            "Phone call",
            "WhatsApp message",
            "Community health worker visit"
        ]
        self.n_arms = len(self.strategies)

        self.results = {
            "linucb": [],
            "fixed": [],
            "random": []
        }

    def extract_features(self, user_context: Dict[str, Any]) -> np.ndarray:
        """
        Extract feature vector from user context.

        Features:
        - Days since last injection (normalized)
        - Past response rate (0-1)
        - Preferred channel (one-hot)
        - Age group (normalized)
        - Number of previous reminders
        """
        features = []

        # Days since last injection (normalized to 0-1, assuming 90 days max)
        days_since = user_context.get('days_since_injection', 0)
        features.append(min(days_since / 90.0, 1.0))

        # Past response rate
        features.append(user_context.get('past_response_rate', 0.5))

        # Preferred channel (one-hot encoding)
        preferred = user_context.get('preferred_channel', 'SMS')
        for strategy in self.strategies:
            features.append(1.0 if preferred in strategy else 0.0)

        # Age group (normalized: 15-49 -> 0-1)
        age = user_context.get('age', 25)
        features.append((age - 15) / 34.0)

        # Number of previous reminders (normalized, max 10)
        prev_reminders = user_context.get('previous_reminders', 0)
        features.append(min(prev_reminders / 10.0, 1.0))

        return np.array(features)

    def simulate_reward(self, user_context: Dict[str, Any], arm: int) -> float:
        """
        Simulate user response to strategy.

        Reward:
        - 1.0 if user responds positively
        - 0.0 if user doesn't respond
        """
        # Get ground truth response pattern
        response_pattern = user_context.get('response_pattern', {})
        base_response_rate = response_pattern.get(self.strategies[arm], 0.5)

        # Add some randomness
        noise = np.random.normal(0, 0.1)
        response_prob = np.clip(base_response_rate + noise, 0, 1)

        # Simulate response
        reward = 1.0 if np.random.random() < response_prob else 0.0

        return reward

    def run_simulation(self, algorithm, dataset: List[Dict[str, Any]], n_rounds: int = 1000) -> Dict[str, Any]:
        """Run simulation for one algorithm."""
        rewards = []
        arms_selected = []
        cumulative_reward = 0

        for round_idx in range(n_rounds):
            # Sample user from dataset
            user = dataset[round_idx % len(dataset)]

            # Extract context features
            context = self.extract_features(user)

            # Select arm
            arm = algorithm.select_arm(context)
            arms_selected.append(arm)

            # Get reward
            reward = self.simulate_reward(user, arm)
            rewards.append(reward)
            cumulative_reward += reward

            # Update algorithm
            algorithm.update(arm, context, reward)

        return {
            "rewards": rewards,
            "cumulative_reward": cumulative_reward,
            "avg_reward": cumulative_reward / n_rounds,
            "arms_selected": arms_selected
        }

    def run_experiment(self, dataset_file: str, output_dir: str = "results/tables", n_rounds: int = 1000):
        """Run adherence RL experiment."""
        logger.info(f"Starting Experiment 5: Adaptive Adherence RL")

        # Load dataset
        dataset_path = Path(dataset_file)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_file}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset = data if isinstance(data, list) else data.get('users', [])
        logger.info(f"Loaded {len(dataset)} user scenarios")

        # Feature dimensionality
        sample_features = self.extract_features(dataset[0])
        n_features = len(sample_features)
        logger.info(f"Feature dimensionality: {n_features}")

        # Initialize algorithms
        linucb = LinUCBBandit(n_arms=self.n_arms, n_features=n_features, alpha=1.0)
        fixed = FixedStrategy(n_arms=self.n_arms, fixed_arm=0)  # Always SMS
        random = RandomStrategy(n_arms=self.n_arms)

        # Run simulations
        logger.info(f"Running LinUCB simulation ({n_rounds} rounds)...")
        linucb_results = self.run_simulation(linucb, dataset, n_rounds)
        self.results['linucb'] = linucb_results

        logger.info(f"Running Fixed strategy simulation ({n_rounds} rounds)...")
        fixed_results = self.run_simulation(fixed, dataset, n_rounds)
        self.results['fixed'] = fixed_results

        logger.info(f"Running Random strategy simulation ({n_rounds} rounds)...")
        random_results = self.run_simulation(random, dataset, n_rounds)
        self.results['random'] = random_results

        # Calculate regret (compared to best fixed strategy)
        best_fixed_reward = max([
            self.run_simulation(FixedStrategy(self.n_arms, arm), dataset, n_rounds)['cumulative_reward']
            for arm in range(self.n_arms)
        ])

        linucb_regret = best_fixed_reward - linucb_results['cumulative_reward']
        fixed_regret = best_fixed_reward - fixed_results['cumulative_reward']
        random_regret = best_fixed_reward - random_results['cumulative_reward']

        # Statistical tests
        t_linucb_vs_fixed, p_linucb_vs_fixed = stats.ttest_ind(
            linucb_results['rewards'][-100:],  # Last 100 rounds
            fixed_results['rewards'][-100:]
        )

        t_linucb_vs_random, p_linucb_vs_random = stats.ttest_ind(
            linucb_results['rewards'][-100:],
            random_results['rewards'][-100:]
        )

        summary = {
            "experiment": "Experiment 5: Adaptive Adherence RL",
            "n_rounds": n_rounds,
            "n_users": len(dataset),
            "n_strategies": self.n_arms,
            "strategies": self.strategies,

            "linucb": {
                "cumulative_reward": float(linucb_results['cumulative_reward']),
                "avg_reward": float(linucb_results['avg_reward']),
                "regret": float(linucb_regret)
            },

            "fixed_strategy": {
                "cumulative_reward": float(fixed_results['cumulative_reward']),
                "avg_reward": float(fixed_results['avg_reward']),
                "regret": float(fixed_regret)
            },

            "random_strategy": {
                "cumulative_reward": float(random_results['cumulative_reward']),
                "avg_reward": float(random_results['avg_reward']),
                "regret": float(random_regret)
            },

            "comparison": {
                "linucb_vs_fixed": {
                    "improvement": float(linucb_results['avg_reward'] - fixed_results['avg_reward']),
                    "improvement_percent": float((linucb_results['avg_reward'] - fixed_results['avg_reward']) / fixed_results['avg_reward'] * 100),
                    "t_statistic": float(t_linucb_vs_fixed),
                    "p_value": float(p_linucb_vs_fixed),
                    "significant": bool(p_linucb_vs_fixed < 0.05)
                },
                "linucb_vs_random": {
                    "improvement": float(linucb_results['avg_reward'] - random_results['avg_reward']),
                    "improvement_percent": float((linucb_results['avg_reward'] - random_results['avg_reward']) / random_results['avg_reward'] * 100),
                    "t_statistic": float(t_linucb_vs_random),
                    "p_value": float(p_linucb_vs_random),
                    "significant": bool(p_linucb_vs_random < 0.05)
                }
            },

            "timestamp": datetime.now().isoformat()
        }

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        detailed_file = output_path / "exp5_adherence_rl_detailed.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            # Don't save full reward arrays in summary
            results_for_save = {
                "linucb": {"avg_reward": linucb_results['avg_reward'], "cumulative_reward": linucb_results['cumulative_reward']},
                "fixed": {"avg_reward": fixed_results['avg_reward'], "cumulative_reward": fixed_results['cumulative_reward']},
                "random": {"avg_reward": random_results['avg_reward'], "cumulative_reward": random_results['cumulative_reward']}
            }
            json.dump({
                "summary": summary,
                "results": results_for_save
            }, f, indent=2, ensure_ascii=False)

        # Save summary
        summary_file = output_path / "exp5_adherence_rl_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        # Generate reward curves plot
        self.plot_reward_curves(output_dir)

        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT 5 RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Rounds simulated: {n_rounds}")
        logger.info(f"\nCumulative Rewards:")
        logger.info(f"  LinUCB:  {linucb_results['cumulative_reward']:.1f}")
        logger.info(f"  Fixed:   {fixed_results['cumulative_reward']:.1f}")
        logger.info(f"  Random:  {random_results['cumulative_reward']:.1f}")
        logger.info(f"\nAverage Rewards:")
        logger.info(f"  LinUCB:  {linucb_results['avg_reward']:.4f}")
        logger.info(f"  Fixed:   {fixed_results['avg_reward']:.4f}")
        logger.info(f"  Random:  {random_results['avg_reward']:.4f}")
        logger.info(f"\nLinUCB vs Fixed:")
        logger.info(f"  Improvement: +{summary['comparison']['linucb_vs_fixed']['improvement_percent']:.2f}%")
        logger.info(f"  p-value: {p_linucb_vs_fixed:.4f}")
        logger.info(f"  Significant: {'✓ Yes' if p_linucb_vs_fixed < 0.05 else '✗ No'}")
        logger.info(f"\nResults saved to: {output_path}")
        logger.info(f"{'='*60}\n")

        return summary

    def plot_reward_curves(self, output_dir: str):
        """Generate reward curves visualization."""
        plt.figure(figsize=(12, 6))

        # Calculate cumulative rewards over time
        linucb_cumulative = np.cumsum(self.results['linucb']['rewards'])
        fixed_cumulative = np.cumsum(self.results['fixed']['rewards'])
        random_cumulative = np.cumsum(self.results['random']['rewards'])

        rounds = range(len(linucb_cumulative))

        plt.plot(rounds, linucb_cumulative, label='LinUCB', linewidth=2)
        plt.plot(rounds, fixed_cumulative, label='Fixed Strategy', linewidth=2)
        plt.plot(rounds, random_cumulative, label='Random Strategy', linewidth=2)

        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Cumulative Reward', fontsize=12)
        plt.title('Adherence RL: Cumulative Reward Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = Path(output_dir).parent / "plots" / "exp5_reward_curves.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"[PLOT] Reward curves saved to: {plot_path}")


def main():
    """Run Experiment 5."""
    # Configuration
    DATASET_FILE = "data/synthetic/adherence_simulation.json"
    OUTPUT_DIR = "results/tables"
    N_ROUNDS = 1000

    # Check if data file exists
    if not Path(DATASET_FILE).exists():
        logger.error(f"[FAIL] Data file not found: {DATASET_FILE}")
        logger.error("Please generate adherence simulation dataset first using the data generation script.")
        return

    # Run experiment
    experiment = AdherenceRLExperiment()
    results = experiment.run_experiment(DATASET_FILE, OUTPUT_DIR, N_ROUNDS)

    print("\n[SUCCESS] Experiment 5 complete!")
    print(f"[METRIC] LinUCB avg reward: {results['linucb']['avg_reward']:.4f}")
    print(f"[METRIC] Fixed avg reward: {results['fixed_strategy']['avg_reward']:.4f}")
    print(f"[METRIC] Random avg reward: {results['random_strategy']['avg_reward']:.4f}")
    print(f"[METRIC] LinUCB improvement: +{results['comparison']['linucb_vs_fixed']['improvement_percent']:.2f}%")


if __name__ == "__main__":
    main()
