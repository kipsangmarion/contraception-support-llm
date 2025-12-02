"""
Experiment 4B: Multi-Session Memory Test

This experiment compares different memory strategies across multiple sessions:
1. No memory (each session is independent)
2. Full conversation history (all previous turns)
3. Summarized memory (condensed session summaries)

Metrics:
- Recall accuracy across sessions
- Personalization consistency
- Response quality with different memory strategies
- Statistical comparison of strategies

Reference: outline.md Section 4, Experiment 4B
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime
import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.generator import RAGGenerator
from src.memory.conversation_memory import ConversationMemory
from src.evaluation.metrics import EvaluationMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiSessionMemoryTest:
    """Compare memory strategies across multiple sessions."""

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.generator = RAGGenerator(llm_config={'provider': 'ollama', 'model_name': model_name})
        self.evaluator = EvaluationMetrics()
        self.results = {
            "no_memory": [],
            "full_memory": [],
            "summarized_memory": []
        }

    def run_with_no_memory(self, user_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run sessions with no memory (baseline)."""
        user_id = user_scenario['user_id']
        sessions = user_scenario['sessions']

        recall_tests = []

        for session_idx, session in enumerate(sessions):
            session_id = f"no_mem_{user_id}_s{session_idx}"

            for turn in session['turns']:
                question = turn['user_message']
                is_recall = turn.get('is_recall_test', False)
                expected = turn.get('expected_recall', '')

                # Generate without any memory context
                response = self.generator.generate(
                    query=question,
                    context="",
                    language="english"
                )

                if is_recall:
                    recalled = expected.lower() in response['response'].lower()
                    recall_tests.append({
                        "session": session_idx,
                        "expected": expected,
                        "recalled": recalled
                    })

        recall_accuracy = sum(1 for r in recall_tests if r['recalled']) / len(recall_tests) if recall_tests else 0

        return {
            "user_id": user_id,
            "strategy": "no_memory",
            "num_sessions": len(sessions),
            "num_recall_tests": len(recall_tests),
            "recall_accuracy": recall_accuracy,
            "recall_details": recall_tests
        }

    def run_with_full_memory(self, user_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run sessions with full conversation history."""
        user_id = user_scenario['user_id']
        sessions = user_scenario['sessions']

        memory = ConversationMemory()
        session_id = f"full_mem_{user_id}"

        recall_tests = []

        for session_idx, session in enumerate(sessions):
            for turn in session['turns']:
                question = turn['user_message']
                is_recall = turn.get('is_recall_test', False)
                expected = turn.get('expected_recall', '')

                # Get full conversation history
                history = memory.get_history(session_id)

                # Generate with full history context
                response = self.generator.generate(
                    query=question,
                    context="",
                    language="english",
                    conversation_history=history
                )

                # Update memory with new turn
                memory.add_turn(session_id, question, response['response'])

                if is_recall:
                    recalled = expected.lower() in response['response'].lower()
                    recall_tests.append({
                        "session": session_idx,
                        "expected": expected,
                        "recalled": recalled
                    })

        recall_accuracy = sum(1 for r in recall_tests if r['recalled']) / len(recall_tests) if recall_tests else 0

        return {
            "user_id": user_id,
            "strategy": "full_memory",
            "num_sessions": len(sessions),
            "num_recall_tests": len(recall_tests),
            "recall_accuracy": recall_accuracy,
            "recall_details": recall_tests
        }

    def run_with_summarized_memory(self, user_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run sessions with summarized session history."""
        user_id = user_scenario['user_id']
        sessions = user_scenario['sessions']

        memory = ConversationMemory()
        session_summaries = []

        recall_tests = []

        for session_idx, session in enumerate(sessions):
            session_id = f"summ_mem_{user_id}_s{session_idx}"
            session_turns = []

            for turn in session['turns']:
                question = turn['user_message']
                is_recall = turn.get('is_recall_test', False)
                expected = turn.get('expected_recall', '')

                # Create context from previous session summaries
                summary_context = "\n\n".join([
                    f"Previous session {i+1}: {summ}"
                    for i, summ in enumerate(session_summaries)
                ])

                # Generate response (use summary_context as the context)
                response = self.generator.generate(
                    query=question,
                    context=summary_context if summary_context else "",
                    language="english"
                )

                session_turns.append({
                    "user": question,
                    "assistant": response['response']
                })

                if is_recall:
                    recalled = expected.lower() in response['response'].lower()
                    recall_tests.append({
                        "session": session_idx,
                        "expected": expected,
                        "recalled": recalled
                    })

            # Summarize this session for future use
            summary = memory.summarize_conversation(session_id)
            session_summaries.append(summary)

        recall_accuracy = sum(1 for r in recall_tests if r['recalled']) / len(recall_tests) if recall_tests else 0

        return {
            "user_id": user_id,
            "strategy": "summarized_memory",
            "num_sessions": len(sessions),
            "num_recall_tests": len(recall_tests),
            "recall_accuracy": recall_accuracy,
            "recall_details": recall_tests
        }

    def run_experiment(self, scenarios_file: str, output_dir: str = "results/tables"):
        """Run multi-session memory experiment."""
        logger.info(f"Starting Experiment 4B: Multi-Session Memory Test")

        # Load scenarios
        scenarios_path = Path(scenarios_file)
        if not scenarios_path.exists():
            raise FileNotFoundError(f"Scenarios file not found: {scenarios_file}")

        with open(scenarios_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        scenarios = data if isinstance(data, list) else data.get('scenarios', [])
        logger.info(f"Loaded {len(scenarios)} user scenarios")

        # Run all three strategies
        no_memory_results = []
        full_memory_results = []
        summarized_memory_results = []

        for scenario in scenarios:
            logger.info(f"Testing user {scenario['user_id']} with all 3 strategies...")

            # Strategy 1: No memory
            no_mem = self.run_with_no_memory(scenario)
            no_memory_results.append(no_mem)
            self.results['no_memory'].append(no_mem)

            # Strategy 2: Full memory
            full_mem = self.run_with_full_memory(scenario)
            full_memory_results.append(full_mem)
            self.results['full_memory'].append(full_mem)

            # Strategy 3: Summarized memory
            summ_mem = self.run_with_summarized_memory(scenario)
            summarized_memory_results.append(summ_mem)
            self.results['summarized_memory'].append(summ_mem)

        # Calculate aggregate metrics
        no_mem_accuracy = np.mean([r['recall_accuracy'] for r in no_memory_results])
        full_mem_accuracy = np.mean([r['recall_accuracy'] for r in full_memory_results])
        summ_mem_accuracy = np.mean([r['recall_accuracy'] for r in summarized_memory_results])

        # Statistical tests
        no_mem_scores = [r['recall_accuracy'] for r in no_memory_results]
        full_mem_scores = [r['recall_accuracy'] for r in full_memory_results]
        summ_mem_scores = [r['recall_accuracy'] for r in summarized_memory_results]

        # Paired t-tests
        t_full_vs_none, p_full_vs_none = stats.ttest_rel(full_mem_scores, no_mem_scores)
        t_summ_vs_none, p_summ_vs_none = stats.ttest_rel(summ_mem_scores, no_mem_scores)
        t_full_vs_summ, p_full_vs_summ = stats.ttest_rel(full_mem_scores, summ_mem_scores)

        summary = {
            "experiment": "Experiment 4B: Multi-Session Memory Test",
            "model": self.model_name,
            "num_scenarios": len(scenarios),

            "no_memory": {
                "recall_accuracy": float(no_mem_accuracy),
                "std_dev": float(np.std(no_mem_scores))
            },

            "full_memory": {
                "recall_accuracy": float(full_mem_accuracy),
                "std_dev": float(np.std(full_mem_scores)),
                "improvement_vs_none": float(full_mem_accuracy - no_mem_accuracy),
                "t_vs_none": float(t_full_vs_none),
                "p_vs_none": float(p_full_vs_none),
                "significant_vs_none": bool(p_full_vs_none < 0.05)
            },

            "summarized_memory": {
                "recall_accuracy": float(summ_mem_accuracy),
                "std_dev": float(np.std(summ_mem_scores)),
                "improvement_vs_none": float(summ_mem_accuracy - no_mem_accuracy),
                "t_vs_none": float(t_summ_vs_none),
                "p_vs_none": float(p_summ_vs_none),
                "significant_vs_none": bool(p_summ_vs_none < 0.05)
            },

            "full_vs_summarized": {
                "t_statistic": float(t_full_vs_summ),
                "p_value": float(p_full_vs_summ),
                "significant": bool(p_full_vs_summ < 0.05)
            },

            "timestamp": datetime.now().isoformat()
        }

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        detailed_file = output_path / "exp4b_multi_session_detailed.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary,
                "results": self.results
            }, f, indent=2, ensure_ascii=False)

        # Save summary
        summary_file = output_path / "exp4b_multi_session_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT 4B RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Scenarios tested: {len(scenarios)}")
        logger.info(f"\nMemory Strategy Comparison:")
        logger.info(f"  No Memory:        {no_mem_accuracy:.2%}")
        logger.info(f"  Full Memory:      {full_mem_accuracy:.2%} (Δ +{(full_mem_accuracy - no_mem_accuracy):.2%})")
        logger.info(f"  Summarized:       {summ_mem_accuracy:.2%} (Δ +{(summ_mem_accuracy - no_mem_accuracy):.2%})")
        logger.info(f"\nStatistical Significance:")
        logger.info(f"  Full vs None:     p={p_full_vs_none:.4f} {'✓ Significant' if p_full_vs_none < 0.05 else '✗ Not significant'}")
        logger.info(f"  Summarized vs None: p={p_summ_vs_none:.4f} {'✓ Significant' if p_summ_vs_none < 0.05 else '✗ Not significant'}")
        logger.info(f"  Full vs Summarized: p={p_full_vs_summ:.4f} {'✓ Significant' if p_full_vs_summ < 0.05 else '✗ Not significant'}")
        logger.info(f"\nResults saved to: {output_path}")
        logger.info(f"{'='*60}\n")

        return summary


def main():
    """Run Experiment 4B."""
    # Configuration
    SCENARIOS_FILE = "data/synthetic/multi_session_scenarios.json"
    OUTPUT_DIR = "results/tables"
    MODEL_NAME = "llama3.2"

    # Check if data file exists
    if not Path(SCENARIOS_FILE).exists():
        logger.error(f"❌ Data file not found: {SCENARIOS_FILE}")
        logger.error("Please generate multi-session scenarios first using the data generation script.")
        return

    # Run experiment
    experiment = MultiSessionMemoryTest(model_name=MODEL_NAME)
    results = experiment.run_experiment(SCENARIOS_FILE, OUTPUT_DIR)

    print("\n[SUCCESS] Experiment 4B complete!")
    print(f"[METRIC] No Memory: {results['no_memory']['recall_accuracy']:.2%}")
    print(f"[METRIC] Full Memory: {results['full_memory']['recall_accuracy']:.2%}")
    print(f"[METRIC] Summarized: {results['summarized_memory']['recall_accuracy']:.2%}")


if __name__ == "__main__":
    main()
