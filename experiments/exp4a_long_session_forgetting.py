"""
Experiment 4A: Long-Session Forgetting Test

This experiment tests whether the LLM forgets or contradicts itself during long
conversations (20-40 turns), measuring consistency and recall accuracy.

Metrics:
- Contradiction rate
- Recall accuracy (remembering earlier facts)
- Consistency score across turns
- Turn-by-turn performance degradation

Reference: outline.md Section 4, Experiment 4A
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.generator import RAGGenerator
from src.memory.conversation_memory import ConversationMemory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LongSessionForgettingTest:
    """Test memory consistency in long conversations."""

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.generator = RAGGenerator(model_name=model_name)
        self.results = []

    def detect_contradiction(self, statement1: str, statement2: str) -> bool:
        """
        Simple contradiction detection using keyword analysis.
        In production, could use NLI models or LLM-as-judge.
        """
        # Check for explicit negations
        negations = ['not', 'never', 'no longer', 'cannot', 'shouldn\'t', 'avoid']

        # Extract key facts
        s1_lower = statement1.lower()
        s2_lower = statement2.lower()

        # Simple heuristic: if one contains negation and other doesn't for same topic
        # This is a simplified version - real implementation would be more sophisticated
        has_negation_1 = any(neg in s1_lower for neg in negations)
        has_negation_2 = any(neg in s2_lower for neg in negations)

        # If one negates what the other affirms, it's a contradiction
        if has_negation_1 != has_negation_2:
            # Check if they're about the same topic (simple word overlap)
            words1 = set(s1_lower.split())
            words2 = set(s2_lower.split())
            overlap = len(words1 & words2)
            if overlap > 3:  # Arbitrary threshold
                return True

        return False

    def extract_facts(self, response: str) -> List[str]:
        """Extract factual statements from response."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', response)
        facts = [s.strip() for s in sentences if len(s.strip()) > 20]
        return facts

    def run_long_conversation(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single long conversation and measure forgetting."""
        conv_id = conversation_data['conversation_id']
        turns = conversation_data['turns']
        user_facts = conversation_data.get('user_facts', {})

        logger.info(f"Running conversation {conv_id} with {len(turns)} turns")

        # Track conversation state
        memory = ConversationMemory()
        session_id = f"exp4a_{conv_id}"

        # Track facts mentioned and contradictions
        mentioned_facts = []
        contradictions = []
        recall_tests = []

        responses = []

        for i, turn in enumerate(turns, 1):
            question = turn['user_message']
            is_recall_test = turn.get('is_recall_test', False)
            expected_fact = turn.get('expected_fact', None)

            # Generate response
            response = self.generator.generate(
                question=question,
                context_docs=[],  # Using without retrieval to test pure memory
                language="english",
                session_id=session_id
            )

            answer = response['response']
            responses.append(answer)

            # Update memory
            memory.add_turn(session_id, question, answer)

            # Extract facts from response
            current_facts = self.extract_facts(answer)
            mentioned_facts.extend(current_facts)

            # Check for contradictions with previous facts
            for prev_fact in mentioned_facts[:-len(current_facts)]:
                for curr_fact in current_facts:
                    if self.detect_contradiction(prev_fact, curr_fact):
                        contradictions.append({
                            "turn": i,
                            "previous_fact": prev_fact,
                            "current_fact": curr_fact
                        })

            # If this is a recall test, check if expected fact is mentioned
            if is_recall_test and expected_fact:
                # Simple keyword matching for recall
                fact_recalled = expected_fact.lower() in answer.lower()
                recall_tests.append({
                    "turn": i,
                    "expected_fact": expected_fact,
                    "recalled": fact_recalled,
                    "response": answer
                })

        # Calculate metrics
        num_turns = len(turns)
        num_contradictions = len(contradictions)
        contradiction_rate = num_contradictions / num_turns if num_turns > 0 else 0

        num_recall_tests = len(recall_tests)
        num_successful_recalls = sum(1 for r in recall_tests if r['recalled'])
        recall_accuracy = num_successful_recalls / num_recall_tests if num_recall_tests > 0 else 0

        result = {
            "conversation_id": conv_id,
            "num_turns": num_turns,
            "num_contradictions": num_contradictions,
            "contradiction_rate": contradiction_rate,
            "num_recall_tests": num_recall_tests,
            "num_successful_recalls": num_successful_recalls,
            "recall_accuracy": recall_accuracy,
            "contradictions": contradictions,
            "recall_tests": recall_tests,
            "responses": responses
        }

        return result

    def run_experiment(self, conversations_file: str, output_dir: str = "results/tables"):
        """Run long-session forgetting experiment."""
        logger.info(f"Starting Experiment 4A: Long-Session Forgetting Test")

        # Load conversations
        conv_path = Path(conversations_file)
        if not conv_path.exists():
            raise FileNotFoundError(f"Conversations file not found: {conversations_file}")

        with open(conv_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        if isinstance(conversations, dict):
            conversations = conversations.get('conversations', [])

        logger.info(f"Loaded {len(conversations)} long conversations")

        # Run each conversation
        all_results = []
        for conv_data in conversations:
            result = self.run_long_conversation(conv_data)
            all_results.append(result)
            self.results.append(result)

        # Aggregate metrics
        avg_contradiction_rate = sum(r['contradiction_rate'] for r in all_results) / len(all_results)
        avg_recall_accuracy = sum(r['recall_accuracy'] for r in all_results) / len(all_results)
        total_contradictions = sum(r['num_contradictions'] for r in all_results)
        total_turns = sum(r['num_turns'] for r in all_results)

        summary = {
            "experiment": "Experiment 4A: Long-Session Forgetting Test",
            "model": self.model_name,
            "num_conversations": len(conversations),
            "total_turns": total_turns,
            "avg_turns_per_conversation": total_turns / len(conversations),
            "total_contradictions": total_contradictions,
            "avg_contradiction_rate": avg_contradiction_rate,
            "avg_recall_accuracy": avg_recall_accuracy,
            "timestamp": datetime.now().isoformat()
        }

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        detailed_file = output_path / "exp4a_long_session_detailed.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary,
                "results": self.results
            }, f, indent=2, ensure_ascii=False)

        # Save summary
        summary_file = output_path / "exp4a_long_session_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT 4A RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Conversations tested: {len(conversations)}")
        logger.info(f"Total turns: {total_turns}")
        logger.info(f"Avg turns/conversation: {total_turns / len(conversations):.1f}")
        logger.info(f"Total contradictions: {total_contradictions}")
        logger.info(f"Avg contradiction rate: {avg_contradiction_rate:.2%}")
        logger.info(f"Avg recall accuracy: {avg_recall_accuracy:.2%}")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"{'='*60}\n")

        return summary


def main():
    """Run Experiment 4A."""
    # Configuration
    CONVERSATIONS_FILE = "data/synthetic/long_session_conversations.json"
    OUTPUT_DIR = "results/tables"
    MODEL_NAME = "llama3.2"

    # Check if data file exists
    if not Path(CONVERSATIONS_FILE).exists():
        logger.error(f"❌ Data file not found: {CONVERSATIONS_FILE}")
        logger.error("Please generate long-session conversations first using the data generation script.")
        return

    # Run experiment
    experiment = LongSessionForgettingTest(model_name=MODEL_NAME)
    results = experiment.run_experiment(CONVERSATIONS_FILE, OUTPUT_DIR)

    print("\n[SUCCESS] Experiment 4A complete!")
    print(f"[METRIC] Conversations tested: {results['num_conversations']}")
    print(f"[METRIC] Total turns: {results['total_turns']}")
    print(f"[METRIC] Contradiction rate: {results['avg_contradiction_rate']:.2%}")
    print(f"[METRIC] Recall accuracy: {results['avg_recall_accuracy']:.2%}")


if __name__ == "__main__":
    main()
