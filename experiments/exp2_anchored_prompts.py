"""
Experiment 2: Anchored Prompt Evaluation

This experiment evaluates the effectiveness of strict guideline-following prompts
in improving accuracy and reducing hallucinations.

Metrics:
- BERTScore F1 (with anchored prompts)
- Safety fallback rate
- Hallucination reduction vs baseline
- Accuracy improvement

Reference: outline.md Section 4, Experiment 2
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import EvaluationMetrics
import ollama

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnchoredPromptTest:
    """Test LLM with strict guideline-following prompts."""

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.evaluator = EvaluationMetrics()
        self.results = []
        self.safety_fallback_count = 0

    def get_anchored_prompt(self, question: str) -> str:
        """Generate strict guideline-following prompt."""
        return f"""You are a WHO-certified contraception counseling assistant. Follow these strict guidelines:

1. ONLY provide information based on WHO Family Planning Handbook 2022 and BCS+ Toolkit
2. If you don't know the answer with certainty, say "I should consult WHO guidelines for this specific information"
3. Always prioritize user safety and informed consent
4. Never make medical recommendations outside evidence-based guidelines
5. Always mention the importance of consulting healthcare providers

Question: {question}

Provide an accurate, guideline-based answer:"""

    def run_anchored_query(self, question: str) -> Dict[str, Any]:
        """Query LLM with anchored prompt."""
        start_time = time.time()

        prompt = self.get_anchored_prompt(question)

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response['message']['content']
            latency = time.time() - start_time

            # Check for safety fallback
            safety_phrases = [
                "should consult",
                "consult WHO guidelines",
                "consult healthcare provider",
                "don't know",
                "uncertain",
                "please consult"
            ]
            has_safety_fallback = any(phrase.lower() in answer.lower() for phrase in safety_phrases)

            if has_safety_fallback:
                self.safety_fallback_count += 1

            return {
                "question": question,
                "answer": answer,
                "latency": latency,
                "has_safety_fallback": has_safety_fallback,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return {
                "question": question,
                "answer": "",
                "latency": time.time() - start_time,
                "has_safety_fallback": False,
                "success": False,
                "error": str(e)
            }

    def run_experiment(self, questions_file: str, output_dir: str = "results/tables"):
        """Run anchored prompt test on question set."""
        logger.info(f"Starting Experiment 2: Anchored Prompt Evaluation")
        logger.info(f"Model: {self.model_name}")

        # Load questions
        questions_path = Path(questions_file)
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")

        with open(questions_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        questions = qa_data if isinstance(qa_data, list) else qa_data.get('questions', [])
        logger.info(f"Loaded {len(questions)} questions")

        # Run queries
        all_predictions = []
        all_references = []
        latencies = []

        for i, item in enumerate(questions, 1):
            question = item.get('question', '')
            reference = item.get('ground_truth', '')

            logger.info(f"Processing question {i}/{len(questions)}")

            result = self.run_anchored_query(question)

            if result['success']:
                all_predictions.append(result['answer'])
                all_references.append(reference)
                latencies.append(result['latency'])

                self.results.append({
                    "question_id": i,
                    "question": question,
                    "reference": reference,
                    "prediction": result['answer'],
                    "latency": result['latency'],
                    "has_safety_fallback": result['has_safety_fallback']
                })

        # Calculate metrics
        logger.info("Calculating BERTScore metrics...")
        # Evaluate each prediction-reference pair
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for pred, ref in zip(all_predictions, all_references):
            scores = self.evaluator.bertscore_similarity(pred, ref, lang='en')
            precision_scores.append(scores['precision'])
            recall_scores.append(scores['recall'])
            f1_scores.append(scores['f1'])

        bert_scores = {
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores
        }

        # Aggregate results
        avg_precision = sum(bert_scores['precision']) / len(bert_scores['precision'])
        avg_recall = sum(bert_scores['recall']) / len(bert_scores['recall'])
        avg_f1 = sum(bert_scores['f1']) / len(bert_scores['f1'])
        avg_latency = sum(latencies) / len(latencies)
        safety_fallback_rate = self.safety_fallback_count / len(all_predictions)

        summary = {
            "experiment": "Experiment 2: Anchored Prompt Evaluation",
            "model": self.model_name,
            "num_questions": len(questions),
            "num_successful": len(all_predictions),
            "bertscore_precision": avg_precision,
            "bertscore_recall": avg_recall,
            "bertscore_f1": avg_f1,
            "avg_latency_seconds": avg_latency,
            "safety_fallback_rate": safety_fallback_rate,
            "safety_fallback_count": self.safety_fallback_count,
            "timestamp": datetime.now().isoformat()
        }

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        detailed_file = output_path / "exp2_anchored_detailed.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary,
                "results": self.results
            }, f, indent=2, ensure_ascii=False)

        # Save summary table
        summary_file = output_path / "exp2_anchored_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT 2 RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Questions processed: {summary['num_successful']}/{summary['num_questions']}")
        logger.info(f"BERTScore Precision: {avg_precision:.4f}")
        logger.info(f"BERTScore Recall: {avg_recall:.4f}")
        logger.info(f"BERTScore F1: {avg_f1:.4f}")
        logger.info(f"Average Latency: {avg_latency:.2f}s")
        logger.info(f"Safety Fallback Rate: {safety_fallback_rate:.2%}")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"{'='*60}\n")

        return summary


def main():
    """Run Experiment 2."""
    # Configuration
    QUESTIONS_FILE = "data/synthetic/qa_pairs.json"
    OUTPUT_DIR = "results/tables"
    MODEL_NAME = "llama3.2"

    # Run experiment
    experiment = AnchoredPromptTest(model_name=MODEL_NAME)
    results = experiment.run_experiment(QUESTIONS_FILE, OUTPUT_DIR)

    print("\n[SUCCESS] Experiment 2 complete!")
    print(f"[METRIC] BERTScore F1: {results['bertscore_f1']:.4f}")
    print(f"[METRIC] Safety Fallback Rate: {results['safety_fallback_rate']:.2%}")
    print(f"[TIME] Avg Latency: {results['avg_latency_seconds']:.2f}s")


if __name__ == "__main__":
    main()
