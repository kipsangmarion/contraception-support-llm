"""
Experiment 3: RAG vs Non-RAG Comparison

This experiment compares RAG-enabled responses against baseline and anchored prompts,
measuring improvements in accuracy, grounding, and safety.

Metrics:
- BERTScore F1 (all three conditions)
- Citation/grounding rate
- Response latency
- Statistical significance (paired t-test)

Reference: outline.md Section 4, Experiment 3
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime
import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import EvaluationMetrics
from src.rag.retriever import RAGRetriever
from src.rag.generator import RAGGenerator
import ollama

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGComparisonTest:
    """Compare RAG vs non-RAG performance."""

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.evaluator = EvaluationMetrics()

        # Initialize RAG components
        self.retriever = RAGRetriever(
            vector_store_path="data/processed/vector_store",
            embeddings_config={'model_name': 'all-MiniLM-L6-v2', 'provider': 'sentence-transformers'}
        )
        self.generator = RAGGenerator(
            llm_config={'provider': 'ollama', 'model_name': model_name}
        )

        self.results = []

    def run_baseline_query(self, question: str) -> Dict[str, Any]:
        """Query LLM without RAG (baseline)."""
        start_time = time.time()

        prompt = f"""You are a contraception counseling assistant. Answer the following question accurately based on your knowledge.

Question: {question}

Answer:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response['message']['content']
            latency = time.time() - start_time

            return {
                "answer": answer,
                "latency": latency,
                "has_citations": False,
                "num_sources": 0,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error in baseline query: {e}")
            return {"answer": "", "latency": 0, "has_citations": False, "num_sources": 0, "success": False}

    def run_rag_query(self, question: str) -> Dict[str, Any]:
        """Query with full RAG pipeline."""
        start_time = time.time()

        try:
            # Retrieve relevant documents
            docs = self.retriever.retrieve(question, top_k=3)

            # Build context from retrieved documents
            context = "\n\n".join([doc.get('text', '') for doc in docs])

            # Generate response
            response = self.generator.generate(
                query=question,
                context=context,
                language="english"
            )

            latency = time.time() - start_time
            has_citations = len(response.get('citations', [])) > 0
            num_sources = len(response.get('citations', []))

            return {
                "answer": response['response'],
                "latency": latency,
                "has_citations": has_citations,
                "num_sources": num_sources,
                "sources": response.get('citations', []),
                "success": True
            }
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {"answer": "", "latency": 0, "has_citations": False, "num_sources": 0, "success": False}

    def run_experiment(self, questions_file: str, output_dir: str = "results/tables"):
        """Run RAG comparison experiment."""
        logger.info(f"Starting Experiment 3: RAG vs Non-RAG Comparison")
        logger.info(f"Model: {self.model_name}")

        # Load questions
        questions_path = Path(questions_file)
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")

        with open(questions_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        questions = qa_data if isinstance(qa_data, list) else qa_data.get('questions', [])
        logger.info(f"Loaded {len(questions)} questions")

        # Storage for results
        baseline_predictions = []
        rag_predictions = []
        references = []

        baseline_latencies = []
        rag_latencies = []
        citation_counts = []

        for i, item in enumerate(questions, 1):
            question = item.get('question', '')
            reference = item.get('ground_truth', '')

            logger.info(f"Processing question {i}/{len(questions)}")

            # Run baseline
            baseline_result = self.run_baseline_query(question)

            # Run RAG
            rag_result = self.run_rag_query(question)

            if baseline_result['success'] and rag_result['success']:
                baseline_predictions.append(baseline_result['answer'])
                rag_predictions.append(rag_result['answer'])
                references.append(reference)

                baseline_latencies.append(baseline_result['latency'])
                rag_latencies.append(rag_result['latency'])
                citation_counts.append(rag_result['num_sources'])

                self.results.append({
                    "question_id": i,
                    "question": question,
                    "reference": reference,
                    "baseline_answer": baseline_result['answer'],
                    "rag_answer": rag_result['answer'],
                    "baseline_latency": baseline_result['latency'],
                    "rag_latency": rag_result['latency'],
                    "has_citations": rag_result['has_citations'],
                    "num_sources": rag_result['num_sources']
                })

        # Calculate BERTScore for both conditions
        logger.info("Calculating BERTScore for baseline...")
        baseline_precision = []
        baseline_recall = []
        baseline_f1 = []
        for pred, ref in zip(baseline_predictions, references):
            scores = self.evaluator.bertscore_similarity(pred, ref, lang='en')
            baseline_precision.append(scores['precision'])
            baseline_recall.append(scores['recall'])
            baseline_f1.append(scores['f1'])

        baseline_scores = {
            'precision': baseline_precision,
            'recall': baseline_recall,
            'f1': baseline_f1
        }

        logger.info("Calculating BERTScore for RAG...")
        rag_precision = []
        rag_recall = []
        rag_f1 = []
        for pred, ref in zip(rag_predictions, references):
            scores = self.evaluator.bertscore_similarity(pred, ref, lang='en')
            rag_precision.append(scores['precision'])
            rag_recall.append(scores['recall'])
            rag_f1.append(scores['f1'])

        rag_scores = {
            'precision': rag_precision,
            'recall': rag_recall,
            'f1': rag_f1
        }

        # Aggregate metrics
        baseline_f1 = np.mean(baseline_scores['f1'])
        rag_f1 = np.mean(rag_scores['f1'])

        baseline_latency = np.mean(baseline_latencies)
        rag_latency = np.mean(rag_latencies)

        citation_rate = sum(1 for c in citation_counts if c > 0) / len(citation_counts) if len(citation_counts) > 0 else 0
        avg_citations = np.mean(citation_counts) if len(citation_counts) > 0 else 0

        # Statistical significance test (paired t-test)
        t_stat, p_value = stats.ttest_rel(rag_scores['f1'], baseline_scores['f1'])

        summary = {
            "experiment": "Experiment 3: RAG vs Non-RAG Comparison",
            "model": self.model_name,
            "num_questions": len(questions),
            "num_successful": len(references),

            "baseline": {
                "bertscore_f1": float(baseline_f1),
                "avg_latency_seconds": float(baseline_latency)
            },

            "rag": {
                "bertscore_f1": float(rag_f1),
                "avg_latency_seconds": float(rag_latency),
                "citation_rate": float(citation_rate),
                "avg_citations_per_response": float(avg_citations)
            },

            "improvement": {
                "f1_improvement": float(rag_f1 - baseline_f1),
                "f1_improvement_percent": float((rag_f1 - baseline_f1) / baseline_f1 * 100),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_at_0.05": bool(p_value < 0.05)
            },

            "timestamp": datetime.now().isoformat()
        }

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        detailed_file = output_path / "exp3_rag_comparison_detailed.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary,
                "results": self.results
            }, f, indent=2, ensure_ascii=False)

        # Save summary
        summary_file = output_path / "exp3_rag_comparison_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT 3 RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Questions processed: {summary['num_successful']}/{summary['num_questions']}")
        logger.info(f"\nBaseline Performance:")
        logger.info(f"  BERTScore F1: {baseline_f1:.4f}")
        logger.info(f"  Avg Latency: {baseline_latency:.2f}s")
        logger.info(f"\nRAG Performance:")
        logger.info(f"  BERTScore F1: {rag_f1:.4f}")
        logger.info(f"  Avg Latency: {rag_latency:.2f}s")
        logger.info(f"  Citation Rate: {citation_rate:.2%}")
        logger.info(f"  Avg Citations: {avg_citations:.2f}")
        logger.info(f"\nImprovement:")
        logger.info(f"  F1 Δ: +{summary['improvement']['f1_improvement']:.4f} ({summary['improvement']['f1_improvement_percent']:.2f}%)")
        logger.info(f"  Statistical Test: t={t_stat:.3f}, p={p_value:.4f}")
        logger.info(f"  Significant: {'✓ Yes' if p_value < 0.05 else '✗ No'}")
        logger.info(f"\nResults saved to: {output_path}")
        logger.info(f"{'='*60}\n")

        return summary


def main():
    """Run Experiment 3."""
    # Configuration
    QUESTIONS_FILE = "data/synthetic/qa_pairs.json"
    OUTPUT_DIR = "results/tables"
    MODEL_NAME = "llama3.2"

    # Run experiment
    experiment = RAGComparisonTest(model_name=MODEL_NAME)
    results = experiment.run_experiment(QUESTIONS_FILE, OUTPUT_DIR)

    print("\n[SUCCESS] Experiment 3 complete!")
    print(f"[METRIC] Baseline F1: {results['baseline']['bertscore_f1']:.4f}")
    print(f"[METRIC] RAG F1: {results['rag']['bertscore_f1']:.4f}")
    print(f"[METRIC] Improvement: +{results['improvement']['f1_improvement_percent']:.2f}%")
    print(f"[STATS] Statistically significant: {'Yes' if results['improvement']['significant_at_0.05'] else 'No'}")


if __name__ == "__main__":
    main()
