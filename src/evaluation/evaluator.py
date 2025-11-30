"""
System evaluator for running comprehensive RAG evaluations.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm
from loguru import logger

from src.evaluation.metrics import EvaluationMetrics
from src.rag.rag_pipeline import RAGPipeline, RAGPipelineWithMemory


class SystemEvaluator:
    """Run systematic evaluations of the RAG system."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        eval_data_path: str = "data/synthetic/eval_questions.json",
        output_dir: str = "results/evaluation",
        use_bertscore: bool = True,
        use_llm_judge: bool = False
    ):
        """
        Initialize system evaluator with graduate-level metrics.

        Args:
            pipeline: RAG pipeline to evaluate
            eval_data_path: Path to evaluation questions JSON
            output_dir: Directory to save results
            use_bertscore: Use BERTScore for semantic similarity (recommended)
            use_llm_judge: Use LLM-as-judge evaluation (slower but more accurate)
        """
        self.pipeline = pipeline
        self.eval_data_path = Path(eval_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = EvaluationMetrics()
        self.use_bertscore = use_bertscore
        self.use_llm_judge = use_llm_judge

        # Get LLM client for judge evaluation
        self.llm_client = None
        if use_llm_judge:
            try:
                self.llm_client = self.pipeline.generator.llm_client
                logger.info("LLM-as-judge evaluation enabled")
            except Exception as e:
                logger.warning(f"Could not access LLM client for judge evaluation: {e}")
                self.use_llm_judge = False

        # Load evaluation data
        self.eval_questions = self._load_eval_data()

        logger.info(f"SystemEvaluator initialized with {len(self.eval_questions)} questions")
        logger.info(f"BERTScore: {'enabled' if use_bertscore else 'disabled'}")
        logger.info(f"LLM Judge: {'enabled' if self.use_llm_judge else 'disabled'}")

    def _load_eval_data(self) -> List[Dict]:
        """Load evaluation questions from JSON file."""
        if not self.eval_data_path.exists():
            raise FileNotFoundError(f"Evaluation data not found: {self.eval_data_path}")

        with open(self.eval_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} evaluation questions")
        return data

    def evaluate_single_question(
        self,
        question_data: Dict,
        language: str = 'english',
        include_sources: bool = True
    ) -> Dict:
        """
        Evaluate system on a single question.

        Args:
            question_data: Question dictionary with keys: question, ground_truth, category, etc.
            language: Language for response
            include_sources: Whether to retrieve sources

        Returns:
            Evaluation result dictionary
        """
        question_id = question_data.get('question_id', 'unknown')
        question = question_data['question']
        ground_truth = question_data['ground_truth']
        category = question_data.get('category', 'general')
        difficulty = question_data.get('difficulty', 'medium')
        expects_safety_fallback = question_data.get('expects_safety_fallback', False)

        try:
            # Query the system
            start_time = time.time()
            result = self.pipeline.query(
                question=question,
                language=language,
                include_sources=include_sources
            )
            query_time = time.time() - start_time

            response = result['response']
            sources = result.get('sources', [])

            # Map language to BERTScore language code
            lang_map = {'english': 'en', 'french': 'fr', 'kinyarwanda': 'en'}  # rw not supported
            lang_code = lang_map.get(language, 'en')

            # Evaluate response with graduate-level metrics
            evaluation = self.metrics.evaluate_response(
                response=response,
                ground_truth=ground_truth,
                sources=sources,
                expects_safety_fallback=expects_safety_fallback,
                question=question,
                llm_client=self.llm_client if self.use_llm_judge else None,
                use_bertscore=self.use_bertscore,
                use_llm_judge=self.use_llm_judge,
                lang=lang_code
            )

            # Add question metadata
            evaluation.update({
                'question_id': question_id,
                'question': question,
                'ground_truth': ground_truth,
                'response': response,
                'category': category,
                'difficulty': difficulty,
                'query_time_seconds': round(query_time, 3),
                'language': language,
                'timestamp': datetime.now().isoformat()
            })

            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating question {question_id}: {e}")
            return {
                'question_id': question_id,
                'question': question,
                'error': str(e),
                'passed': False,
                'category': category,
                'difficulty': difficulty
            }

    def evaluate_batch(
        self,
        max_questions: Optional[int] = None,
        language: str = 'english',
        categories: Optional[List[str]] = None,
        difficulty_levels: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate system on a batch of questions.

        Args:
            max_questions: Maximum number of questions to evaluate (None for all)
            language: Language for responses
            categories: Filter by specific categories (None for all)
            difficulty_levels: Filter by difficulty levels (None for all)

        Returns:
            Complete evaluation results
        """
        # Filter questions
        questions = self.eval_questions

        if categories:
            questions = [q for q in questions if q.get('category') in categories]

        if difficulty_levels:
            questions = [q for q in questions if q.get('difficulty') in difficulty_levels]

        if max_questions:
            questions = questions[:max_questions]

        logger.info(f"Evaluating {len(questions)} questions")

        # Run evaluations
        results = []
        for question_data in tqdm(questions, desc="Evaluating"):
            result = self.evaluate_single_question(
                question_data=question_data,
                language=language
            )
            results.append(result)

            # Small delay to avoid overwhelming the system
            time.sleep(0.1)

        # Aggregate results
        summary = self._generate_summary(results)

        # Save results
        self._save_results(results, summary, language)

        return {
            'individual_results': results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from individual results."""
        # Overall statistics
        overall_stats = self.metrics.aggregate_results(results)

        # Category breakdown
        categories = list(set(r['category'] for r in results if 'category' in r))
        category_stats = self.metrics.category_breakdown(results, categories)

        # Difficulty breakdown
        difficulty_stats = self.metrics.difficulty_breakdown(results)

        # Performance statistics
        query_times = [r['query_time_seconds'] for r in results if 'query_time_seconds' in r]
        avg_query_time = sum(query_times) / len(query_times) if query_times else 0

        return {
            'overall': overall_stats,
            'by_category': category_stats,
            'by_difficulty': difficulty_stats,
            'performance': {
                'avg_query_time_seconds': round(avg_query_time, 3),
                'min_query_time': round(min(query_times), 3) if query_times else 0,
                'max_query_time': round(max(query_times), 3) if query_times else 0
            }
        }

    def _save_results(self, results: List[Dict], summary: Dict, language: str):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save individual results
        results_file = self.output_dir / f"eval_results_{language}_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved individual results to {results_file}")

        # Save summary
        summary_file = self.output_dir / f"eval_summary_{language}_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved summary to {summary_file}")

        # Save human-readable report
        report_file = self.output_dir / f"eval_report_{language}_{timestamp}.txt"
        self._save_text_report(summary, report_file)

        logger.info(f"Saved text report to {report_file}")

    def _save_text_report(self, summary: Dict, output_path: Path):
        """Generate and save human-readable text report."""
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("CONTRACEPTION COUNSELING SYSTEM - EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Overall statistics
        overall = summary['overall']
        report_lines.append("OVERALL PERFORMANCE")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Questions:        {overall['total_questions']}")
        report_lines.append(f"Passed:                 {overall['passed_count']} ({overall['pass_rate']}%)")
        report_lines.append(f"Failed:                 {overall['failed_count']}")
        report_lines.append(f"Average Accuracy:       {overall['avg_accuracy_score']}")
        report_lines.append(f"Average Quality:        {overall['avg_quality_score']}")
        report_lines.append(f"Safety Language Rate:   {overall['safety_language_rate']}%")
        report_lines.append(f"Citation Rate:          {overall['citation_rate']}%")
        report_lines.append("")

        # Performance
        perf = summary['performance']
        report_lines.append("PERFORMANCE METRICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Average Query Time:     {perf['avg_query_time_seconds']}s")
        report_lines.append(f"Min Query Time:         {perf['min_query_time']}s")
        report_lines.append(f"Max Query Time:         {perf['max_query_time']}s")
        report_lines.append("")

        # Category breakdown
        report_lines.append("PERFORMANCE BY CATEGORY")
        report_lines.append("-" * 80)
        for category, stats in summary['by_category'].items():
            if stats['total_questions'] > 0:
                report_lines.append(f"{category.upper()}")
                report_lines.append(f"  Questions: {stats['total_questions']}")
                report_lines.append(f"  Pass Rate: {stats['pass_rate']}%")
                report_lines.append(f"  Accuracy:  {stats['avg_accuracy_score']}")
                report_lines.append("")

        # Difficulty breakdown
        report_lines.append("PERFORMANCE BY DIFFICULTY")
        report_lines.append("-" * 80)
        for difficulty, stats in summary['by_difficulty'].items():
            if stats['total_questions'] > 0:
                report_lines.append(f"{difficulty.upper()}")
                report_lines.append(f"  Questions: {stats['total_questions']}")
                report_lines.append(f"  Pass Rate: {stats['pass_rate']}%")
                report_lines.append(f"  Accuracy:  {stats['avg_accuracy_score']}")
                report_lines.append("")

        report_lines.append("=" * 80)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

    def quick_test(self, num_questions: int = 10, language: str = 'english') -> Dict:
        """
        Run a quick evaluation test.

        Args:
            num_questions: Number of questions to test
            language: Language for responses

        Returns:
            Quick test results
        """
        logger.info(f"Running quick test with {num_questions} questions")

        return self.evaluate_batch(
            max_questions=num_questions,
            language=language
        )

    def category_test(self, category: str, language: str = 'english') -> Dict:
        """
        Test a specific category.

        Args:
            category: Category to test
            language: Language for responses

        Returns:
            Category test results
        """
        logger.info(f"Testing category: {category}")

        return self.evaluate_batch(
            categories=[category],
            language=language
        )

    def difficulty_test(self, difficulty: str, language: str = 'english') -> Dict:
        """
        Test a specific difficulty level.

        Args:
            difficulty: Difficulty level (easy, medium, hard)
            language: Language for responses

        Returns:
            Difficulty test results
        """
        logger.info(f"Testing difficulty: {difficulty}")

        return self.evaluate_batch(
            difficulty_levels=[difficulty],
            language=language
        )

    def full_evaluation(self, language: str = 'english') -> Dict:
        """
        Run full evaluation on all questions.

        Args:
            language: Language for responses

        Returns:
            Complete evaluation results
        """
        logger.info("Running full evaluation")

        return self.evaluate_batch(language=language)
