"""
Command-line script to run RAG system evaluations.

Usage:
    python run_evaluation.py --quick              # Quick test (10 questions)
    python run_evaluation.py --category effectiveness  # Test specific category
    python run_evaluation.py --difficulty easy     # Test specific difficulty
    python run_evaluation.py --full               # Full evaluation (all questions)
    python run_evaluation.py --num 50             # Custom number of questions
"""

import argparse
from loguru import logger

from src.rag.rag_pipeline import RAGPipelineWithMemory
from src.evaluation.evaluator import SystemEvaluator
from src.utils.logger import setup_logger


def main():
    """Run evaluation based on command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate RAG system performance')

    # Evaluation mode
    parser.add_argument('--quick', action='store_true', help='Quick test (10 questions)')
    parser.add_argument('--full', action='store_true', help='Full evaluation (all questions)')
    parser.add_argument('--num', type=int, help='Number of questions to evaluate')
    parser.add_argument('--category', type=str, help='Test specific category')
    parser.add_argument('--difficulty', type=str, choices=['easy', 'medium', 'hard'],
                        help='Test specific difficulty level')

    # Configuration
    parser.add_argument('--language', type=str, default='english',
                        choices=['english', 'french', 'kinyarwanda'],
                        help='Response language')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--eval-data', type=str, default='data/synthetic/eval_questions.json',
                        help='Path to evaluation questions')

    # Graduate-level metrics
    parser.add_argument('--no-bertscore', action='store_true',
                        help='Disable BERTScore (use Jaccard similarity)')
    parser.add_argument('--llm-judge', action='store_true',
                        help='Enable LLM-as-judge evaluation (slower but more accurate)')

    args = parser.parse_args()

    # Setup logger
    setup_logger()

    logger.info("="*80)
    logger.info("RAG SYSTEM EVALUATION")
    logger.info("="*80)

    # Initialize pipeline
    logger.info(f"Initializing RAG pipeline from {args.config}")
    pipeline = RAGPipelineWithMemory(
        config_path=args.config,
        use_hybrid_retrieval=False,
        use_multilingual=True
    )

    # Initialize evaluator with graduate-level metrics
    logger.info(f"Loading evaluation data from {args.eval_data}")
    use_bertscore = not args.no_bertscore
    use_llm_judge = args.llm_judge

    logger.info(f"Evaluation metrics: BERTScore={'ON' if use_bertscore else 'OFF'}, LLM-Judge={'ON' if use_llm_judge else 'OFF'}")

    evaluator = SystemEvaluator(
        pipeline=pipeline,
        eval_data_path=args.eval_data,
        use_bertscore=use_bertscore,
        use_llm_judge=use_llm_judge
    )

    # Run evaluation
    if args.quick:
        logger.info("Running quick test (10 questions)")
        results = evaluator.quick_test(language=args.language)

    elif args.full:
        logger.info("Running full evaluation (all questions)")
        results = evaluator.full_evaluation(language=args.language)

    elif args.category:
        logger.info(f"Testing category: {args.category}")
        results = evaluator.category_test(args.category, language=args.language)

    elif args.difficulty:
        logger.info(f"Testing difficulty: {args.difficulty}")
        results = evaluator.difficulty_test(args.difficulty, language=args.language)

    elif args.num:
        logger.info(f"Testing {args.num} questions")
        results = evaluator.evaluate_batch(max_questions=args.num, language=args.language)

    else:
        # Default to quick test
        logger.info("No mode specified, running quick test (10 questions)")
        results = evaluator.quick_test(language=args.language)

    # Print summary
    summary = results['summary']
    logger.info("")
    logger.info("="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total Questions:   {summary['overall']['total_questions']}")
    logger.info(f"Passed:            {summary['overall']['passed_count']} ({summary['overall']['pass_rate']}%)")
    logger.info(f"Average Accuracy:  {summary['overall']['avg_accuracy_score']}")
    logger.info(f"Average Quality:   {summary['overall']['avg_quality_score']}")
    logger.info(f"Avg Query Time:    {summary['performance']['avg_query_time_seconds']}s")
    logger.info("="*80)
    logger.info(f"Results saved to: results/evaluation/")
    logger.info("="*80)


if __name__ == "__main__":
    main()
