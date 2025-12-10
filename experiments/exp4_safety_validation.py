"""
Experiment 4: Safety Validation Layer Evaluation

Purpose:
    Evaluate the impact of adding SafetyValidator to compliance-aware prompting.
    Compare with Experiment 1 (baseline) and Experiment 2 (compliance prompts only).

Research Question:
    Does adding lightweight post-generation safety validation improve compliance
    without degrading performance?

Expected Results:
    - Similar compliance to Exp2 (76.25%)
    - Reduced critical issues (auto-disclaimers on medical questions)
    - Logged unsafe patterns for human review
    - Minimal latency overhead (<100ms)

Approach:
    - Same models as Exp1/Exp2: Claude Opus 4.5, o3, Grok
    - Same compliance test set (80 questions)
    - NEW: Safety validation enabled
    - Compare: compliance scores, safety issues, latency
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.generator import ComplianceGenerator, SafetyValidator
from src.utils.multilang_llm_client import MultiLanguageLLMClient
from loguru import logger
import yaml

# Setup logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    "results/logs/exp4_safety_validation_{time}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG"
)


def load_compliance_test_set() -> List[Dict]:
    """Load compliance test dataset."""
    test_set_path = Path("data/compliance_test_set.json")

    if not test_set_path.exists():
        logger.error(f"Test set not found: {test_set_path}")
        raise FileNotFoundError(f"Compliance test set not found at {test_set_path}")

    with open(test_set_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_cases = data.get('test_cases', [])
    logger.info(f"Loaded {len(test_cases)} test cases")
    return test_cases


def run_experiment(
    model_name: str,
    provider: str,
    test_questions: List[Dict],
    enable_safety_validation: bool = True
) -> Dict:
    """
    Run experiment with specified model and safety validation setting.

    Args:
        model_name: Model to test (e.g., 'claude-opus-4-5-20251101')
        provider: Provider ('anthropic', 'openai', 'xai')
        test_questions: List of test questions
        enable_safety_validation: Enable SafetyValidator

    Returns:
        Results dictionary
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running Experiment 4: {model_name}")
    logger.info(f"Safety Validation: {'ENABLED' if enable_safety_validation else 'DISABLED'}")
    logger.info(f"{'='*80}\n")

    # Configure model
    llm_config = {
        'provider': provider,
        'model_name': model_name,
        'temperature': 0.7,
        'max_tokens': 1024
    }

    # Initialize generator with safety validation
    generator = ComplianceGenerator(
        llm_config=llm_config,
        enable_safety_validation=enable_safety_validation
    )

    results = []
    safety_stats = {
        'total_validations': 0,
        'safe_responses': 0,
        'unsafe_responses': 0,
        'disclaimers_added': 0,
        'high_severity_issues': 0,
        'medium_severity_issues': 0,
        'low_severity_issues': 0,
        'validation_times_ms': []
    }

    start_time = time.time()

    for i, question_data in enumerate(test_questions, 1):
        # Extract question from 'scenario' field (compliance test set format)
        question = question_data.get('scenario', question_data.get('user_query', question_data.get('question', '')))
        category = question_data.get('category', 'general')
        test_case_id = question_data.get('id', f'test_{i}')

        logger.info(f"\n[{i}/{len(test_questions)}] Question: {question[:100]}...")

        try:
            # Generate response
            gen_start = time.time()
            result = generator.generate(
                query=question,
                context="",  # No RAG context
                language='english'
            )
            gen_time = (time.time() - gen_start) * 1000  # Convert to ms

            # Extract safety validation metadata
            safety_val = result['metadata'].get('safety_validation')

            if safety_val:
                safety_stats['total_validations'] += 1

                if safety_val['is_safe']:
                    safety_stats['safe_responses'] += 1
                else:
                    safety_stats['unsafe_responses'] += 1

                    # Count by severity
                    severity = safety_val['severity']
                    if severity == 'high':
                        safety_stats['high_severity_issues'] += 1
                    elif severity == 'medium':
                        safety_stats['medium_severity_issues'] += 1
                    else:
                        safety_stats['low_severity_issues'] += 1

                if safety_val.get('requires_disclaimer'):
                    safety_stats['disclaimers_added'] += 1

            # Store result in format compatible with compliance scoring
            results.append({
                'test_case_id': test_case_id,
                'category': category,
                'severity': question_data.get('severity', 'unknown'),
                'scenario': question,
                'ground_truth': question_data.get('ground_truth', ''),
                'model_response': result['response'],
                'latency_seconds': gen_time / 1000,
                'timestamp': datetime.now().isoformat(),
                'safety_validation': safety_val,
                'compliant_criteria': question_data.get('compliant_response_criteria', {}),
                'non_compliant_indicators': question_data.get('non_compliant_indicators', []),
                'who_guideline': question_data.get('who_guideline', {}),
                'rwanda_context': question_data.get('rwanda_context'),
                'rag_context': None
            })

            logger.info(f"✓ Generated ({gen_time:.0f}ms)")
            if safety_val and not safety_val['is_safe']:
                logger.warning(f"  Safety issues: {safety_val['issues']}")

        except Exception as e:
            logger.error(f"✗ Error: {e}")
            results.append({
                'test_case_id': test_case_id,
                'category': category,
                'scenario': question,
                'model_response': f'[ERROR] {str(e)}',
                'error': str(e),
                'safety_validation': None
            })

    total_time = time.time() - start_time

    # Calculate statistics
    avg_gen_time = sum(r['generation_time_ms'] for r in results if 'generation_time_ms' in r) / len(results)

    logger.info(f"\n{'='*80}")
    logger.info(f"EXPERIMENT COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Avg generation time: {avg_gen_time:.0f}ms")
    logger.info(f"\nSafety Statistics:")
    logger.info(f"  Total validations: {safety_stats['total_validations']}")
    logger.info(f"  Safe responses: {safety_stats['safe_responses']} ({safety_stats['safe_responses']/safety_stats['total_validations']*100:.1f}%)")
    logger.info(f"  Unsafe responses: {safety_stats['unsafe_responses']} ({safety_stats['unsafe_responses']/safety_stats['total_validations']*100:.1f}%)")
    logger.info(f"  High severity issues: {safety_stats['high_severity_issues']}")
    logger.info(f"  Medium severity issues: {safety_stats['medium_severity_issues']}")
    logger.info(f"  Disclaimers added: {safety_stats['disclaimers_added']}")

    return {
        'experiment_metadata': {
            'experiment_number': 4,
            'model': model_name,
            'provider': provider,
            'rag_used': False,
            'safety_validation_enabled': enable_safety_validation,
            'system_prompt': 'Compliance-aware prompting with SafetyValidator (Trust but Verify approach)',
            'num_test_cases': len(test_questions),
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat()
        },
        'responses': results,
        'safety_statistics': safety_stats,
        'total_time_seconds': total_time,
        'avg_generation_time_ms': avg_gen_time
    }


def save_results(results: Dict, output_dir: Path):
    """Save experiment results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with model name and timestamp
    model_slug = results['experiment_metadata']['model'].replace(':', '-').replace('/', '-')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"exp4_{model_slug}_safety_validation_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Results saved to: {output_path}")
    return output_path


def compare_with_previous_experiments(exp4_results: Dict):
    """
    Compare Experiment 4 results with Experiments 1 and 2.

    Loads previous experiment results and compares:
    - Compliance scores
    - Critical issues
    - Generation latency
    - Safety validation impact
    """
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON WITH PREVIOUS EXPERIMENTS")
    logger.info(f"{'='*80}\n")

    # Look for Exp1 and Exp2 results
    results_dir = Path("results/compliance_experiments")

    model_name = exp4_results['model_name']

    # Try to find matching results from Exp1 and Exp2
    exp1_files = list(results_dir.glob(f"exp1_*{model_name.split('-')[0]}*.json"))
    exp2_files = list(results_dir.glob(f"exp2_*{model_name.split('-')[0]}*.json"))

    logger.info("Comparison Summary:")
    logger.info(f"  Experiment 4 (Safety Validation):")
    logger.info(f"    - Safe responses: {exp4_results['safety_statistics']['safe_responses']}/{exp4_results['total_questions']}")
    logger.info(f"    - Avg latency: {exp4_results['avg_generation_time_ms']:.0f}ms")
    logger.info(f"    - Disclaimers added: {exp4_results['safety_statistics']['disclaimers_added']}")

    if exp1_files:
        logger.info(f"\n  Experiment 1 results found: {exp1_files[0].name}")
    else:
        logger.warning("  No Experiment 1 results found for comparison")

    if exp2_files:
        logger.info(f"  Experiment 2 results found: {exp2_files[0].name}")
    else:
        logger.warning("  No Experiment 2 results found for comparison")

    logger.info(f"\n  Note: Run compliance scoring on Exp4 results to compare compliance metrics")


def main():
    """Main experiment runner."""
    logger.info("="*80)
    logger.info("EXPERIMENT 4: SAFETY VALIDATION LAYER EVALUATION")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load test questions
    test_questions = load_compliance_test_set()

    # Define models to test (same as Exp1/Exp2)
    models = [
        {
            'name': 'claude-opus-4-5-20251101',
            'provider': 'anthropic',
            'description': 'Claude Opus 4.5 (Exp2 winner: 76.25% compliant)'
        },
        # Uncomment to test other models:
        # {
        #     'name': 'o3-2025-04-16',
        #     'provider': 'openai',
        #     'description': 'OpenAI o3 (Exp2: 85% compliant)'
        # },
        # {
        #     'name': 'grok-4-1-fast-reasoning',
        #     'provider': 'xai',
        #     'description': 'Grok 4.1 Fast Reasoning (Exp2: 73.75% compliant)'
        # }
    ]

    output_dir = Path("results/compliance_experiments")

    # Run experiment for each model
    all_results = []

    for model_config in models:
        logger.info(f"\nTesting: {model_config['description']}")

        try:
            results = run_experiment(
                model_name=model_config['name'],
                provider=model_config['provider'],
                test_questions=test_questions,
                enable_safety_validation=True  # NEW: Safety validation enabled
            )

            # Save results
            output_path = save_results(results, output_dir)
            all_results.append(output_path)

            # Compare with previous experiments
            compare_with_previous_experiments(results)

        except Exception as e:
            logger.error(f"Error running experiment for {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Files generated:")
    for path in all_results:
        logger.info(f"  - {path.name}")

    logger.info(f"\nNext steps:")
    logger.info(f"  1. Run compliance scoring: python scripts/auto_annotate_with_llm.py <result_file> --judge-model gpt-4o")
    logger.info(f"  2. Compare with Exp1/Exp2 compliance scores")
    logger.info(f"  3. Analyze safety validation impact")


if __name__ == "__main__":
    main()
