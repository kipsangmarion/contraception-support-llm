#!/usr/bin/env python3
"""
Experiment 3: RAG Comparison Workflow

This script automates the complete Experiment 3 workflow:
1. Runs baseline models with RAG enhancement
2. Annotates results with LLM judge
3. Compares RAG vs non-RAG performance
4. Generates visualizations

Usage:
    # Run all models with RAG
    python scripts/run_exp3_rag_workflow.py --all

    # Run specific model
    python scripts/run_exp3_rag_workflow.py --model o3

    # Run without annotation (manual later)
    python scripts/run_exp3_rag_workflow.py --model claude --no-annotate

Author: Research Team
Date: December 8, 2025
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# Model configuration
MODEL_CONFIG = {
    'o3': {
        'full_name': 'o3-2025-04-16',
        'judge': 'gpt-4o',
        'exp1_summary': 'exp1_o3-2025-04-16_20251208_110608_auto_summary.json',
        'exp2_summary': 'exp2_o3-2025-04-16_20251208_155856_auto_summary.json'
    },
    'claude': {
        'full_name': 'claude-opus-4-5-20251101',
        'judge': 'gpt-4o',
        'exp1_summary': 'exp1_claude-opus-4-5-20251101_20251208_050610_auto_summary.json',
        'exp2_summary': 'exp2_claude-opus-4-5-20251101_20251208_131952_auto_summary.json'
    },
    'grok': {
        'full_name': 'grok-4-1-fast-reasoning',
        'judge': 'gpt-4o',
        'exp1_summary': 'exp1_grok-4-1-fast-reasoning_20251208_053400_auto_summary.json',
        'exp2_summary': 'exp2_grok-4-1-fast-reasoning_20251208_152103_auto_summary.json'
    }
}


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def print_step(step_num, total_steps, description):
    """Print formatted step."""
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    print("-" * 80)


def run_command(cmd, description, dry_run=False):
    """
    Run a shell command with error handling.

    Args:
        cmd: Command to run (list of strings)
        description: Description of what the command does
        dry_run: If True, print command without executing

    Returns:
        True if successful, False otherwise
    """
    print(f"\n▶ {description}")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN - Command not executed]")
        return True

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.stdout:
            print(f"\n{result.stdout}")

        print("  ✓ Success")
        return True

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e}")
        if e.stdout:
            print(f"  Output: {e.stdout}")
        if e.stderr:
            print(f"  Error: {e.stderr}")
        return False


def verify_prerequisites():
    """Verify that required files and dependencies exist."""
    print_header("Verifying Prerequisites")

    checks = {
        "Test dataset": Path("data/compliance_test_set.json"),
        "Vector store": Path("data/processed/vector_store"),
        "Experiment runner": Path("scripts/run_compliance_experiments.py"),
        "Annotation script": Path("scripts/auto_annotate_with_llm.py"),
        "Comparison script": Path("scripts/compare_models.py")
    }

    all_ok = True
    for name, path in checks.items():
        if path.exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: NOT FOUND at {path}")
            all_ok = False

    if not all_ok:
        print("\n⚠ Some prerequisites are missing. Please fix before continuing.")
        return False

    print("\n✓ All prerequisites verified")
    return True


def run_experiment_3(model_key, dry_run=False):
    """
    Run Experiment 3 (RAG) for a specific model.

    Args:
        model_key: Model key from MODEL_CONFIG
        dry_run: If True, print commands without executing

    Returns:
        Path to results file if successful, None otherwise
    """
    config = MODEL_CONFIG[model_key]
    model_name = config['full_name']

    print_step(1, 3, f"Running Experiment 3 (RAG) for {model_name}")

    cmd = [
        "python", "scripts/run_compliance_experiments.py",
        "--experiment", "3",
        "--model", model_name,
        "--rag"
    ]

    success = run_command(
        cmd,
        f"Run RAG experiment for {model_name}",
        dry_run=dry_run
    )

    if not success and not dry_run:
        print(f"\n✗ Experiment 3 failed for {model_name}")
        return None

    # Find the generated results file
    results_dir = Path("results/compliance_experiments")

    if dry_run:
        return "exp3_mock_results.json"

    # Look for newest exp3 file for this model
    pattern = f"exp3_{model_name}_*_responses.json"
    files = sorted(results_dir.glob(pattern), reverse=True)

    if files:
        results_file = files[0]
        print(f"\n✓ Results saved to: {results_file}")
        return results_file
    else:
        print(f"\n⚠ Warning: Could not find results file matching {pattern}")
        return None


def annotate_results(results_file, judge_model, dry_run=False):
    """
    Annotate experiment results with LLM judge.

    Args:
        results_file: Path to results JSON file
        judge_model: Judge model name (e.g., 'gpt-4o')
        dry_run: If True, print command without executing

    Returns:
        True if successful, False otherwise
    """
    print_step(2, 3, "Annotating results with LLM judge")

    cmd = [
        "python", "scripts/auto_annotate_with_llm.py",
        "--results-file", str(results_file),
        "--judge-model", judge_model,
        "--output-suffix", "auto"
    ]

    success = run_command(
        cmd,
        f"Annotate with {judge_model}",
        dry_run=dry_run
    )

    if success or dry_run:
        annotated_file = str(results_file).replace('.json', '_auto_annotated.json')
        print(f"\n✓ Annotated results saved to: {annotated_file}")
        return True
    else:
        print(f"\n✗ Annotation failed")
        return False


def compare_experiments(model_key, dry_run=False):
    """
    Compare Experiment 1, 2, and 3 results.

    Args:
        model_key: Model key from MODEL_CONFIG
        dry_run: If True, print command without executing

    Returns:
        True if successful, False otherwise
    """
    print_step(3, 3, "Comparing across experiments")

    config = MODEL_CONFIG[model_key]
    results_dir = Path("results/compliance_experiments")

    # Find exp3 summary file
    model_name = config['full_name']
    pattern = f"exp3_{model_name}_*_auto_summary.json"
    exp3_files = sorted(results_dir.glob(pattern), reverse=True)

    if not exp3_files and not dry_run:
        print(f"⚠ Warning: No exp3 summary file found for {model_name}")
        print(f"  Looking for: {pattern}")
        return False

    exp3_summary = exp3_files[0] if exp3_files else "exp3_mock_summary.json"

    cmd = [
        "python", "scripts/compare_models.py",
        str(results_dir / config['exp1_summary']),
        str(results_dir / config.get('exp2_summary', '')),
        str(exp3_summary)
    ]

    # Remove empty strings
    cmd = [c for c in cmd if c]

    success = run_command(
        cmd,
        f"Compare Exp1 vs Exp2 vs Exp3 for {model_name}",
        dry_run=dry_run
    )

    return success


def run_full_workflow(model_key, skip_annotation=False, dry_run=False):
    """
    Run complete Experiment 3 workflow for a model.

    Args:
        model_key: Model key from MODEL_CONFIG
        skip_annotation: Skip annotation step
        dry_run: Print commands without executing

    Returns:
        True if successful, False otherwise
    """
    config = MODEL_CONFIG[model_key]
    model_name = config['full_name']

    print_header(f"Experiment 3 Workflow: {model_name}")

    # Step 1: Run experiment with RAG
    results_file = run_experiment_3(model_key, dry_run=dry_run)
    if not results_file and not dry_run:
        return False

    # Step 2: Annotate results
    if not skip_annotation:
        success = annotate_results(
            results_file,
            config['judge'],
            dry_run=dry_run
        )
        if not success and not dry_run:
            return False
    else:
        print("\n⊳ Skipping annotation (--no-annotate flag)")

    # Step 3: Compare experiments
    success = compare_experiments(model_key, dry_run=dry_run)

    if success or dry_run:
        print_header(f"✓ Workflow Complete for {model_name}")
        return True
    else:
        print_header(f"✗ Workflow Failed for {model_name}")
        return False


def estimate_cost_and_time(models):
    """
    Estimate cost and time for running experiments.

    Args:
        models: List of model keys to run
    """
    print_header("Cost & Time Estimate")

    # Constants (approximate)
    TEST_CASES = 80
    AVG_LATENCY = {
        'o3': 14.0,
        'claude': 8.5,
        'grok': 10.0
    }
    COST_PER_CALL = {
        'o3': 0.15,  # $60/1M input + $240/1M output, ~1k tokens each
        'claude': 0.02,  # $3/1M input + $15/1M output
        'grok': 0.01   # Estimated
    }

    total_time = 0
    total_cost = 0

    print(f"Models to run: {', '.join(models)}")
    print(f"Test cases: {TEST_CASES}")
    print(f"\nPer-model estimates:")
    print("-" * 60)

    for model_key in models:
        config = MODEL_CONFIG[model_key]
        latency = AVG_LATENCY.get(model_key, 10.0)
        cost = COST_PER_CALL.get(model_key, 0.02)

        model_time = (TEST_CASES * latency) / 60  # minutes
        model_cost = TEST_CASES * cost

        # Add annotation cost (GPT-4o)
        annotation_cost = TEST_CASES * 0.01  # ~$0.01 per annotation
        model_cost += annotation_cost

        total_time += model_time
        total_cost += model_cost

        print(f"{config['full_name']:40s} ~{model_time:5.1f} min  ~${model_cost:5.2f}")

    print("-" * 60)
    print(f"{'TOTAL':40s} ~{total_time:5.1f} min  ~${total_cost:5.2f}")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Experiment 3 (RAG Comparison) Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models
  python scripts/run_exp3_rag_workflow.py --all

  # Run specific model
  python scripts/run_exp3_rag_workflow.py --model o3

  # Dry run (see commands without executing)
  python scripts/run_exp3_rag_workflow.py --model claude --dry-run

  # Skip annotation (annotate later manually)
  python scripts/run_exp3_rag_workflow.py --model grok --no-annotate
        """
    )

    parser.add_argument(
        '--model',
        choices=['o3', 'claude', 'grok'],
        help='Specific model to run'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all models (o3, claude, grok)'
    )
    parser.add_argument(
        '--no-annotate',
        action='store_true',
        help='Skip annotation step'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip prerequisite verification'
    )

    args = parser.parse_args()

    # Determine which models to run
    if args.all:
        models_to_run = ['o3', 'claude', 'grok']
    elif args.model:
        models_to_run = [args.model]
    else:
        print("Error: Specify --model <name> or --all")
        parser.print_help()
        return 1

    # Print welcome
    print_header("Experiment 3: RAG Comparison Workflow")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {', '.join(models_to_run)}")

    if args.dry_run:
        print("\n⊳ DRY RUN MODE - Commands will not be executed")

    # Estimate cost and time
    estimate_cost_and_time(models_to_run)

    # Verify prerequisites
    if not args.skip_verify and not args.dry_run:
        if not verify_prerequisites():
            return 1

    # Run workflow for each model
    success_count = 0
    for i, model_key in enumerate(models_to_run, 1):
        print(f"\n{'#'*80}")
        print(f"# Model {i}/{len(models_to_run)}: {MODEL_CONFIG[model_key]['full_name']}")
        print(f"{'#'*80}")

        success = run_full_workflow(
            model_key,
            skip_annotation=args.no_annotate,
            dry_run=args.dry_run
        )

        if success or args.dry_run:
            success_count += 1

    # Final summary
    print_header("Workflow Summary")
    print(f"Models processed: {success_count}/{len(models_to_run)}")

    if success_count == len(models_to_run) or args.dry_run:
        print("\n✓ All workflows completed successfully!")
        print("\nNext steps:")
        print("  1. Review results in results/compliance_experiments/")
        print("  2. Check visualizations in results/comparisons/")
        print("  3. Compare RAG vs non-RAG performance")
        print("  4. Update thesis with Experiment 3 findings")
        return 0
    else:
        print(f"\n⚠ {len(models_to_run) - success_count} workflow(s) failed")
        print("  Review error messages above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
