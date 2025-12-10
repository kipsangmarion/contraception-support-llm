#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Experiment 2 Workflow

Runs the complete Experiment 2 pipeline:
1. Run compliance experiments with enhanced prompts
2. Auto-annotate results with LLM judge
3. Analyze results
4. Compare with Experiment 1

Usage:
    # Run for top 2 models (o3 + Claude) - RECOMMENDED
    python scripts/run_exp2_workflow.py --models o3 claude

    # Run for specific model only
    python scripts/run_exp2_workflow.py --models o3

    # Run for all models including Grok
    python scripts/run_exp2_workflow.py --models o3 claude grok

Author: Research Team
Date: December 8, 2025
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Model configurations
MODEL_CONFIG = {
    'o3': {
        'full_name': 'o3-2025-04-16',
        'judge': 'gpt-4o',  # Use GPT-4o for all annotations (consistent with Exp 1)
        'exp1_summary': 'exp1_o3-2025-04-16_20251208_110608_auto_summary.json'
    },
    'claude': {
        'full_name': 'claude-opus-4-5-20251101',
        'judge': 'gpt-4o',  # Use GPT-4o for all annotations (consistent with Exp 1)
        'exp1_summary': 'exp1_claude-opus-4-5-20251101_20251208_050610_auto_summary.json'
    },
    'grok': {
        'full_name': 'grok-4-1-fast-reasoning',
        'judge': 'gpt-4o',  # Use GPT-4o for all annotations (consistent with Exp 1)
        'exp1_summary': 'exp1_grok-4-1-fast-reasoning_20251208_053400_auto_summary.json'
    }
}


def run_command(cmd, description, check=True):
    """Run a command and print output."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=check, capture_output=False, text=True)
        elapsed = time.time() - start_time

        print(f"\n✓ Completed in {elapsed:.1f}s")
        return result.returncode == 0

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False


def find_latest_file(pattern):
    """Find the most recently created file matching pattern."""
    files = list(Path("results/compliance_experiments").glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def run_experiment(model_key):
    """Run full Experiment 2 workflow for a model."""

    if model_key not in MODEL_CONFIG:
        print(f"✗ Unknown model: {model_key}")
        return False

    config = MODEL_CONFIG[model_key]
    model_name = config['full_name']
    judge_model = config['judge']
    exp1_summary = config['exp1_summary']

    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT 2: {model_name}")
    print(f"{'#'*80}")

    # Step 1: Run experiment
    step1_success = run_command(
        ['python', 'scripts/run_compliance_experiments.py',
         '--experiment', '2', '--model', model_name],
        f"Step 1: Running Experiment 2 for {model_name}"
    )

    if not step1_success:
        print(f"✗ Experiment 2 failed for {model_name}")
        return False

    # Step 2: Find the newly created results file
    print(f"\nSearching for results file...")
    time.sleep(2)  # Give filesystem time to update

    exp2_file = find_latest_file(f"exp2_{model_name}_*.json")

    if not exp2_file:
        print(f"✗ Could not find results file for {model_name}")
        return False

    # Skip if already annotated
    if '_annotated' in exp2_file.name or '_summary' in exp2_file.name:
        print(f"Found: {exp2_file.name}")
        print("⚠ File appears to be already processed, searching for raw file...")
        # Find non-annotated version
        pattern = f"exp2_{model_name}_*.json"
        candidates = [f for f in Path("results/compliance_experiments").glob(pattern)
                     if '_annotated' not in f.name and '_summary' not in f.name]
        if candidates:
            exp2_file = max(candidates, key=lambda p: p.stat().st_mtime)
            print(f"Using: {exp2_file.name}")
        else:
            print("✗ No raw results file found")
            return False
    else:
        print(f"Found: {exp2_file.name}")

    # Step 3: Auto-annotate
    step3_success = run_command(
        ['python', 'scripts/auto_annotate_with_llm.py',
         str(exp2_file), '--judge-model', judge_model],
        f"Step 2: Auto-annotating with {judge_model}"
    )

    if not step3_success:
        print(f"✗ Auto-annotation failed for {model_name}")
        return False

    # Step 4: Find annotated file
    annotated_file = exp2_file.parent / exp2_file.name.replace('.json', '_auto_annotated.json')

    if not annotated_file.exists():
        print(f"✗ Annotated file not found: {annotated_file}")
        return False

    print(f"Found annotated file: {annotated_file.name}")

    # Step 5: Analyze results
    step5_success = run_command(
        ['python', 'scripts/analyze_compliance_results.py', str(annotated_file)],
        f"Step 3: Analyzing {model_name} results"
    )

    if not step5_success:
        print(f"✗ Analysis failed for {model_name}")
        return False

    # Step 6: Find summary file
    summary_file = annotated_file.parent / annotated_file.name.replace('_auto_annotated.json', '_auto_summary.json')

    if not summary_file.exists():
        print(f"✗ Summary file not found: {summary_file}")
        return False

    print(f"Found summary file: {summary_file.name}")

    # Step 7: Compare with Experiment 1
    exp1_path = Path("results/compliance_experiments") / exp1_summary

    if not exp1_path.exists():
        print(f"⚠ Warning: Experiment 1 summary not found: {exp1_path}")
        print("  Skipping comparison with Experiment 1")
        return True

    step7_success = run_command(
        ['python', 'scripts/compare_models.py', str(exp1_path), str(summary_file)],
        f"Step 4: Comparing Experiment 1 vs Experiment 2 for {model_name}"
    )

    if not step7_success:
        print(f"✗ Comparison failed for {model_name}")
        return False

    print(f"\n{'='*80}")
    print(f"✓ COMPLETE: Experiment 2 workflow for {model_name}")
    print(f"{'='*80}")
    print(f"\nResults:")
    print(f"  - Raw results: {exp2_file}")
    print(f"  - Annotated: {annotated_file}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Comparison plots: results/comparisons/")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run complete Experiment 2 workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for o3 and Claude (recommended)
  python scripts/run_exp2_workflow.py --models o3 claude

  # Run for o3 only
  python scripts/run_exp2_workflow.py --models o3

  # Run for all models
  python scripts/run_exp2_workflow.py --models o3 claude grok
        """
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=['o3', 'claude', 'grok'],
        required=True,
        help='Models to run (o3, claude, grok)'
    )

    args = parser.parse_args()

    print("="*80)
    print("AUTOMATED EXPERIMENT 2 WORKFLOW")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {', '.join(args.models)}")
    print()

    # Cost estimates
    costs = {
        'o3': '$2.50',
        'claude': '$0.40',
        'grok': '$0.20'
    }

    total_cost = sum([2.5 if m == 'o3' else 0.4 if m == 'claude' else 0.2
                     for m in args.models])
    annotation_cost = len(args.models) * 0.5

    print("COST ESTIMATE:")
    for model in args.models:
        print(f"  {model}: ~{costs[model]} (experiment) + $0.50 (annotation)")
    print(f"  TOTAL: ~${total_cost + annotation_cost:.2f}")
    print()

    # Time estimates
    times = {
        'o3': 20,
        'claude': 12,
        'grok': 15
    }

    total_time = sum([times[m] for m in args.models]) + len(args.models) * 5

    print("TIME ESTIMATE:")
    for model in args.models:
        print(f"  {model}: ~{times[model]} min (experiment) + 5 min (annotation/analysis)")
    print(f"  TOTAL: ~{total_time} minutes")
    print()

    input("Press ENTER to continue or Ctrl+C to cancel...")

    # Run experiments
    results = {}
    start_time = time.time()

    for model in args.models:
        success = run_experiment(model)
        results[model] = success

        if not success:
            print(f"\n⚠ Warning: Workflow failed for {model}")
            print("Continuing with remaining models...")

    # Final summary
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print()

    print("Results:")
    for model, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status} - {model}")

    print()

    successful = [m for m, s in results.items() if s]

    if successful:
        print("✓ Experiment 2 complete for:", ', '.join(successful))
        print()
        print("Next steps:")
        print("  1. Check results/comparisons/ for Exp1 vs Exp2 visualizations")
        print("  2. Review improvements in compliance scores")
        print("  3. Write up results section for thesis")
        print()
        print("See RESULTS_INTERPRETATION.md for guidance on interpreting results.")
        return 0
    else:
        print("✗ All experiments failed")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
