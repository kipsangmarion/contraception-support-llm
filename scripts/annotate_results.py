#!/usr/bin/env python3
"""
Unified Annotation Script

Consolidates all annotation functionality:
- LLM-as-judge annotation
- Inter-annotator agreement calculation
- Batch annotation

Usage:
    python scripts/annotate_results.py --mode llm <results_file.json>
    python scripts/annotate_results.py --mode agreement <annotated1.json> <annotated2.json>
    python scripts/annotate_results.py --mode batch <results_dir>
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.annotator import ComplianceAnnotator
from src.evaluation.agreement import AgreementMetrics


def annotate_with_llm(
    input_file: Path,
    output_file: Path,
    judge_model: str = "gpt-4o"
):
    """
    Annotate results using LLM-as-judge.

    Args:
        input_file: Input results JSON file
        output_file: Output annotated JSON file
        judge_model: Judge model to use
    """
    print("\n" + "="*80)
    print("LLM-AS-JUDGE ANNOTATION")
    print("="*80)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Judge: {judge_model}")

    # Load results
    print("\nLoading results...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    responses = data.get('responses', [])
    print(f"Found {len(responses)} responses")

    # Initialize annotator
    print(f"\nInitializing {judge_model}...")
    annotator = ComplianceAnnotator(judge_model=judge_model)

    # Count already annotated
    already_annotated = sum(
        1 for r in responses
        if 'annotation' in r and r['annotation'].get('compliance_score', -1) >= 0
    )
    to_annotate = len(responses) - already_annotated

    if already_annotated > 0:
        print(f"Found {already_annotated} already annotated responses")

    if to_annotate == 0:
        print("\nAll responses already annotated!")
        return

    print(f"Will annotate {to_annotate} responses")

    # Progress callback
    def progress(current, total):
        pct = current / total * 100
        print(f"Progress: {current}/{total} ({pct:.1f}%)", end='\r')

    # Annotate
    print("\nAnnotating...")
    annotated_responses = annotator.batch_annotate(
        responses,
        delay=1.0,
        progress_callback=progress
    )
    print()  # New line after progress

    # Update data
    data['responses'] = annotated_responses

    # Save
    print(f"\nSaving to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Summary
    fully_compliant = sum(
        1 for r in annotated_responses
        if r.get('annotation', {}).get('compliance_score') == 2
    )
    partially_compliant = sum(
        1 for r in annotated_responses
        if r.get('annotation', {}).get('compliance_score') == 1
    )
    non_compliant = sum(
        1 for r in annotated_responses
        if r.get('annotation', {}).get('compliance_score') == 0
    )
    critical_issues = sum(
        1 for r in annotated_responses
        if r.get('annotation', {}).get('has_critical_safety_issue', False)
    )

    print("\n" + "-"*80)
    print("ANNOTATION SUMMARY")
    print("-"*80)
    print(f"Total Responses: {len(annotated_responses)}")
    print(f"Fully Compliant (score=2): {fully_compliant} ({fully_compliant/len(annotated_responses)*100:.1f}%)")
    print(f"Partially Compliant (score=1): {partially_compliant} ({partially_compliant/len(annotated_responses)*100:.1f}%)")
    print(f"Non-Compliant (score=0): {non_compliant} ({non_compliant/len(annotated_responses)*100:.1f}%)")
    print(f"Critical Safety Issues: {critical_issues}")

    print("\n[COMPLETE] Annotation complete!")


def calculate_agreement(
    file1: Path,
    file2: Path,
    output_file: Path
):
    """
    Calculate inter-annotator agreement.

    Args:
        file1: First annotated file
        file2: Second annotated file
        output_file: Output agreement metrics JSON
    """
    print("\n" + "="*80)
    print("INTER-ANNOTATOR AGREEMENT")
    print("="*80)
    print(f"Annotator 1: {file1}")
    print(f"Annotator 2: {file2}")

    # Load files
    print("\nLoading annotations...")
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    responses1 = data1.get('responses', [])
    responses2 = data2.get('responses', [])

    # Extract scores for matching test cases
    scores1 = []
    scores2 = []

    # Build lookup by test_case_id
    lookup1 = {
        r['test_case_id']: r.get('annotation', {}).get('compliance_score', -1)
        for r in responses1
    }
    lookup2 = {
        r['test_case_id']: r.get('annotation', {}).get('compliance_score', -1)
        for r in responses2
    }

    # Get common test cases
    common_ids = set(lookup1.keys()) & set(lookup2.keys())

    for test_id in sorted(common_ids):
        score1 = lookup1[test_id]
        score2 = lookup2[test_id]

        # Skip if either is invalid
        if score1 < 0 or score2 < 0:
            continue

        scores1.append(score1)
        scores2.append(score2)

    if len(scores1) == 0:
        print("\nNo common annotated test cases found!")
        return

    print(f"Found {len(scores1)} common annotated test cases")

    # Calculate metrics
    print("\nCalculating agreement metrics...")
    metrics = AgreementMetrics.calculate_agreement_metrics(scores1, scores2)

    # Print results
    print("\n" + "-"*80)
    print("AGREEMENT METRICS")
    print("-"*80)
    print(f"Cohen's Kappa: {metrics['cohens_kappa']:.4f}")
    print(f"Interpretation: {metrics['interpretation']}")
    print(f"Percent Agreement: {metrics['percent_agreement']:.1f}%")
    print(f"Number of Items: {metrics['n_items']}")

    print("\nConfusion Matrix:")
    print("Rows = Annotator 1, Cols = Annotator 2")
    print("       0    1    2")
    for i, row in enumerate(metrics['confusion_matrix']):
        print(f"  {i}:  {row[0]:3d}  {row[1]:3d}  {row[2]:3d}")

    # Save metrics
    if output_file:
        print(f"\nSaving metrics to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

    print("\n[COMPLETE] Agreement calculation complete!")


def batch_annotate(
    results_dir: Path,
    judge_model: str = "gpt-4o"
):
    """
    Batch annotate all files in a directory.

    Args:
        results_dir: Directory containing result JSON files
        judge_model: Judge model to use
    """
    print("\n" + "="*80)
    print("BATCH ANNOTATION")
    print("="*80)
    print(f"Directory: {results_dir}")
    print(f"Judge: {judge_model}")

    # Find all JSON files
    json_files = list(results_dir.glob("*.json"))

    # Filter out already annotated files
    unannotated = [
        f for f in json_files
        if not f.stem.endswith('_auto_annotated')
    ]

    print(f"\nFound {len(unannotated)} files to annotate")

    for i, input_file in enumerate(unannotated, 1):
        print(f"\n{'='*80}")
        print(f"File {i}/{len(unannotated)}: {input_file.name}")
        print("="*80)

        # Create output filename
        output_file = input_file.parent / f"{input_file.stem}_auto_annotated.json"

        # Skip if output already exists
        if output_file.exists():
            print(f"Output already exists, skipping: {output_file}")
            continue

        # Annotate
        try:
            annotate_with_llm(input_file, output_file, judge_model)
        except Exception as e:
            print(f"\n[ERROR] Failed to annotate {input_file.name}: {e}")
            continue

    print("\n" + "="*80)
    print("BATCH ANNOTATION COMPLETE")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Unified annotation tool")
    parser.add_argument(
        '--mode',
        choices=['llm', 'agreement', 'batch'],
        required=True,
        help='Annotation mode'
    )
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input file(s) or directory'
    )
    parser.add_argument(
        '--judge-model',
        default='gpt-4o',
        help='Judge model for annotation (default: gpt-4o)'
    )
    parser.add_argument(
        '--output',
        help='Output file (auto-generated if not specified)'
    )

    args = parser.parse_args()

    if args.mode == 'llm':
        # Single file LLM annotation
        if len(args.input_files) != 1:
            print("Error: --mode llm requires exactly 1 input file")
            return 1

        input_file = Path(args.input_files[0])
        if not input_file.exists():
            print(f"Error: File not found: {input_file}")
            return 1

        # Auto-generate output filename
        if args.output:
            output_file = Path(args.output)
        else:
            output_file = input_file.parent / f"{input_file.stem}_auto_annotated.json"

        annotate_with_llm(input_file, output_file, args.judge_model)

    elif args.mode == 'agreement':
        # Agreement calculation
        if len(args.input_files) != 2:
            print("Error: --mode agreement requires exactly 2 input files")
            return 1

        file1 = Path(args.input_files[0])
        file2 = Path(args.input_files[1])

        if not file1.exists() or not file2.exists():
            print("Error: One or both files not found")
            return 1

        # Auto-generate output filename
        if args.output:
            output_file = Path(args.output)
        else:
            output_file = file1.parent / "agreement_metrics.json"

        calculate_agreement(file1, file2, output_file)

    elif args.mode == 'batch':
        # Batch annotation
        if len(args.input_files) != 1:
            print("Error: --mode batch requires exactly 1 directory")
            return 1

        results_dir = Path(args.input_files[0])
        if not results_dir.is_dir():
            print(f"Error: Not a directory: {results_dir}")
            return 1

        batch_annotate(results_dir, args.judge_model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
