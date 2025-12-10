#!/usr/bin/env python3
"""
Compliance Annotation Tool

Manually annotate LLM responses for WHO guideline compliance.

Usage:
    python scripts/annotate_compliance_results.py <results_file.json>

Example:
    python scripts/annotate_compliance_results.py results/compliance_experiments/exp1_claude-opus-4-5-20251101_20251208_050610.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def load_results(file_path: str) -> Dict:
    """Load experiment results from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_annotations(results: Dict, output_path: str):
    """Save annotated results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Annotations saved to: {output_path}")


def annotate_response(response: Dict, index: int, total: int) -> Dict:
    """Interactively annotate a single response."""

    print("\n" + "=" * 80)
    print(f"Test Case {index + 1}/{total}: {response['test_case_id']}")
    print("=" * 80)
    print(f"Category: {response['category']}")
    print(f"Severity: {response['severity']}")
    print(f"\nScenario:\n{response['scenario']}")
    print(f"\n--- Model Response ---")
    print(response['model_response'])
    print("\n--- Ground Truth ---")
    print(response['ground_truth'])

    # Show compliance criteria
    print("\n--- Compliance Criteria ---")
    print("\nMust Include:")
    for item in response['compliant_criteria']['must_include']:
        print(f"  • {item}")
    print("\nMust Avoid:")
    for item in response['compliant_criteria']['must_avoid']:
        print(f"  • {item}")

    # Check if already annotated
    if 'annotation' in response:
        print(f"\n⚠ Already annotated as: {response['annotation']['compliance_score']}")
        skip = input("Skip this annotation? (y/n): ").strip().lower()
        if skip == 'y':
            return response

    # Get annotation
    print("\n" + "-" * 80)
    print("ANNOTATION:")

    # Compliance score
    while True:
        score = input("\nCompliance Score (0=Non-compliant, 1=Partially, 2=Fully compliant): ").strip()
        if score in ['0', '1', '2']:
            compliance_score = int(score)
            break
        print("Invalid input. Enter 0, 1, or 2.")

    # Critical safety issue
    while True:
        critical = input("Critical Safety Issue? (y/n): ").strip().lower()
        if critical in ['y', 'n']:
            has_critical_issue = (critical == 'y')
            break
        print("Invalid input. Enter y or n.")

    # Notes
    notes = input("Notes (optional, press Enter to skip): ").strip()

    # Which criteria were met
    criteria_met = []
    print("\nWhich 'must include' criteria were met? (Enter numbers separated by commas, or 'none'):")
    for i, item in enumerate(response['compliant_criteria']['must_include']):
        print(f"  {i+1}. {item}")
    met_input = input("Met criteria: ").strip()
    if met_input and met_input.lower() != 'none':
        try:
            indices = [int(x.strip()) - 1 for x in met_input.split(',')]
            criteria_met = [response['compliant_criteria']['must_include'][i] for i in indices
                           if 0 <= i < len(response['compliant_criteria']['must_include'])]
        except:
            print("Invalid input, skipping criteria_met")

    # Which criteria were violated
    criteria_violated = []
    print("\nWhich 'must avoid' items were violated? (Enter numbers separated by commas, or 'none'):")
    for i, item in enumerate(response['compliant_criteria']['must_avoid']):
        print(f"  {i+1}. {item}")
    violated_input = input("Violated items: ").strip()
    if violated_input and violated_input.lower() != 'none':
        try:
            indices = [int(x.strip()) - 1 for x in violated_input.split(',')]
            criteria_violated = [response['compliant_criteria']['must_avoid'][i] for i in indices
                                if 0 <= i < len(response['compliant_criteria']['must_avoid'])]
        except:
            print("Invalid input, skipping criteria_violated")

    # Add annotation to response
    response['annotation'] = {
        'compliance_score': compliance_score,
        'has_critical_safety_issue': has_critical_issue,
        'criteria_met': criteria_met,
        'criteria_violated': criteria_violated,
        'notes': notes
    }

    print("\n✓ Annotation saved for this response")
    return response


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/annotate_compliance_results.py <results_file.json>")
        print("\nAvailable result files:")
        results_dir = Path("results/compliance_experiments")
        if results_dir.exists():
            for file in sorted(results_dir.glob("*.json")):
                print(f"  {file}")
        sys.exit(1)

    input_file = sys.argv[1]

    # Load results
    print(f"Loading results from: {input_file}")
    results = load_results(input_file)

    metadata = results['experiment_metadata']
    responses = results['responses']

    print(f"\n{'='*80}")
    print(f"Annotating Experiment {metadata['experiment_number']}")
    print(f"Model: {metadata['model']}")
    print(f"Total responses: {len(responses)}")
    print(f"{'='*80}")

    # Filter out errors
    valid_responses = [r for r in responses if not r['model_response'].startswith('[ERROR')]
    error_count = len(responses) - len(valid_responses)

    print(f"\nValid responses: {len(valid_responses)}")
    print(f"Errors/blocked: {error_count}")

    # Check how many already annotated
    annotated_count = sum(1 for r in valid_responses if 'annotation' in r)
    print(f"Already annotated: {annotated_count}/{len(valid_responses)}")

    # Start annotation
    proceed = input("\nStart annotation? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Annotation cancelled.")
        sys.exit(0)

    # Annotate each response
    for i, response in enumerate(valid_responses):
        annotate_response(response, i, len(valid_responses))

        # Save every 10 responses
        if (i + 1) % 10 == 0:
            output_path = input_file.replace('.json', '_annotated.json')
            save_annotations(results, output_path)
            print(f"\n✓ Progress saved ({i + 1}/{len(valid_responses)} annotated)")

    # Save final annotations
    output_path = input_file.replace('.json', '_annotated.json')
    save_annotations(results, output_path)

    print("\n" + "="*80)
    print("✓ Annotation Complete!")
    print(f"Annotated: {len(valid_responses)} responses")
    print(f"Output: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
