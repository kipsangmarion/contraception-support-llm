#!/usr/bin/env python3
"""
Compliance Results Analysis

Analyze annotated compliance experiment results and generate statistics.

Usage:
    python scripts/analyze_compliance_results.py <annotated_results.json>

Example:
    python scripts/analyze_compliance_results.py results/compliance_experiments/exp1_claude_annotated.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_results(file_path: str) -> Dict:
    """Load annotated results from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_compliance(results: Dict) -> Dict:
    """Analyze compliance scores and generate statistics."""

    responses = results['responses']
    metadata = results['experiment_metadata']

    # Filter valid responses with annotations
    annotated = [r for r in responses
                 if not r['model_response'].startswith('[ERROR') and 'annotation' in r]

    if not annotated:
        print("⚠ No annotated responses found!")
        return {}

    print(f"\n{'='*80}")
    print(f"Compliance Analysis: {metadata['model']}")
    print(f"Experiment {metadata['experiment_number']}")
    print(f"{'='*80}")

    # Overall statistics
    total = len(annotated)
    fully_compliant = sum(1 for r in annotated if r['annotation']['compliance_score'] == 2)
    partially_compliant = sum(1 for r in annotated if r['annotation']['compliance_score'] == 1)
    non_compliant = sum(1 for r in annotated if r['annotation']['compliance_score'] == 0)

    critical_issues = sum(1 for r in annotated if r['annotation']['has_critical_safety_issue'])

    print(f"\nOverall Compliance:")
    print(f"  Total responses analyzed: {total}")
    print(f"  Fully compliant: {fully_compliant} ({fully_compliant/total*100:.1f}%)")
    print(f"  Partially compliant: {partially_compliant} ({partially_compliant/total*100:.1f}%)")
    print(f"  Non-compliant: {non_compliant} ({non_compliant/total*100:.1f}%)")
    print(f"  Critical safety issues: {critical_issues} ({critical_issues/total*100:.1f}%)")

    # Average compliance score
    avg_score = sum(r['annotation']['compliance_score'] for r in annotated) / total
    print(f"\n  Average compliance score: {avg_score:.2f}/2.0")

    # By category
    print(f"\nCompliance by Category:")
    categories = {}
    for r in annotated:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r['annotation']['compliance_score'])

    for cat, scores in sorted(categories.items()):
        avg = sum(scores) / len(scores)
        print(f"  {cat}: {avg:.2f}/2.0 (n={len(scores)})")

    # By severity
    print(f"\nCompliance by Severity:")
    severities = {}
    for r in annotated:
        sev = r.get('severity', 'N/A')
        if sev not in severities:
            severities[sev] = []
        severities[sev].append(r['annotation']['compliance_score'])

    for sev, scores in sorted(severities.items()):
        avg = sum(scores) / len(scores)
        print(f"  {sev}: {avg:.2f}/2.0 (n={len(scores)})")

    # Critical issues breakdown
    if critical_issues > 0:
        print(f"\nCritical Safety Issues ({critical_issues} total):")
        for r in annotated:
            if r['annotation']['has_critical_safety_issue']:
                print(f"  - {r['test_case_id']}: {r['category']}")
                if r['annotation'].get('notes'):
                    print(f"    Notes: {r['annotation']['notes']}")

    # Most common violations
    print(f"\nMost Common Violations:")
    all_violations = []
    for r in annotated:
        all_violations.extend(r['annotation'].get('criteria_violated', []))

    if all_violations:
        from collections import Counter
        violation_counts = Counter(all_violations)
        for violation, count in violation_counts.most_common(10):
            print(f"  {count}x - {violation}")
    else:
        print("  None")

    # Latency statistics
    latencies = [r['latency_seconds'] for r in annotated]
    print(f"\nResponse Latency:")
    print(f"  Average: {sum(latencies)/len(latencies):.2f}s")
    print(f"  Min: {min(latencies):.2f}s")
    print(f"  Max: {max(latencies):.2f}s")

    print(f"\n{'='*80}\n")

    return {
        'model': metadata['model'],
        'experiment': metadata['experiment_number'],
        'total_responses': total,
        'fully_compliant': fully_compliant,
        'partially_compliant': partially_compliant,
        'non_compliant': non_compliant,
        'critical_issues': critical_issues,
        'avg_compliance_score': avg_score,
        'avg_latency': sum(latencies)/len(latencies),
        'by_category': {cat: sum(scores)/len(scores) for cat, scores in categories.items()},
        'by_severity': {sev: sum(scores)/len(scores) for sev, scores in severities.items()}
    }


def export_to_csv(results: Dict, output_path: str):
    """Export results to CSV for further analysis."""

    responses = results['responses']
    annotated = [r for r in responses
                 if not r['model_response'].startswith('[ERROR') and 'annotation' in r]

    # Flatten for CSV
    rows = []
    for r in annotated:
        row = {
            'test_case_id': r['test_case_id'],
            'category': r['category'],
            'severity': r.get('severity', 'N/A'),
            'scenario': r['scenario'],
            'model_response': r['model_response'],
            'compliance_score': r['annotation']['compliance_score'],
            'has_critical_issue': r['annotation']['has_critical_safety_issue'],
            'latency_seconds': r['latency_seconds'],
            'notes': r['annotation'].get('notes', '')
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ CSV exported to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_compliance_results.py <annotated_results.json>")
        print("\nLooking for annotated files...")
        results_dir = Path("results/compliance_experiments")
        if results_dir.exists():
            for file in sorted(results_dir.glob("*_annotated.json")):
                print(f"  {file}")
        sys.exit(1)

    input_file = sys.argv[1]

    # Load results
    print(f"Loading annotated results from: {input_file}")
    results = load_results(input_file)

    # Analyze
    stats = analyze_compliance(results)

    if stats:
        # Export CSV
        csv_path = input_file.replace('_annotated.json', '_analysis.csv')
        export_to_csv(results, csv_path)

        # Save summary stats
        summary_path = input_file.replace('_annotated.json', '_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Summary stats saved to: {summary_path}")


if __name__ == "__main__":
    main()
