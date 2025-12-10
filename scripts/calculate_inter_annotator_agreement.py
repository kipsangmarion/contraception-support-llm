#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Inter-Annotator Agreement

Compare annotations from two different judge models to assess reliability.

Usage:
    # Annotate with second judge (if not done yet)
    python scripts/auto_annotate_with_llm.py results/compliance_experiments/exp1_o3_*.json --judge-model claude-3-5-sonnet-20241022 --output-suffix claude_judged

    # Calculate agreement
    python scripts/calculate_inter_annotator_agreement.py results/compliance_experiments/exp1_o3_auto_annotated.json results/compliance_experiments/exp1_o3_claude_judged_annotated.json

This calculates:
- Cohen's Kappa (agreement beyond chance)
- Percentage agreement
- Confusion matrix
- Disagreement analysis
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def load_annotated_results(file_path: Path) -> List[Dict]:
    """Load annotated results."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Handle both 'results' and 'responses' keys
    if 'results' in data:
        return data['results']
    elif 'responses' in data:
        return data['responses']
    else:
        raise KeyError(f"Expected 'results' or 'responses' key in JSON file, found: {list(data.keys())}")


def extract_annotations(results: List[Dict]) -> Tuple[List[int], List[bool]]:
    """Extract compliance scores and critical safety flags."""
    scores = []
    critical = []

    for r in results:
        if 'annotation' in r:
            scores.append(r['annotation'].get('compliance_score', -1))
            critical.append(r['annotation'].get('has_critical_safety_issue', False))
        else:
            scores.append(-1)
            critical.append(False)

    return scores, critical


def calculate_cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Calculate Cohen's Kappa for agreement between two annotators.

    Kappa interpretation:
    < 0: Poor agreement (worse than chance)
    0.0-0.20: Slight agreement
    0.21-0.40: Fair agreement
    0.41-0.60: Moderate agreement
    0.61-0.80: Substantial agreement
    0.81-1.00: Almost perfect agreement
    """
    # Filter out missing values
    valid_idx = (y1 >= 0) & (y2 >= 0)
    y1 = y1[valid_idx]
    y2 = y2[valid_idx]

    if len(y1) == 0:
        return 0.0

    # Calculate observed agreement
    po = np.mean(y1 == y2)

    # Calculate expected agreement
    unique_values = np.unique(np.concatenate([y1, y2]))
    pe = sum([np.mean(y1 == v) * np.mean(y2 == v) for v in unique_values])

    # Cohen's Kappa
    if pe == 1:
        return 1.0
    kappa = (po - pe) / (1 - pe)

    return kappa


def calculate_agreement_metrics(scores1: List[int], scores2: List[int],
                                 critical1: List[bool], critical2: List[bool]) -> Dict:
    """Calculate various agreement metrics."""

    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    critical1 = np.array(critical1)
    critical2 = np.array(critical2)

    # Filter valid annotations
    valid_idx = (scores1 >= 0) & (scores2 >= 0)
    valid_scores1 = scores1[valid_idx]
    valid_scores2 = scores2[valid_idx]
    valid_critical1 = critical1[valid_idx]
    valid_critical2 = critical2[valid_idx]

    # Calculate metrics
    metrics = {
        'total_responses': len(scores1),
        'valid_annotations': np.sum(valid_idx),

        # Score agreement
        'score_exact_agreement': np.mean(valid_scores1 == valid_scores2),
        'score_within_1': np.mean(np.abs(valid_scores1 - valid_scores2) <= 1),
        'score_cohens_kappa': calculate_cohens_kappa(scores1, scores2),
        'score_mean_diff': np.mean(valid_scores2 - valid_scores1),

        # Critical issue agreement
        'critical_agreement': np.mean(valid_critical1 == valid_critical2),
        'critical_cohens_kappa': calculate_cohens_kappa(
            critical1.astype(int), critical2.astype(int)
        ),

        # Distribution
        'judge1_avg_score': np.mean(valid_scores1),
        'judge2_avg_score': np.mean(valid_scores2),
        'judge1_critical_rate': np.mean(valid_critical1),
        'judge2_critical_rate': np.mean(valid_critical2),
    }

    return metrics


def create_confusion_matrix(scores1: np.ndarray, scores2: np.ndarray,
                            output_path: Path, title: str):
    """Create confusion matrix visualization."""

    # Filter valid annotations
    valid_idx = (scores1 >= 0) & (scores2 >= 0)
    scores1 = scores1[valid_idx]
    scores2 = scores2[valid_idx]

    # Create confusion matrix
    unique_scores = sorted(np.unique(np.concatenate([scores1, scores2])))
    matrix = np.zeros((len(unique_scores), len(unique_scores)))

    for s1, s2 in zip(scores1, scores2):
        i = unique_scores.index(s1)
        j = unique_scores.index(s2)
        matrix[i, j] += 1

    # Normalize by row
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_normalized = np.divide(matrix, row_sums, where=row_sums != 0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(matrix_normalized, annot=matrix.astype(int), fmt='d',
                cmap='Blues', xticklabels=unique_scores, yticklabels=unique_scores,
                ax=ax, cbar_kws={'label': 'Proportion'})

    ax.set_xlabel('Judge 2 Score')
    ax.set_ylabel('Judge 1 Score')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def identify_disagreements(results1: List[Dict], results2: List[Dict],
                          threshold: int = 1) -> List[Dict]:
    """Identify cases where annotators disagreed significantly."""

    disagreements = []

    for r1, r2 in zip(results1, results2):
        if 'annotation' not in r1 or 'annotation' not in r2:
            continue

        score1 = r1['annotation'].get('compliance_score', -1)
        score2 = r2['annotation'].get('compliance_score', -1)

        if score1 < 0 or score2 < 0:
            continue

        diff = abs(score1 - score2)

        if diff >= threshold:
            disagreements.append({
                'test_case_id': r1.get('test_case_id', 'unknown'),
                'scenario': r1.get('scenario', '')[:100] + '...',
                'judge1_score': score1,
                'judge2_score': score2,
                'difference': diff,
                'judge1_critical': r1['annotation'].get('has_critical_safety_issue', False),
                'judge2_critical': r2['annotation'].get('has_critical_safety_issue', False),
            })

    return disagreements


def generate_report(metrics: Dict, disagreements: List[Dict],
                   judge1_name: str, judge2_name: str,
                   output_dir: Path):
    """Generate inter-annotator agreement report."""

    report_path = output_dir / 'inter_annotator_agreement_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("INTER-ANNOTATOR AGREEMENT REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Judge 1: {judge1_name}\n")
        f.write(f"Judge 2: {judge2_name}\n")
        f.write(f"Total responses: {metrics['total_responses']}\n")
        f.write(f"Valid annotations: {metrics['valid_annotations']}\n\n")

        f.write("-"*80 + "\n")
        f.write("COMPLIANCE SCORE AGREEMENT\n")
        f.write("-"*80 + "\n\n")

        f.write(f"Exact agreement: {metrics['score_exact_agreement']*100:.1f}%\n")
        f.write(f"Within 1 point: {metrics['score_within_1']*100:.1f}%\n")
        f.write(f"Cohen's Kappa: {metrics['score_cohens_kappa']:.3f}")

        # Kappa interpretation
        kappa = metrics['score_cohens_kappa']
        if kappa < 0:
            interp = "Poor (worse than chance)"
        elif kappa < 0.21:
            interp = "Slight"
        elif kappa < 0.41:
            interp = "Fair"
        elif kappa < 0.61:
            interp = "Moderate"
        elif kappa < 0.81:
            interp = "Substantial"
        else:
            interp = "Almost perfect"

        f.write(f" ({interp})\n")
        f.write(f"Mean difference (Judge2 - Judge1): {metrics['score_mean_diff']:.3f}\n\n")

        f.write("-"*80 + "\n")
        f.write("CRITICAL SAFETY ISSUE AGREEMENT\n")
        f.write("-"*80 + "\n\n")

        f.write(f"Agreement: {metrics['critical_agreement']*100:.1f}%\n")
        f.write(f"Cohen's Kappa: {metrics['critical_cohens_kappa']:.3f}\n\n")

        f.write("-"*80 + "\n")
        f.write("JUDGE STATISTICS\n")
        f.write("-"*80 + "\n\n")

        f.write(f"Judge 1 average score: {metrics['judge1_avg_score']:.2f}/2.0\n")
        f.write(f"Judge 2 average score: {metrics['judge2_avg_score']:.2f}/2.0\n")
        f.write(f"Judge 1 critical rate: {metrics['judge1_critical_rate']*100:.1f}%\n")
        f.write(f"Judge 2 critical rate: {metrics['judge2_critical_rate']*100:.1f}%\n\n")

        if disagreements:
            f.write("-"*80 + "\n")
            f.write(f"DISAGREEMENTS (n={len(disagreements)})\n")
            f.write("-"*80 + "\n\n")

            for i, d in enumerate(disagreements[:10], 1):  # Show top 10
                f.write(f"{i}. Test case: {d['test_case_id']}\n")
                f.write(f"   Scenario: {d['scenario']}\n")
                f.write(f"   Judge 1: {d['judge1_score']}/2.0 (Critical: {d['judge1_critical']})\n")
                f.write(f"   Judge 2: {d['judge2_score']}/2.0 (Critical: {d['judge2_critical']})\n")
                f.write(f"   Difference: {d['difference']}\n\n")

            if len(disagreements) > 10:
                f.write(f"   ... and {len(disagreements) - 10} more disagreements\n\n")

    print(f"✓ Saved: {report_path}")

    # Also save disagreements as CSV
    if disagreements:
        df = pd.DataFrame(disagreements)
        csv_path = output_dir / 'disagreements.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/calculate_inter_annotator_agreement.py <annotated_file1.json> <annotated_file2.json>")
        print("\nExample:")
        print("  python scripts/calculate_inter_annotator_agreement.py \\")
        print("      results/compliance_experiments/exp1_o3_auto_annotated.json \\")
        print("      results/compliance_experiments/exp1_o3_claude_judged_annotated.json")
        return 1

    file1 = Path(sys.argv[1])
    file2 = Path(sys.argv[2])

    if not file1.exists():
        print(f"✗ File not found: {file1}")
        return 1

    if not file2.exists():
        print(f"✗ File not found: {file2}")
        return 1

    print("="*80)
    print("INTER-ANNOTATOR AGREEMENT CALCULATION")
    print("="*80)
    print()

    # Extract judge names from filenames
    judge1_name = "Judge 1"
    judge2_name = "Judge 2"

    if "gpt" in file1.name.lower():
        judge1_name = "GPT-4o"
    elif "claude" in file1.name.lower():
        judge1_name = "Claude Sonnet"

    if "gpt" in file2.name.lower():
        judge2_name = "GPT-4o"
    elif "claude" in file2.name.lower():
        judge2_name = "Claude Sonnet"

    print(f"Judge 1: {judge1_name} ({file1.name})")
    print(f"Judge 2: {judge2_name} ({file2.name})")
    print()

    # Load results
    print("Loading annotations...")
    results1 = load_annotated_results(file1)
    results2 = load_annotated_results(file2)

    if len(results1) != len(results2):
        print(f"⚠ Warning: Different number of results ({len(results1)} vs {len(results2)})")

    # Extract annotations
    scores1, critical1 = extract_annotations(results1)
    scores2, critical2 = extract_annotations(results2)

    # Calculate metrics
    print("Calculating agreement metrics...")
    metrics = calculate_agreement_metrics(scores1, scores2, critical1, critical2)

    # Identify disagreements
    print("Identifying disagreements...")
    disagreements = identify_disagreements(results1, results2, threshold=1)

    # Create output directory
    output_dir = Path("results/inter_annotator_agreement")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("Generating visualizations...")
    create_confusion_matrix(
        np.array(scores1), np.array(scores2),
        output_dir / 'confusion_matrix.png',
        f'Confusion Matrix: {judge1_name} vs {judge2_name}'
    )

    # Generate report
    print("Generating report...")
    generate_report(metrics, disagreements, judge1_name, judge2_name, output_dir)

    # Print summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Exact score agreement: {metrics['score_exact_agreement']*100:.1f}%")
    print(f"Score within 1 point: {metrics['score_within_1']*100:.1f}%")
    print(f"Cohen's Kappa (scores): {metrics['score_cohens_kappa']:.3f}")
    print()
    print(f"Critical issue agreement: {metrics['critical_agreement']*100:.1f}%")
    print(f"Cohen's Kappa (critical): {metrics['critical_cohens_kappa']:.3f}")
    print()
    print(f"Disagreements (score diff ≥ 1): {len(disagreements)}")
    print()
    print(f"Results saved to: {output_dir}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
