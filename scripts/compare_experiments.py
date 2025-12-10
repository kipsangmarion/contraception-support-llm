#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Experiments 1 and 2

Generate separate visualizations for:
- Experiment 1 results
- Experiment 2 results
- Experiment 1 vs Experiment 2 comparison

Usage:
    python scripts/compare_experiments.py

This will automatically find all exp1 and exp2 summary files and generate
separate visualizations in:
- results/comparisons/exp1/
- results/comparisons/exp2/
- results/comparisons/exp1_vs_exp2/
"""

import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def load_summaries(file_paths: List[Path]) -> List[Dict]:
    """Load all summary files."""
    summaries = []
    for fp in file_paths:
        with open(fp, 'r') as f:
            summaries.append(json.load(f))
    return summaries


def create_comparison_table(summaries: List[Dict]) -> pd.DataFrame:
    """Create comparison table."""
    data = []
    for s in summaries:
        data.append({
            'Model': s['model'],
            'Total Responses': s['total_responses'],
            'Fully Compliant (%)': f"{s['fully_compliant']/s['total_responses']*100:.1f}%",
            'Partially Compliant (%)': f"{s['partially_compliant']/s['total_responses']*100:.1f}%",
            'Non-Compliant (%)': f"{s['non_compliant']/s['total_responses']*100:.1f}%",
            'Critical Issues': s['critical_issues'],
            'Avg Score': f"{s['avg_compliance_score']:.2f}/2.0",
            'Avg Latency (s)': f"{s['avg_latency']:.2f}"
        })

    df = pd.DataFrame(data)
    return df


def plot_compliance_comparison(summaries: List[Dict], output_path: Path, title: str):
    """Create bar chart comparing compliance rates."""
    models = [s['model'] for s in summaries]
    fully = [s['fully_compliant']/s['total_responses']*100 for s in summaries]
    partially = [s['partially_compliant']/s['total_responses']*100 for s in summaries]
    non = [s['non_compliant']/s['total_responses']*100 for s in summaries]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(models))
    width = 0.25

    ax.bar([i-width for i in x], fully, width, label='Fully Compliant', color='#2ecc71')
    ax.bar(x, partially, width, label='Partially Compliant', color='#f39c12')
    ax.bar([i+width for i in x], non, width, label='Non-Compliant', color='#e74c3c')

    ax.set_xlabel('Model')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_category_heatmap(summaries: List[Dict], output_path: Path, title: str):
    """Create heatmap of compliance by category."""
    # Get all categories
    all_categories = set()
    for s in summaries:
        all_categories.update(s['by_category'].keys())

    categories = sorted(all_categories)
    models = [s['model'] for s in summaries]

    # Build matrix
    matrix = []
    for s in summaries:
        row = [s['by_category'].get(cat, 0) for cat in categories]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=2,
                xticklabels=categories, yticklabels=models, ax=ax,
                cbar_kws={'label': 'Avg Compliance Score'})

    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_critical_issues(summaries: List[Dict], output_path: Path, title: str):
    """Plot critical safety issues."""
    models = [s['model'] for s in summaries]
    critical = [s['critical_issues'] for s in summaries]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in critical]
    ax.bar(models, critical, color=colors)

    ax.set_xlabel('Model')
    ax.set_ylabel('Number of Critical Safety Issues')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(critical):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_exp1_vs_exp2_comparison(exp1_summaries: List[Dict], exp2_summaries: List[Dict],
                                   output_dir: Path):
    """Create side-by-side comparison plots for Exp 1 vs Exp 2."""

    # Match models between experiments
    exp1_by_model = {s['model']: s for s in exp1_summaries}
    exp2_by_model = {s['model']: s for s in exp2_summaries}

    common_models = set(exp1_by_model.keys()) & set(exp2_by_model.keys())

    if not common_models:
        print("⚠ No common models found between Experiment 1 and 2")
        return

    # Sort models for consistent ordering
    models = sorted(common_models)

    # 1. Compliance Score Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    exp1_scores = [exp1_by_model[m]['avg_compliance_score'] for m in models]
    exp2_scores = [exp2_by_model[m]['avg_compliance_score'] for m in models]

    x = range(len(models))
    width = 0.35

    ax.bar([i - width/2 for i in x], exp1_scores, width, label='Experiment 1 (Baseline)',
           color='#3498db', alpha=0.8)
    ax.bar([i + width/2 for i in x], exp2_scores, width, label='Experiment 2 (Prompted)',
           color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Average Compliance Score (0-2)')
    ax.set_title('Compliance Score: Experiment 1 vs Experiment 2')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 2.2])

    # Add value labels
    for i, (v1, v2) in enumerate(zip(exp1_scores, exp2_scores)):
        ax.text(i - width/2, v1 + 0.05, f'{v1:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, v2 + 0.05, f'{v2:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'score_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'score_comparison.png'}")
    plt.close()

    # 2. Critical Issues Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    exp1_critical = [exp1_by_model[m]['critical_issues'] for m in models]
    exp2_critical = [exp2_by_model[m]['critical_issues'] for m in models]

    ax.bar([i - width/2 for i in x], exp1_critical, width, label='Experiment 1 (Baseline)',
           color='#e74c3c', alpha=0.8)
    ax.bar([i + width/2 for i in x], exp2_critical, width, label='Experiment 2 (Prompted)',
           color='#f39c12', alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Number of Critical Safety Issues')
    ax.set_title('Critical Safety Issues: Experiment 1 vs Experiment 2')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (v1, v2) in enumerate(zip(exp1_critical, exp2_critical)):
        ax.text(i - width/2, v1 + 0.3, str(v1), ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(i + width/2, v2 + 0.3, str(v2), ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'critical_issues_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'critical_issues_comparison.png'}")
    plt.close()

    # 3. Improvement Delta Chart
    fig, ax = plt.subplots(figsize=(10, 6))

    score_delta = [exp2_scores[i] - exp1_scores[i] for i in range(len(models))]
    colors = ['#2ecc71' if d > 0 else '#e74c3c' if d < 0 else '#95a5a6' for d in score_delta]

    ax.bar(models, score_delta, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Model')
    ax.set_ylabel('Compliance Score Change (Exp2 - Exp1)')
    ax.set_title('Impact of Compliance-Aware Prompting')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, v in enumerate(score_delta):
        sign = '+' if v > 0 else ''
        ax.text(i, v + 0.02 if v >= 0 else v - 0.02, f'{sign}{v:.2f}',
                ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_delta.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'improvement_delta.png'}")
    plt.close()

    # 4. Summary statistics table
    summary_data = []
    for model in models:
        exp1 = exp1_by_model[model]
        exp2 = exp2_by_model[model]

        summary_data.append({
            'Model': model,
            'Exp1 Score': f"{exp1['avg_compliance_score']:.2f}",
            'Exp2 Score': f"{exp2['avg_compliance_score']:.2f}",
            'Change': f"{exp2['avg_compliance_score'] - exp1['avg_compliance_score']:+.2f}",
            'Exp1 Critical': exp1['critical_issues'],
            'Exp2 Critical': exp2['critical_issues'],
            'Critical Change': exp2['critical_issues'] - exp1['critical_issues'],
        })

    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / 'exp1_vs_exp2_summary.csv', index=False)
    print(f"✓ Saved: {output_dir / 'exp1_vs_exp2_summary.csv'}")
    print("\nSummary:\n" + df.to_string(index=False))


def main():
    print("="*80)
    print("Comparing Experiments 1 and 2")
    print("="*80)
    print()

    # Find all summary files
    results_dir = Path("results/compliance_experiments")

    exp1_files = sorted(results_dir.glob("exp1_*_auto_summary.json"))
    exp2_files = sorted(results_dir.glob("exp2_*_auto_summary.json"))

    print(f"Found {len(exp1_files)} Experiment 1 summaries")
    print(f"Found {len(exp2_files)} Experiment 2 summaries")
    print()

    if not exp1_files:
        print("✗ No Experiment 1 summary files found")
        return 1

    if not exp2_files:
        print("✗ No Experiment 2 summary files found")
        return 1

    # Load summaries
    exp1_summaries = load_summaries(exp1_files)
    exp2_summaries = load_summaries(exp2_files)

    # Create output directories
    base_dir = Path("results/comparisons")
    exp1_dir = base_dir / "exp1"
    exp2_dir = base_dir / "exp2"
    comparison_dir = base_dir / "exp1_vs_exp2"

    for dir_path in [exp1_dir, exp2_dir, comparison_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Generate Experiment 1 visualizations
    print("="*80)
    print("Generating Experiment 1 Visualizations")
    print("="*80)
    print()

    df1 = create_comparison_table(exp1_summaries)
    df1.to_csv(exp1_dir / "model_comparison.csv", index=False)
    print(f"✓ Saved: {exp1_dir / 'model_comparison.csv'}")

    plot_compliance_comparison(exp1_summaries, exp1_dir / "compliance_comparison.png",
                              "Experiment 1: Baseline Compliance Rates")
    plot_category_heatmap(exp1_summaries, exp1_dir / "category_heatmap.png",
                         "Experiment 1: Compliance by Category")
    plot_critical_issues(exp1_summaries, exp1_dir / "critical_issues.png",
                        "Experiment 1: Critical Safety Issues")

    # Generate Experiment 2 visualizations
    print()
    print("="*80)
    print("Generating Experiment 2 Visualizations")
    print("="*80)
    print()

    df2 = create_comparison_table(exp2_summaries)
    df2.to_csv(exp2_dir / "model_comparison.csv", index=False)
    print(f"✓ Saved: {exp2_dir / 'model_comparison.csv'}")

    plot_compliance_comparison(exp2_summaries, exp2_dir / "compliance_comparison.png",
                              "Experiment 2: Compliance-Aware Prompting")
    plot_category_heatmap(exp2_summaries, exp2_dir / "category_heatmap.png",
                         "Experiment 2: Compliance by Category")
    plot_critical_issues(exp2_summaries, exp2_dir / "critical_issues.png",
                        "Experiment 2: Critical Safety Issues")

    # Generate comparison visualizations
    print()
    print("="*80)
    print("Generating Experiment 1 vs 2 Comparisons")
    print("="*80)
    print()

    plot_exp1_vs_exp2_comparison(exp1_summaries, exp2_summaries, comparison_dir)

    print()
    print("="*80)
    print("✓ All visualizations generated successfully!")
    print("="*80)
    print()
    print("Output directories:")
    print(f"  - Experiment 1: {exp1_dir}")
    print(f"  - Experiment 2: {exp2_dir}")
    print(f"  - Comparison: {comparison_dir}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
