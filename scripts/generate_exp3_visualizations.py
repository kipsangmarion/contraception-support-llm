#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Experiment 3 (RAG) Visualizations

Creates visualizations in results/comparisons/exp3/ folder similar to exp1 and exp2.

Usage:
    python scripts/generate_exp3_visualizations.py
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def calculate_summary_from_annotated(annotated_file):
    """Calculate summary statistics from annotated file."""
    with open(annotated_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    responses = data.get('responses', [])

    # Extract metadata
    experiment_metadata = data.get('experiment_metadata', {})
    model = experiment_metadata.get('model', data.get('model', 'unknown'))

    # Count compliance scores (check both direct and nested annotation)
    scores = []
    critical_issues_count = 0
    latencies = []
    by_category = {}

    for r in responses:
        # Try annotation nested field first, then direct field
        annotation = r.get('annotation', {})
        score = annotation.get('compliance_score', r.get('compliance_score', 0))
        scores.append(score)

        # Check for critical issues
        has_critical = annotation.get('has_critical_safety_issue', r.get('critical_safety_issue', False))
        if has_critical:
            critical_issues_count += 1

        # Collect latency if available
        if 'latency' in r:
            latencies.append(r['latency'])

        # Collect by category
        category = r.get('category', 'Unknown')
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(score)

    # Calculate category averages
    category_averages = {cat: sum(vals)/len(vals) if vals else 0
                        for cat, vals in by_category.items()}

    total = len(responses)
    fully_compliant = sum(1 for s in scores if s == 2)
    partially_compliant = sum(1 for s in scores if s == 1)
    non_compliant = sum(1 for s in scores if s == 0)

    avg_score = sum(scores) / total if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return {
        'model': model,
        'total_responses': total,
        'fully_compliant': fully_compliant,
        'partially_compliant': partially_compliant,
        'non_compliant': non_compliant,
        'critical_issues': critical_issues_count,
        'avg_compliance_score': avg_score,
        'avg_latency': avg_latency,
        'by_category': category_averages
    }


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


def main():
    print("="*80)
    print("Generating Experiment 3 (RAG) Visualizations")
    print("="*80)
    print()

    # Find all Exp3 annotated files
    results_dir = Path("results/compliance_experiments")
    exp3_files = sorted(results_dir.glob("exp3_*_auto_annotated.json"))

    print(f"Found {len(exp3_files)} Experiment 3 annotated files")
    print()

    if not exp3_files:
        print("✗ No Experiment 3 annotated files found")
        print("  Expected files matching: exp3_*_auto_annotated.json")
        return 1

    # Load and calculate summaries
    exp3_summaries = []
    for file_path in exp3_files:
        print(f"Processing: {file_path.name}")
        summary = calculate_summary_from_annotated(file_path)
        exp3_summaries.append(summary)

    print()

    # Create output directory
    output_dir = Path("results/comparisons/exp3")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Generating Visualizations")
    print("="*80)
    print()

    # Generate comparison table
    df = create_comparison_table(exp3_summaries)
    df.to_csv(output_dir / "model_comparison.csv", index=False)
    print(f"✓ Saved: {output_dir / 'model_comparison.csv'}")

    # Generate plots
    plot_compliance_comparison(exp3_summaries, output_dir / "compliance_comparison.png",
                              "Experiment 3: RAG-Enhanced Compliance Rates")
    plot_category_heatmap(exp3_summaries, output_dir / "category_heatmap.png",
                         "Experiment 3: Compliance by Category")
    plot_critical_issues(exp3_summaries, output_dir / "critical_issues.png",
                        "Experiment 3: Critical Safety Issues")

    print()
    print("="*80)
    print("✓ All visualizations generated successfully!")
    print("="*80)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Summary Statistics:")
    print(df.to_string(index=False))
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
