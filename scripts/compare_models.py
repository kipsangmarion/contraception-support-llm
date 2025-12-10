#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Compliance Results Across Models

Create comparison tables and visualizations.

Usage:
    python scripts/compare_models.py <summary1.json> <summary2.json> [summary3.json ...]

Example:
    python scripts/compare_models.py \
        results/compliance_experiments/exp1_claude_summary.json \
        results/compliance_experiments/exp1_grok_summary.json \
        results/compliance_experiments/exp1_o3_summary.json
"""

import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def load_summaries(file_paths):
    """Load all summary files."""
    summaries = []
    for fp in file_paths:
        with open(fp, 'r') as f:
            summaries.append(json.load(f))
    return summaries


def create_comparison_table(summaries):
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


def plot_compliance_comparison(summaries, output_dir):
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
    ax.set_title('Compliance Rates by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/compliance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/compliance_comparison.png")
    plt.close()


def plot_category_heatmap(summaries, output_dir):
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
                xticklabels=categories, yticklabels=models, ax=ax, cbar_kws={'label': 'Avg Compliance Score'})

    ax.set_title('Compliance by Category (Heatmap)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/category_heatmap.png")
    plt.close()


def plot_critical_issues(summaries, output_dir):
    """Plot critical safety issues."""

    models = [s['model'] for s in summaries]
    critical = [s['critical_issues'] for s in summaries]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in critical]
    ax.bar(models, critical, color=colors)

    ax.set_xlabel('Model')
    ax.set_ylabel('Number of Critical Safety Issues')
    ax.set_title('Critical Safety Issues by Model')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(critical):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/critical_issues.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/critical_issues.png")
    plt.close()


def plot_latency_vs_compliance(summaries, output_dir):
    """Scatter plot of latency vs compliance score."""

    models = [s['model'] for s in summaries]
    latencies = [s['avg_latency'] for s in summaries]
    scores = [s['avg_compliance_score'] for s in summaries]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(latencies, scores, s=200, alpha=0.6)

    # Label each point
    for i, model in enumerate(models):
        ax.annotate(model, (latencies[i], scores[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Average Latency (seconds)')
    ax.set_ylabel('Average Compliance Score (0-2)')
    ax.set_title('Latency vs Compliance Trade-off')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 2.2])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/latency_vs_compliance.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/latency_vs_compliance.png")
    plt.close()


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/compare_models.py <summary1.json> <summary2.json> [summary3.json ...]")
        print("\nExample:")
        print("  python scripts/compare_models.py \\")
        print("      results/compliance_experiments/exp1_claude_summary.json \\")
        print("      results/compliance_experiments/exp1_grok_summary.json \\")
        print("      results/compliance_experiments/exp1_o3_summary.json")
        sys.exit(1)

    summary_files = sys.argv[1:]

    # Load summaries
    print(f"Loading {len(summary_files)} summary files...")
    summaries = load_summaries(summary_files)

    # Create output directory
    output_dir = Path("results/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("Creating Model Comparisons")
    print(f"{'='*80}\n")

    # Comparison table
    print("Creating comparison table...")
    df = create_comparison_table(summaries)
    print("\n" + df.to_string(index=False))

    # Save to CSV
    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")

    # Create visualizations
    print("\nCreating visualizations...")
    plot_compliance_comparison(summaries, output_dir)
    plot_category_heatmap(summaries, output_dir)
    plot_critical_issues(summaries, output_dir)
    plot_latency_vs_compliance(summaries, output_dir)

    print(f"\n{'='*80}")
    print("✓ Model comparison complete!")
    print(f"All files saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
