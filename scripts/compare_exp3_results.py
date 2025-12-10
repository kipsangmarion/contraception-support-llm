#!/usr/bin/env python3
"""
Compare Experiment 1, 2, and 3 Results

Generates comparison visualizations and summary statistics.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Force UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_summary(file_path):
    """Load summary JSON file and normalize format."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalize field names (handle both formats)
    total = data.get('total_responses', 0)
    fully = data.get('fully_compliant', 0)
    partially = data.get('partially_compliant', 0)
    non = data.get('non_compliant', 0)

    return {
        'total_responses': total,
        'fully_compliant': fully,
        'partially_compliant': partially,
        'non_compliant': non,
        'fully_compliant_pct': (fully / total * 100) if total > 0 else 0,
        'partially_compliant_pct': (partially / total * 100) if total > 0 else 0,
        'non_compliant_pct': (non / total * 100) if total > 0 else 0,
        'critical_issues': data.get('critical_issues', 0),
        'average_score': data.get('avg_compliance_score', data.get('average_score', 0))
    }


def calculate_summary_from_annotated(annotated_file):
    """Calculate summary statistics from annotated file."""
    with open(annotated_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    responses = data.get('responses', [])

    # Count compliance scores (check both direct and nested annotation)
    scores = []
    critical_issues_count = 0

    for r in responses:
        # Try annotation nested field first, then direct field
        annotation = r.get('annotation', {})
        score = annotation.get('compliance_score', r.get('compliance_score', 0))
        scores.append(score)

        # Check for critical issues
        has_critical = annotation.get('has_critical_safety_issue', r.get('critical_safety_issue', False))
        if has_critical:
            critical_issues_count += 1

    total = len(responses)
    fully_compliant = sum(1 for s in scores if s == 2)
    partially_compliant = sum(1 for s in scores if s == 1)
    non_compliant = sum(1 for s in scores if s == 0)

    avg_score = sum(scores) / total if total > 0 else 0

    return {
        'total_responses': total,
        'fully_compliant': fully_compliant,
        'partially_compliant': partially_compliant,
        'non_compliant': non_compliant,
        'fully_compliant_pct': (fully_compliant / total * 100) if total > 0 else 0,
        'partially_compliant_pct': (partially_compliant / total * 100) if total > 0 else 0,
        'non_compliant_pct': (non_compliant / total * 100) if total > 0 else 0,
        'critical_issues': critical_issues_count,
        'average_score': avg_score
    }


def main():
    results_dir = Path('results/compliance_experiments')

    # Define file paths (updated with actual Exp2 timestamps)
    models = {
        'o3': {
            'exp1': 'exp1_o3-2025-04-16_20251208_110608_auto_summary.json',
            'exp2': 'exp2_o3-2025-04-16_20251208_212508_auto_summary.json',
            'exp3': 'exp3_o3-2025-04-16_rag_20251209_004927_auto_annotated.json'
        },
        'claude': {
            'exp1': 'exp1_claude-opus-4-5-20251101_20251208_050610_auto_summary.json',
            'exp2': 'exp2_claude-opus-4-5-20251101_20251208_214528_auto_summary.json',
            'exp3': 'exp3_claude-opus-4-5-20251101_rag_20251209_010156_auto_annotated.json'
        },
        'grok': {
            'exp1': 'exp1_grok-4-1-fast-reasoning_20251208_053400_auto_summary.json',
            'exp2': 'exp2_grok-4-1-fast-reasoning_20251208_220605_auto_summary.json',
            'exp3': 'exp3_grok-4-1-fast-reasoning_rag_20251209_010840_auto_annotated.json'
        }
    }

    # Collect data
    comparison_data = []

    for model_name, files in models.items():
        for exp_num in ['exp1', 'exp2', 'exp3']:
            file_path = results_dir / files[exp_num]

            if not file_path.exists():
                print(f"⚠ Warning: {file_path} not found, skipping...")
                continue

            # Load or calculate summary
            if exp_num == 'exp3':
                # Calculate from annotated file
                summary = calculate_summary_from_annotated(file_path)
            else:
                # Load existing summary
                summary = load_summary(file_path)

            comparison_data.append({
                'Model': model_name,
                'Experiment': exp_num.upper(),
                'Avg Score': summary.get('average_score', 0),
                'Fully Compliant (%)': summary.get('fully_compliant_pct', 0),
                'Partially Compliant (%)': summary.get('partially_compliant_pct', 0),
                'Non-Compliant (%)': summary.get('non_compliant_pct', 0),
                'Critical Issues': summary.get('critical_issues', 0)
            })

    df = pd.DataFrame(comparison_data)

    # Create output directory
    output_dir = Path('results/comparisons/exp1_vs_exp2_vs_exp3')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_file = output_dir / 'model_comparison.csv'
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved comparison CSV to: {csv_file}")

    # Print summary table
    print("\n" + "="*80)
    print("EXPERIMENT 1 vs 2 vs 3 COMPARISON")
    print("="*80 + "\n")
    print(df.to_string(index=False))
    print()

    # Create visualizations
    sns.set_style("whitegrid")

    # 1. Average Score Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot(index='Model', columns='Experiment', values='Avg Score')
    pivot.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Average Compliance Score (0-2)')
    ax.set_title('Compliance Score: Exp1 (Baseline) vs Exp2 (Prompted) vs Exp3 (RAG)')
    ax.legend(title='Experiment')
    ax.set_ylim(0, 2)
    ax.axhline(y=1.85, color='gray', linestyle='--', alpha=0.5, label='Target (1.85)')
    plt.tight_layout()
    plt.savefig(output_dir / 'score_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved score comparison to: {output_dir / 'score_comparison.png'}")
    plt.close()

    # 2. Fully Compliant Percentage
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot(index='Model', columns='Experiment', values='Fully Compliant (%)')
    pivot.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Fully Compliant (%)')
    ax.set_title('Fully Compliant Responses: Exp1 vs Exp2 vs Exp3')
    ax.legend(title='Experiment')
    ax.set_ylim(0, 100)
    ax.axhline(y=85, color='gray', linestyle='--', alpha=0.5, label='Target (85%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'fully_compliant_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved compliance comparison to: {output_dir / 'fully_compliant_comparison.png'}")
    plt.close()

    # 3. Critical Issues
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot(index='Model', columns='Experiment', values='Critical Issues')
    pivot.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Critical Safety Issues (count)')
    ax.set_title('Critical Safety Issues: Exp1 vs Exp2 vs Exp3')
    ax.legend(title='Experiment')
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Target (0)')
    plt.tight_layout()
    plt.savefig(output_dir / 'critical_issues_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved critical issues comparison to: {output_dir / 'critical_issues_comparison.png'}")
    plt.close()

    # 4. Improvement Delta (Exp3 vs Exp1)
    exp1_data = df[df['Experiment'] == 'EXP1'].set_index('Model')
    exp3_data = df[df['Experiment'] == 'EXP3'].set_index('Model')

    delta_df = pd.DataFrame({
        'Model': exp1_data.index,
        'Score Delta': exp3_data['Avg Score'] - exp1_data['Avg Score'],
        'Compliance % Delta': exp3_data['Fully Compliant (%)'] - exp1_data['Fully Compliant (%)'],
        'Critical Issues Delta': exp3_data['Critical Issues'] - exp1_data['Critical Issues']
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(delta_df))
    width = 0.25

    ax.bar([i - width for i in x], delta_df['Score Delta'] * 50, width, label='Score Δ (×50)', color='#3498db')
    ax.bar(x, delta_df['Compliance % Delta'], width, label='Fully Compliant % Δ', color='#2ecc71')
    ax.bar([i + width for i in x], delta_df['Critical Issues Delta'], width, label='Critical Issues Δ', color='#e74c3c')

    ax.set_ylabel('Change (Exp3 - Exp1)')
    ax.set_title('RAG Impact: Improvement from Baseline (Exp1) to RAG (Exp3)')
    ax.set_xticks(x)
    ax.set_xticklabels(delta_df['Model'])
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'rag_improvement_delta.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved improvement delta to: {output_dir / 'rag_improvement_delta.png'}")
    plt.close()

    # Save delta CSV
    delta_csv = output_dir / 'rag_improvement_delta.csv'
    delta_df.to_csv(delta_csv, index=False)
    print(f"✓ Saved improvement delta CSV to: {delta_csv}")

    print("\n" + "="*80)
    print("RAG IMPROVEMENT SUMMARY (Exp3 vs Exp1)")
    print("="*80 + "\n")
    print(delta_df.to_string(index=False))
    print()

    print("\n✓ All comparisons complete!")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
