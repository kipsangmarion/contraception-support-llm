#!/usr/bin/env python3
"""
Unified Result Comparison Script

Consolidates functionality from:
- compare_experiments.py
- compare_models.py
- compute_statistical_significance.py
- visualize_experiments.py

Usage:
    python scripts/compare_results.py --mode statistical
    python scripts/compare_results.py --mode visualize
    python scripts/compare_results.py --mode comprehensive
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.comparator import ExperimentComparator
from src.analysis.visualizer import UnifiedVisualizer


def statistical_comparison(
    results_dir: Path,
    output_dir: Path,
    models: list
):
    """
    Perform statistical comparison between experiments.

    Tests:
    - McNemar's test (paired samples)
    - Chi-square test (independence)
    - Cohen's h (effect size)
    """
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)

    comparator = ExperimentComparator()

    # Load results
    print("\nLoading results...")
    comparator.load_results(results_dir)
    print(f"Loaded {len(comparator.results)} experiments")

    # Perform pairwise comparisons
    print("\nPerforming pairwise comparisons...")

    all_comparisons = []
    experiments = sorted(comparator.results.keys())

    for model in models:
        print(f"\n{model}:")
        print("-" * 80)

        for i, exp1 in enumerate(experiments):
            for exp2 in experiments[i+1:]:
                comparison = comparator.compare_experiments_pairwise(exp1, exp2, model)
                all_comparisons.append(comparison)

                # Print summary
                print(f"\nExp{exp1} vs Exp{exp2}:")

                # McNemar's test
                mcnemar = comparison['mcnemar']
                if mcnemar.get('p_value') is not None:
                    print(f"  McNemar's test: {mcnemar['interpretation']}")
                    print(f"    Statistic: {mcnemar['statistic']:.4f}")
                    print(f"    P-value: {mcnemar['p_value']:.4f}")
                else:
                    print(f"  McNemar's test: {mcnemar.get('note', 'Not applicable')}")

                # Chi-square test
                chi_square = comparison['chi_square']
                if chi_square.get('p_value') is not None:
                    print(f"  Chi-square test: {chi_square['interpretation']}")
                    print(f"    Statistic: {chi_square['statistic']:.4f}")
                    print(f"    P-value: {chi_square['p_value']:.4f}")

                # Cohen's h
                cohens = comparison['cohens_h']
                if cohens.get('effect_size') is not None:
                    print(f"  Cohen's h: {cohens['interpretation']}")
                    print(f"    Effect size: {cohens['effect_size']:.3f}")

    # Save to CSV
    csv_data = []
    for comp in all_comparisons:
        if comp['mcnemar'].get('p_value') is not None:
            csv_data.append({
                'Model': comp['model'],
                'Exp1': comp['exp1'],
                'Exp2': comp['exp2'],
                'McNemar_Statistic': comp['mcnemar']['statistic'],
                'McNemar_P_Value': comp['mcnemar']['p_value'],
                'McNemar_Significant': comp['mcnemar']['significant'],
                'ChiSquare_Statistic': comp['chi_square'].get('statistic'),
                'ChiSquare_P_Value': comp['chi_square'].get('p_value'),
                'ChiSquare_Significant': comp['chi_square'].get('significant'),
                'Cohens_h': comp['cohens_h'].get('effect_size'),
                'Effect_Magnitude': comp['cohens_h'].get('magnitude')
            })

    df = pd.DataFrame(csv_data)
    csv_path = output_dir / "statistical_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n\nSaved: {csv_path}")

    print("\n[COMPLETE] Statistical comparison complete")


def visualize_comparison(
    results_dir: Path,
    output_dir: Path,
    models: list
):
    """
    Generate comparison visualizations.

    Creates:
    - Compliance rate comparison
    - Critical issues comparison
    - Average score comparison
    - Model trajectories
    """
    print("\n" + "="*80)
    print("VISUALIZATION COMPARISON")
    print("="*80)

    comparator = ExperimentComparator()
    visualizer = UnifiedVisualizer(output_dir / "visualizations")

    # Load results
    print("\nLoading results...")
    comparator.load_results(results_dir)

    # Get summary statistics for each experiment
    print("\nComputing summary statistics...")
    experiment_stats = {}

    for exp_num in sorted(comparator.results.keys()):
        # Aggregate across all models
        all_stats = []
        for model in models:
            stats = comparator.get_summary_statistics(exp_num, model)
            if stats:
                all_stats.append(stats)

        if all_stats:
            # Average across models
            experiment_stats[exp_num] = {
                'fully_compliant_pct': sum(s['fully_compliant_pct'] for s in all_stats) / len(all_stats),
                'critical_issues': sum(s['critical_issues'] for s in all_stats),
                'avg_score': sum(s['avg_score'] for s in all_stats) / len(all_stats)
            }

    # Generate visualizations
    print("\nGenerating visualizations...")

    visualizer.plot_compliance_comparison(
        experiment_stats,
        title="Compliance Rate Comparison Across Experiments",
        output_name="compliance_comparison.png"
    )

    visualizer.plot_critical_issues_comparison(
        experiment_stats,
        title="Critical Safety Issues Comparison",
        output_name="critical_issues_comparison.png"
    )

    visualizer.plot_score_comparison(
        experiment_stats,
        title="Average Compliance Score Comparison",
        output_name="score_comparison.png"
    )

    # Model trajectories
    model_trajectories = {}
    for model in models:
        model_trajectories[model] = {
            'experiments': [],
            'error_rate': [],
            'error_counts': [],
            'critical_counts': []
        }

        for exp_num in sorted(comparator.results.keys()):
            stats = comparator.get_summary_statistics(exp_num, model)
            if stats:
                model_trajectories[model]['experiments'].append(exp_num)
                model_trajectories[model]['error_rate'].append(
                    100 - stats['fully_compliant_pct']
                )
                model_trajectories[model]['error_counts'].append(
                    stats['non_compliant']
                )
                model_trajectories[model]['critical_counts'].append(
                    stats['critical_issues']
                )

    visualizer.plot_model_trajectories(
        model_trajectories,
        metric='error_rate',
        title="Model Error Rates Across Experiments",
        output_name="model_error_trajectories.png"
    )

    # Generate comparison table
    print("\nGenerating comparison table...")
    experiments = sorted(comparator.results.keys())
    table = comparator.generate_comparison_table(experiments, models)

    table_path = output_dir / "comparison_table.csv"
    table.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")

    print("\n[COMPLETE] Visualization comparison complete")


def comprehensive_comparison(
    results_dir: Path,
    output_dir: Path,
    models: list
):
    """
    Run both statistical and visualization comparisons.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON")
    print("="*80)

    statistical_comparison(results_dir, output_dir, models)
    visualize_comparison(results_dir, output_dir, models)

    print("\n" + "="*80)
    print("ALL COMPARISONS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Unified result comparison")
    parser.add_argument(
        '--mode',
        choices=['statistical', 'visualize', 'comprehensive'],
        default='comprehensive',
        help='Comparison mode'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/compliance_experiments',
        help='Directory containing annotated results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/unified_comparison',
        help='Output directory'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['claude-opus-4-5-20251101', 'o3-2025-04-16', 'grok-4-1-fast-reasoning'],
        help='Models to compare'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison
    if args.mode == 'statistical':
        statistical_comparison(results_dir, output_dir, args.models)
    elif args.mode == 'visualize':
        visualize_comparison(results_dir, output_dir, args.models)
    elif args.mode == 'comprehensive':
        comprehensive_comparison(results_dir, output_dir, args.models)


if __name__ == "__main__":
    main()
