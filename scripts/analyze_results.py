#!/usr/bin/env python3
"""
Unified Result Analysis Script

Consolidates functionality from:
- analyze_error_patterns.py
- analyze_errors_by_experiment.py
- analyze_errors_by_model.py

Usage:
    python scripts/analyze_results.py --mode comprehensive
    python scripts/analyze_results.py --mode error-patterns
    python scripts/analyze_results.py --mode by-experiment
    python scripts/analyze_results.py --mode by-model
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.visualizer import UnifiedVisualizer


class UnifiedAnalyzer(BaseAnalyzer):
    """Unified analyzer combining all analysis modes."""

    def __init__(self, output_dir: str = "results/analysis"):
        super().__init__(output_dir)
        self.visualizer = UnifiedVisualizer(self.output_dir / "visualizations")

    def analyze_error_patterns(self, results_dir: Path, test_set_path: Path):
        """
        Analyze error patterns across models and experiments.

        Replaces: analyze_error_patterns.py
        """
        print("\n" + "="*80)
        print("ERROR PATTERN ANALYSIS")
        print("="*80)

        # Load data
        print("\nLoading data...")
        self.load_test_cases(test_set_path)
        self.experiments = self.load_all_experiments(results_dir)
        print(f"Loaded {len(self.experiments)} experiment results")

        # Extract all errors
        print("\nExtracting errors...")
        all_errors = []
        for (exp_num, model_name), exp_data in self.experiments.items():
            errors = self.extract_errors_from_experiment(
                exp_num, model_name, exp_data['responses']
            )
            all_errors.extend(errors)

        print(f"Found {len(all_errors)} total errors")

        # Compute statistics
        stats = self.compute_summary_stats(all_errors)

        # Print summary
        print("\n" + "-"*80)
        print("SUMMARY STATISTICS")
        print("-"*80)
        print(f"Total Errors: {stats['total_errors']}")
        print(f"Critical Errors: {stats['critical_errors']}")
        print(f"\nError Type Distribution:")
        for error_type, count in sorted(stats['error_type_distribution'].items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")

        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualizer.plot_error_distribution(
            all_errors,
            group_by='error_type',
            title="Error Type Distribution",
            output_name="error_type_distribution.png"
        )

        self.visualizer.plot_error_distribution(
            all_errors,
            group_by='model',
            title="Error Distribution by Model",
            output_name="error_by_model.png"
        )

        self.visualizer.plot_category_breakdown(
            all_errors,
            title="Errors by Category",
            output_name="error_by_category.png"
        )

        # Save detailed CSV
        self._save_error_csv(all_errors, "error_patterns_detailed.csv")

        # Generate markdown report
        self._generate_error_pattern_report(all_errors, stats)

        print("\n[COMPLETE] Error pattern analysis complete")

    def analyze_by_experiment(self, results_dir: Path, test_set_path: Path):
        """
        Analyze errors separately by experiment.

        Replaces: analyze_errors_by_experiment.py
        """
        print("\n" + "="*80)
        print("ANALYSIS BY EXPERIMENT")
        print("="*80)

        # Load data
        print("\nLoading data...")
        self.load_test_cases(test_set_path)
        self.experiments = self.load_all_experiments(results_dir)

        # Group by experiment
        experiment_errors = defaultdict(list)
        experiment_stats = {}

        for (exp_num, model_name), exp_data in self.experiments.items():
            errors = self.extract_errors_from_experiment(
                exp_num, model_name, exp_data['responses']
            )
            experiment_errors[exp_num].extend(errors)

        # Analyze each experiment
        print("\nAnalyzing experiments...")
        for exp_num in sorted(experiment_errors.keys()):
            errors = experiment_errors[exp_num]
            total_responses = sum(
                len(exp_data['responses'])
                for (e, _), exp_data in self.experiments.items()
                if e == exp_num
            )

            print(f"\nExperiment {exp_num}:")
            print(f"  Total Responses: {total_responses}")
            print(f"  Total Errors: {len(errors)}")
            print(f"  Error Rate: {len(errors) / total_responses * 100:.1f}%")

            critical = sum(1 for e in errors if e['has_critical_safety_issue'])
            print(f"  Critical Errors: {critical}")

            experiment_stats[exp_num] = {
                'total_responses': total_responses,
                'total_errors': len(errors),
                'error_rate': len(errors) / total_responses * 100,
                'critical_rate': critical / total_responses * 100
            }

        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualizer.plot_experiment_comparison(
            experiment_stats,
            title="Experiment Comparison",
            output_name="experiment_comparison.png"
        )

        for exp_num, errors in experiment_errors.items():
            self.visualizer.plot_error_distribution(
                errors,
                group_by='error_type',
                title=f"Exp{exp_num} Error Types",
                output_name=f"exp{exp_num}_error_types.png"
            )

        # Save CSV
        all_errors = []
        for errors_list in experiment_errors.values():
            all_errors.extend(errors_list)
        self._save_error_csv(all_errors, "errors_by_experiment.csv")

        # Generate report
        self._generate_experiment_report(experiment_errors, experiment_stats)

        print("\n[COMPLETE] Experiment analysis complete")

    def analyze_by_model(self, results_dir: Path, test_set_path: Path):
        """
        Analyze errors by model across experiments.

        Replaces: analyze_errors_by_model.py
        """
        print("\n" + "="*80)
        print("ANALYSIS BY MODEL")
        print("="*80)

        # Load data
        print("\nLoading data...")
        self.load_test_cases(test_set_path)
        self.experiments = self.load_all_experiments(results_dir)

        # Group by model
        model_errors = defaultdict(list)
        model_trajectories = defaultdict(lambda: {
            'experiments': [],
            'error_counts': [],
            'error_rates': [],
            'critical_counts': []
        })

        for (exp_num, model_name), exp_data in self.experiments.items():
            errors = self.extract_errors_from_experiment(
                exp_num, model_name, exp_data['responses']
            )
            model_errors[model_name].extend(errors)

            # Track trajectory
            total_responses = len(exp_data['responses'])
            critical = sum(1 for e in errors if e['has_critical_safety_issue'])

            model_trajectories[model_name]['experiments'].append(exp_num)
            model_trajectories[model_name]['error_counts'].append(len(errors))
            model_trajectories[model_name]['error_rates'].append(
                len(errors) / total_responses * 100
            )
            model_trajectories[model_name]['critical_counts'].append(critical)

        # Print summary
        print("\nModel Summary:")
        for model_name, errors in sorted(model_errors.items()):
            print(f"\n{model_name}:")
            print(f"  Total Errors: {len(errors)}")
            critical = sum(1 for e in errors if e['has_critical_safety_issue'])
            print(f"  Critical Errors: {critical}")

        # Generate visualizations
        print("\nGenerating visualizations...")

        # Model trajectories
        self.visualizer.plot_model_trajectories(
            model_trajectories,
            metric='error_rate',
            title="Model Error Rates Across Experiments",
            output_name="model_error_trajectories.png"
        )

        self.visualizer.plot_model_trajectories(
            model_trajectories,
            metric='critical_counts',
            title="Model Critical Errors Across Experiments",
            output_name="model_critical_trajectories.png"
        )

        # Error distribution by model
        all_errors = []
        for errors_list in model_errors.values():
            all_errors.extend(errors_list)

        self.visualizer.plot_error_distribution(
            all_errors,
            group_by='model',
            title="Error Distribution by Model",
            output_name="model_error_distribution.png"
        )

        # Save CSV
        self._save_error_csv(all_errors, "errors_by_model.csv")

        # Generate report
        self._generate_model_report(model_errors, model_trajectories)

        print("\n[COMPLETE] Model analysis complete")

    def analyze_comprehensive(self, results_dir: Path, test_set_path: Path):
        """
        Run all analysis modes.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS")
        print("="*80)

        self.analyze_error_patterns(results_dir, test_set_path)
        self.analyze_by_experiment(results_dir, test_set_path)
        self.analyze_by_model(results_dir, test_set_path)

        print("\n" + "="*80)
        print("ALL ANALYSES COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Visualizations saved to: {self.visualizer.output_dir}")

    def _save_error_csv(self, errors: List[Dict], filename: str):
        """Save errors to CSV file."""
        if not errors:
            return

        df = pd.DataFrame(errors)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved: {output_path}")

    def _generate_error_pattern_report(self, errors: List[Dict], stats: Dict):
        """Generate error pattern markdown report."""
        sections = [
            {
                'heading': 'Overview',
                'content': f"""
Total Errors: {stats['total_errors']}
Critical Errors: {stats['critical_errors']}
Critical Rate: {stats['critical_errors'] / max(stats['total_errors'], 1) * 100:.1f}%
"""
            },
            {
                'heading': 'Error Type Distribution',
                'content': '\n'.join(
                    f"- {error_type}: {count}"
                    for error_type, count in sorted(
                        stats['error_type_distribution'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                )
            },
            {
                'heading': 'Category Distribution',
                'content': '\n'.join(
                    f"- {category}: {count}"
                    for category, count in sorted(
                        stats['category_distribution'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                )
            }
        ]

        self.generate_markdown_report(
            "Error Pattern Analysis",
            sections,
            self.output_dir / "error_pattern_report.md"
        )

    def _generate_experiment_report(self, experiment_errors: Dict, experiment_stats: Dict):
        """Generate experiment comparison report."""
        sections = []

        for exp_num in sorted(experiment_errors.keys()):
            stats = experiment_stats[exp_num]
            errors = experiment_errors[exp_num]

            error_types = Counter()
            for error in errors:
                for error_type in error['error_types']:
                    error_types[error_type] += 1

            content = f"""
Total Responses: {stats['total_responses']}
Total Errors: {stats['total_errors']}
Error Rate: {stats['error_rate']:.1f}%
Critical Rate: {stats['critical_rate']:.1f}%

Top Error Types:
""" + '\n'.join(f"- {et}: {count}" for et, count in error_types.most_common(5))

            sections.append({
                'heading': f'Experiment {exp_num}',
                'content': content
            })

        self.generate_markdown_report(
            "Experiment Comparison Report",
            sections,
            self.output_dir / "experiment_report.md"
        )

    def _generate_model_report(self, model_errors: Dict, model_trajectories: Dict):
        """Generate model comparison report."""
        sections = []

        for model_name in sorted(model_errors.keys()):
            errors = model_errors[model_name]
            trajectory = model_trajectories[model_name]

            error_types = Counter()
            for error in errors:
                for error_type in error['error_types']:
                    error_types[error_type] += 1

            content = f"""
Total Errors: {len(errors)}
Critical Errors: {sum(1 for e in errors if e['has_critical_safety_issue'])}

Experiments Participated: {sorted(trajectory['experiments'])}
Error Rate Range: {min(trajectory['error_rates']):.1f}% - {max(trajectory['error_rates']):.1f}%

Top Error Types:
""" + '\n'.join(f"- {et}: {count}" for et, count in error_types.most_common(5))

            sections.append({
                'heading': model_name,
                'content': content
            })

        self.generate_markdown_report(
            "Model Comparison Report",
            sections,
            self.output_dir / "model_report.md"
        )


def main():
    parser = argparse.ArgumentParser(description="Unified result analysis")
    parser.add_argument(
        '--mode',
        choices=['comprehensive', 'error-patterns', 'by-experiment', 'by-model'],
        default='comprehensive',
        help='Analysis mode'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/compliance_experiments',
        help='Directory containing annotated results'
    )
    parser.add_argument(
        '--test-set',
        type=str,
        default='data/compliance_test_set.json',
        help='Path to test set JSON'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/unified_analysis',
        help='Output directory'
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = UnifiedAnalyzer(output_dir=args.output_dir)

    results_dir = Path(args.results_dir)
    test_set_path = Path(args.test_set)

    # Run analysis
    if args.mode == 'comprehensive':
        analyzer.analyze_comprehensive(results_dir, test_set_path)
    elif args.mode == 'error-patterns':
        analyzer.analyze_error_patterns(results_dir, test_set_path)
    elif args.mode == 'by-experiment':
        analyzer.analyze_by_experiment(results_dir, test_set_path)
    elif args.mode == 'by-model':
        analyzer.analyze_by_model(results_dir, test_set_path)


if __name__ == "__main__":
    main()
