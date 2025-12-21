#!/usr/bin/env python3
"""
Error Analysis by Model Across Experiments

Analyzes each model's performance across all experiments to identify:
1. Which models consistently perform well/poorly
2. Persistent error patterns for each model
3. Whether certain models are more prone to specific error types
4. How each model responds to different experimental conditions

Usage:
    python scripts/analyze_errors_by_model.py --output-dir results/error_analysis_by_model
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class ModelErrorAnalyzer:
    """Analyze errors by model across experiments."""

    def __init__(self, output_dir: str = "results/error_analysis_by_model"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track model performance across experiments
        self.model_data = defaultdict(lambda: {
            'experiments': {},  # {exp_num: {'errors': [], 'total_responses': 0}}
            'total_errors': 0,
            'total_responses': 0,
            'persistent_error_types': Counter(),
            'persistent_test_cases': Counter()
        })

        self.test_cases = {}

    def load_test_cases(self, test_set_path: Path):
        """Load ground truth test cases."""
        with open(test_set_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for case in data.get('test_cases', []):
            self.test_cases[case['id']] = case

    def load_experiment_results(self, file_path: Path) -> Dict:
        """Load annotated experiment results."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract experiment metadata
        if 'experiment_metadata' in data:
            exp_num = data['experiment_metadata']['experiment_number']
            model_name = data['experiment_metadata']['model']
            responses = data['responses']
        else:
            filename = file_path.stem
            if 'exp1' in filename:
                exp_num = 1
            elif 'exp2' in filename:
                exp_num = 2
            elif 'exp3' in filename:
                exp_num = 3
            elif 'exp4' in filename:
                exp_num = 4
            else:
                exp_num = 0

            parts = filename.split('_')
            if len(parts) >= 2:
                model_name = parts[1]
            else:
                model_name = "unknown"

            responses = data.get('responses', data.get('results', []))

        return {
            'experiment': exp_num,
            'model': model_name,
            'responses': responses,
            'file_path': str(file_path)
        }

    def categorize_error_type(self, response: Dict) -> List[str]:
        """Categorize error types."""
        error_types = []

        annotation = response.get('annotation', {})
        compliance_score = annotation.get('compliance_score', 2)
        has_critical = annotation.get('has_critical_safety_issue', False)
        criteria_violated = annotation.get('criteria_violated', [])

        test_case_id = response.get('test_case_id', '')
        test_case = self.test_cases.get(test_case_id, {})

        if has_critical:
            error_types.append('CRITICAL_SAFETY_VIOLATION')

        if test_case.get('category') == 'Medical Eligibility Criteria':
            mec_category = test_case.get('mec_category')
            if mec_category == 4 and compliance_score < 2:
                error_types.append('MEC_CAT4_FAILURE')

        must_include = test_case.get('compliant_response_criteria', {}).get('must_include', [])
        criteria_met = annotation.get('criteria_met', [])
        if len(must_include) > 0 and len(criteria_met) < len(must_include) * 0.5:
            error_types.append('MISSING_REQUIRED_INFO')

        if len(criteria_violated) > 0:
            error_types.append('PROHIBITED_INFO_INCLUDED')

        if test_case.get('category') == 'Rwanda Context' and compliance_score < 2:
            error_types.append('RWANDA_CONTEXT_MISSING')

        if test_case.get('category') == 'Myth Correction' and compliance_score == 0:
            error_types.append('MYTH_REINFORCEMENT')

        if test_case.get('category') == 'Timing Guidelines' and compliance_score < 2:
            error_types.append('TIMING_GUIDELINE_ERROR')

        if test_case.get('category') == 'Effectiveness Accuracy' and compliance_score < 2:
            error_types.append('EFFECTIVENESS_MISSTATEMENT')

        if compliance_score < 2 and not error_types:
            error_types.append('GENERIC_NON_COMPLIANCE')

        return error_types if error_types else ['NO_ERROR']

    def process_model_data(self):
        """Process all loaded data to analyze by model."""
        for model_name, model_info in self.model_data.items():
            for exp_num, exp_data in model_info['experiments'].items():
                errors = exp_data['errors']

                # Count error types
                for error in errors:
                    for error_type in error['error_types']:
                        model_info['persistent_error_types'][error_type] += 1

                    # Track which test cases this model fails
                    model_info['persistent_test_cases'][error['test_case_id']] += 1

    def identify_persistent_errors(self, model_name: str) -> Dict:
        """Identify errors that persist across multiple experiments for a model."""
        model_info = self.model_data[model_name]

        # Errors that appear in 2+ experiments
        persistent_error_types = {
            error_type: count
            for error_type, count in model_info['persistent_error_types'].items()
            if count >= 2 and error_type != 'NO_ERROR'
        }

        # Test cases that fail in 2+ experiments
        persistent_test_cases = {
            test_case_id: count
            for test_case_id, count in model_info['persistent_test_cases'].items()
            if count >= 2
        }

        # Experiments where model participated
        num_experiments = len(model_info['experiments'])

        return {
            'persistent_error_types': persistent_error_types,
            'persistent_test_cases': persistent_test_cases,
            'num_experiments': num_experiments,
            'total_errors': model_info['total_errors'],
            'total_responses': model_info['total_responses'],
            'error_rate': model_info['total_errors'] / model_info['total_responses'] if model_info['total_responses'] > 0 else 0
        }

    def compare_models_across_experiments(self) -> pd.DataFrame:
        """Generate comparison table of models across experiments."""
        comparison_data = []

        for model_name in sorted(self.model_data.keys()):
            model_info = self.model_data[model_name]

            row = {
                'Model': model_name,
                'Experiments Tested': len(model_info['experiments']),
                'Total Responses': model_info['total_responses'],
                'Total Errors': model_info['total_errors'],
                'Overall Error Rate (%)': f"{(model_info['total_errors'] / model_info['total_responses'] * 100) if model_info['total_responses'] > 0 else 0:.1f}%"
            }

            # Add per-experiment errors
            for exp_num in [1, 2, 3, 4]:
                if exp_num in model_info['experiments']:
                    exp_data = model_info['experiments'][exp_num]
                    error_count = len(exp_data['errors'])
                    total = exp_data['total_responses']
                    row[f'Exp {exp_num} Errors'] = f"{error_count}/{total}"
                    row[f'Exp {exp_num} Rate'] = f"{(error_count/total*100) if total > 0 else 0:.1f}%"
                else:
                    row[f'Exp {exp_num} Errors'] = '-'
                    row[f'Exp {exp_num} Rate'] = '-'

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def visualize_model_comparison(self):
        """Create visualizations comparing models."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Subplot 1: Overall error count by model
        ax = axes[0, 0]
        models = sorted(self.model_data.keys())
        error_counts = [self.model_data[m]['total_errors'] for m in models]

        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = ax.barh(models, error_counts, color=colors)
        ax.set_xlabel('Total Errors Across All Experiments', fontsize=12)
        ax.set_title('Total Error Count by Model', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{int(width)}',
                   ha='left', va='center', fontsize=10, fontweight='bold')

        # Subplot 2: Error rate by model
        ax = axes[0, 1]
        error_rates = [
            (self.model_data[m]['total_errors'] / self.model_data[m]['total_responses'] * 100)
            if self.model_data[m]['total_responses'] > 0 else 0
            for m in models
        ]

        bars = ax.barh(models, error_rates, color=colors)
        ax.set_xlabel('Error Rate (%)', fontsize=12)
        ax.set_title('Overall Error Rate by Model', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.1f}%',
                   ha='left', va='center', fontsize=10, fontweight='bold')

        # Subplot 3: Model performance across experiments (heatmap)
        ax = axes[1, 0]

        # Build matrix: models x experiments
        matrix_data = []
        for model in models:
            row = []
            for exp_num in [1, 2, 3, 4]:
                if exp_num in self.model_data[model]['experiments']:
                    exp_data = self.model_data[model]['experiments'][exp_num]
                    error_rate = (len(exp_data['errors']) / exp_data['total_responses'] * 100) if exp_data['total_responses'] > 0 else 0
                    row.append(error_rate)
                else:
                    row.append(np.nan)
            matrix_data.append(row)

        matrix = np.array(matrix_data)
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=['Exp 1', 'Exp 2', 'Exp 3', 'Exp 4'],
                   yticklabels=[m.replace('-', '-\n') for m in models],
                   cbar_kws={'label': 'Error Rate (%)'},
                   ax=ax, mask=np.isnan(matrix))
        ax.set_title('Error Rate by Model and Experiment', fontsize=14, fontweight='bold')

        # Subplot 4: Persistent error types by model
        ax = axes[1, 1]

        # Get top 5 persistent error types across all models
        all_persistent_errors = Counter()
        for model_name in models:
            persistent = self.identify_persistent_errors(model_name)
            all_persistent_errors.update(persistent['persistent_error_types'])

        top_error_types = [et for et, _ in all_persistent_errors.most_common(5)]

        # Build data for grouped bar chart
        x = np.arange(len(models))
        width = 0.15

        for i, error_type in enumerate(top_error_types):
            counts = []
            for model in models:
                persistent = self.identify_persistent_errors(model)
                counts.append(persistent['persistent_error_types'].get(error_type, 0))

            offset = width * (i - 2)
            ax.bar(x + offset, counts, width, label=error_type.replace('_', ' ').title()[:20])

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Occurrences Across Experiments', fontsize=12)
        ax.set_title('Persistent Error Types by Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('-')[0] for m in models], rotation=45, ha='right')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'model_comparison_overview.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()

    def generate_detailed_report(self):
        """Generate detailed report analyzing each model."""
        report_path = self.output_dir / 'model_by_model_analysis.txt'

        models = sorted(self.model_data.keys())

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("ERROR ANALYSIS BY MODEL ACROSS EXPERIMENTS\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Purpose: Identify persistent errors for each model across experiments\n\n")

            # Overall comparison
            f.write("-" * 100 + "\n")
            f.write("MODEL COMPARISON SUMMARY\n")
            f.write("-" * 100 + "\n\n")

            comparison_df = self.compare_models_across_experiments()
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")

            # Rank models
            model_rankings = [
                (m, self.model_data[m]['total_errors'])
                for m in models
            ]
            model_rankings.sort(key=lambda x: x[1])

            f.write("RANKING (Best to Worst):\n")
            for rank, (model, errors) in enumerate(model_rankings, 1):
                total_responses = self.model_data[model]['total_responses']
                error_rate = (errors / total_responses * 100) if total_responses > 0 else 0
                f.write(f"  {rank}. {model}: {errors} errors ({error_rate:.1f}% error rate)\n")
            f.write("\n")

            # Detailed analysis for each model
            for model_name in models:
                f.write("\n" + "=" * 100 + "\n")
                f.write(f"MODEL: {model_name.upper()}\n")
                f.write("=" * 100 + "\n\n")

                model_info = self.model_data[model_name]
                persistent = self.identify_persistent_errors(model_name)

                f.write(f"Experiments Tested: {persistent['num_experiments']}\n")
                f.write(f"Total Responses: {persistent['total_responses']}\n")
                f.write(f"Total Errors: {persistent['total_errors']}\n")
                f.write(f"Overall Error Rate: {persistent['error_rate']*100:.1f}%\n\n")

                # Performance by experiment
                f.write("-" * 100 + "\n")
                f.write("PERFORMANCE BY EXPERIMENT\n")
                f.write("-" * 100 + "\n\n")

                for exp_num in sorted(model_info['experiments'].keys()):
                    exp_data = model_info['experiments'][exp_num]
                    errors = exp_data['errors']
                    total = exp_data['total_responses']
                    error_rate = (len(errors) / total * 100) if total > 0 else 0

                    f.write(f"Experiment {exp_num}:\n")
                    f.write(f"  Responses: {total}\n")
                    f.write(f"  Errors: {len(errors)}\n")
                    f.write(f"  Error Rate: {error_rate:.1f}%\n")

                    # Error types in this experiment
                    error_types = [et for e in errors for et in e['error_types']]
                    error_type_counts = Counter(error_types)

                    f.write(f"  Top Error Types:\n")
                    for error_type, count in error_type_counts.most_common(3):
                        f.write(f"    - {error_type}: {count}\n")
                    f.write("\n")

                # Persistent errors
                f.write("-" * 100 + "\n")
                f.write("PERSISTENT ERROR PATTERNS\n")
                f.write("-" * 100 + "\n\n")

                f.write("Error Types That Persist Across Experiments:\n")
                if persistent['persistent_error_types']:
                    for error_type, count in sorted(persistent['persistent_error_types'].items(),
                                                   key=lambda x: x[1], reverse=True):
                        f.write(f"  {error_type}: {count} occurrences across {persistent['num_experiments']} experiments\n")
                else:
                    f.write("  None (model only tested in 1 experiment or no persistent patterns)\n")

                f.write("\n")

                f.write("Test Cases That Consistently Fail:\n")
                if persistent['persistent_test_cases']:
                    sorted_cases = sorted(persistent['persistent_test_cases'].items(),
                                        key=lambda x: x[1], reverse=True)
                    for test_case_id, count in sorted_cases[:10]:  # Top 10
                        test_case = self.test_cases.get(test_case_id, {})
                        category = test_case.get('category', 'unknown')
                        scenario = test_case.get('scenario', '')[:80]
                        f.write(f"  {test_case_id} ({category}): Failed in {count} experiments\n")
                        f.write(f"    Scenario: {scenario}...\n")
                else:
                    f.write("  None (model only tested in 1 experiment)\n")

                f.write("\n")

                # Model-specific insights
                f.write("-" * 100 + "\n")
                f.write("MODEL-SPECIFIC INSIGHTS\n")
                f.write("-" * 100 + "\n\n")

                self._write_model_insights(f, model_name, model_info, persistent)

            # Cross-model patterns
            f.write("\n" + "=" * 100 + "\n")
            f.write("CROSS-MODEL PATTERNS\n")
            f.write("=" * 100 + "\n\n")

            self._write_cross_model_patterns(f, models)

        print(f"[OK] Saved: {report_path}")

        # Save comparison table as CSV
        csv_path = self.output_dir / 'model_comparison.csv'
        comparison_df.to_csv(csv_path, index=False)
        print(f"[OK] Saved: {csv_path}")

    def _write_model_insights(self, f, model_name: str, model_info: Dict, persistent: Dict):
        """Write model-specific insights."""

        # Check for consistent strengths/weaknesses
        if 'claude' in model_name.lower():
            f.write("Claude Opus 4.5 Analysis:\n")
            f.write("- Expected Strength: Medical reasoning, instruction following\n")
            f.write("- Expected Weakness: May be overly cautious (false positives)\n\n")

            if persistent['error_rate'] > 0.5:
                f.write("  CONCERN: Error rate above 50% suggests challenges with compliance tasks\n")
            if 'PROHIBITED_INFO_INCLUDED' in persistent['persistent_error_types']:
                f.write("  PATTERN: Struggles with negative constraints despite strong instruction following\n")

        elif 'o3' in model_name.lower():
            f.write("OpenAI o3 Analysis:\n")
            f.write("- Expected Strength: Complex reasoning, structured thinking\n")
            f.write("- Expected Weakness: May over-reason and include unnecessary information\n\n")

            if persistent['error_rate'] < 0.4:
                f.write("  STRENGTH: Strong compliance performance, validates reasoning approach\n")

        elif 'grok' in model_name.lower():
            f.write("Grok 4.1 Fast Reasoning Analysis:\n")
            f.write("- Expected Strength: Speed, efficiency\n")
            f.write("- Expected Weakness: May sacrifice accuracy for speed\n\n")

            if persistent['error_rate'] > 0.5:
                f.write("  CONCERN: Fast reasoning may compromise compliance accuracy\n")

        elif 'gemini' in model_name.lower():
            f.write("Google Gemini Analysis:\n")
            f.write("- Expected Strength: Multimodal capabilities, Google Search integration\n")
            f.write("- Expected Weakness: May rely on web knowledge vs WHO guidelines\n\n")

        # Experiment-specific performance
        if len(model_info['experiments']) > 1:
            exp_error_rates = {}
            for exp_num, exp_data in model_info['experiments'].items():
                errors = len(exp_data['errors'])
                total = exp_data['total_responses']
                exp_error_rates[exp_num] = (errors / total) if total > 0 else 0

            best_exp = min(exp_error_rates.items(), key=lambda x: x[1])
            worst_exp = max(exp_error_rates.items(), key=lambda x: x[1])

            f.write(f"Performance Variation:\n")
            f.write(f"  Best in: Experiment {best_exp[0]} ({best_exp[1]*100:.1f}% error rate)\n")
            f.write(f"  Worst in: Experiment {worst_exp[0]} ({worst_exp[1]*100:.1f}% error rate)\n")

            if worst_exp[0] == 3:
                f.write("  PATTERN: RAG (Exp 3) degrades this model's performance\n")
            elif best_exp[0] == 4:
                f.write("  PATTERN: Safety validation (Exp 4) improves this model's performance\n")

            f.write("\n")

    def _write_cross_model_patterns(self, f, models: List[str]):
        """Write patterns that appear across multiple models."""

        # Find errors that ALL models make
        all_persistent_test_cases = defaultdict(list)
        for model_name in models:
            persistent = self.identify_persistent_errors(model_name)
            for test_case_id in persistent['persistent_test_cases']:
                all_persistent_test_cases[test_case_id].append(model_name)

        universal_failures = {
            tc: models_list
            for tc, models_list in all_persistent_test_cases.items()
            if len(models_list) >= len(models) * 0.75  # 75% or more models fail
        }

        f.write("Test Cases Where Most/All Models Fail:\n\n")
        if universal_failures:
            for test_case_id, failing_models in sorted(universal_failures.items(),
                                                       key=lambda x: len(x[1]), reverse=True):
                test_case = self.test_cases.get(test_case_id, {})
                category = test_case.get('category', 'unknown')
                scenario = test_case.get('scenario', '')[:100]

                f.write(f"{test_case_id} ({category}):\n")
                f.write(f"  Scenario: {scenario}...\n")
                f.write(f"  Models that fail: {len(failing_models)}/{len(models)}\n")
                f.write(f"  Failed by: {', '.join(failing_models)}\n\n")

            f.write("INSIGHT: These test cases represent universal challenges independent of model architecture.\n")
            f.write("         Likely require structural improvements (better prompts, safety validation, etc.)\n\n")
        else:
            f.write("None found (models have different failure patterns)\n\n")

        # Find error types that ALL models exhibit
        all_persistent_error_types = defaultdict(list)
        for model_name in models:
            persistent = self.identify_persistent_errors(model_name)
            for error_type in persistent['persistent_error_types']:
                all_persistent_error_types[error_type].append(model_name)

        universal_error_types = {
            et: models_list
            for et, models_list in all_persistent_error_types.items()
            if len(models_list) >= len(models) * 0.75
        }

        f.write("Error Types Common Across Models:\n\n")
        if universal_error_types:
            for error_type, models_with_error in sorted(universal_error_types.items(),
                                                        key=lambda x: len(x[1]), reverse=True):
                f.write(f"{error_type}:\n")
                f.write(f"  Appears in: {len(models_with_error)}/{len(models)} models\n")
                f.write(f"  Models: {', '.join(models_with_error)}\n\n")

            f.write("INSIGHT: These error types are model-agnostic. Prompt engineering or architectural\n")
            f.write("         changes (like safety validation) needed to address them.\n\n")
        else:
            f.write("Models have unique error patterns\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze errors by model across experiments"
    )
    parser.add_argument(
        '--results-dir',
        default='results/compliance_experiments',
        help='Directory containing annotated experiment results'
    )
    parser.add_argument(
        '--test-set',
        default='data/compliance_test_set.json',
        help='Path to compliance test set JSON'
    )
    parser.add_argument(
        '--output-dir',
        default='results/error_analysis_by_model',
        help='Output directory for analysis results'
    )

    args = parser.parse_args()

    print("=" * 100)
    print("ERROR ANALYSIS BY MODEL ACROSS EXPERIMENTS")
    print("=" * 100)
    print()

    analyzer = ModelErrorAnalyzer(output_dir=args.output_dir)

    # Load test cases
    print(f"Loading test cases from: {args.test_set}")
    analyzer.load_test_cases(Path(args.test_set))
    print(f"[OK] Loaded {len(analyzer.test_cases)} test cases\n")

    # Load experiment results
    results_dir = Path(args.results_dir)
    annotated_files = list(results_dir.glob('*_auto_annotated.json'))

    print(f"Loading experiment results from: {args.results_dir}")
    print(f"Found {len(annotated_files)} annotated result files\n")

    for file_path in annotated_files:
        exp_data = analyzer.load_experiment_results(file_path)
        model_name = exp_data['model']
        exp_num = exp_data['experiment']

        # Initialize experiment data for this model
        if exp_num not in analyzer.model_data[model_name]['experiments']:
            analyzer.model_data[model_name]['experiments'][exp_num] = {
                'errors': [],
                'total_responses': 0
            }

        # Process responses
        for response in exp_data['responses']:
            analyzer.model_data[model_name]['experiments'][exp_num]['total_responses'] += 1
            analyzer.model_data[model_name]['total_responses'] += 1

            annotation = response.get('annotation', {})
            compliance_score = annotation.get('compliance_score', 2)

            if compliance_score < 2:
                error_types = analyzer.categorize_error_type(response)

                error_entry = {
                    'experiment': exp_num,
                    'test_case_id': response.get('test_case_id', ''),
                    'category': response.get('category', ''),
                    'compliance_score': compliance_score,
                    'has_critical_safety_issue': annotation.get('has_critical_safety_issue', False),
                    'error_types': error_types,
                    'scenario': response.get('scenario', '')
                }

                analyzer.model_data[model_name]['experiments'][exp_num]['errors'].append(error_entry)
                analyzer.model_data[model_name]['total_errors'] += 1

        print(f"[OK] Loaded {model_name} - Exp {exp_num}: {exp_data['responses'].__len__()} responses")

    print()

    # Process model data to identify persistent errors
    print("Analyzing persistent error patterns...")
    analyzer.process_model_data()
    print("[OK] Analysis complete\n")

    # Generate comparison visualization
    print("Creating model comparison visualizations...")
    analyzer.visualize_model_comparison()
    print("[OK] Visualizations created\n")

    # Generate detailed report
    print("Generating detailed model-by-model report...")
    analyzer.generate_detailed_report()
    print("[OK] Report generated\n")

    # Summary
    models = sorted(analyzer.model_data.keys())
    model_rankings = [
        (m, analyzer.model_data[m]['total_errors'], analyzer.model_data[m]['total_responses'])
        for m in models
    ]
    model_rankings.sort(key=lambda x: x[1])

    print("=" * 100)
    print("SUMMARY - MODEL RANKINGS")
    print("=" * 100)
    for rank, (model, errors, total) in enumerate(model_rankings, 1):
        error_rate = (errors / total * 100) if total > 0 else 0
        print(f"{rank}. {model}")
        print(f"   Errors: {errors}/{total} ({error_rate:.1f}%)")

        # Check for persistent errors
        persistent = analyzer.identify_persistent_errors(model)
        if persistent['persistent_error_types']:
            top_persistent = list(persistent['persistent_error_types'].items())[0]
            print(f"   Most Persistent Error: {top_persistent[0]} ({top_persistent[1]} times)")
        print()

    print(f"Results saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
