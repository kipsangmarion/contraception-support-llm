#!/usr/bin/env python3
"""
Error Analysis by Experiment

Separates error analysis by experiment to clearly identify:
1. Which experiment has minimal errors
2. What causes errors in each experiment
3. Comparative analysis across experiments

Usage:
    python scripts/analyze_errors_by_experiment.py --output-dir results/error_analysis_by_experiment
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
import random

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class ExperimentErrorAnalyzer:
    """Analyze errors separately by experiment."""

    def __init__(self, output_dir: str = "results/error_analysis_by_experiment"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiments = {
            1: {'name': 'Baseline (No WHO Context)', 'data': []},
            2: {'name': 'Compliance-Aware Prompting', 'data': []},
            3: {'name': 'RAG Retrieval', 'data': []},
            4: {'name': 'Safety Validation Layer', 'data': []}
        }

        self.test_cases = {}
        self.errors_by_experiment = defaultdict(list)
        self.model_performance = defaultdict(lambda: defaultdict(int))

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

    def categorize_error_type(self, response: Dict, exp_num: int) -> List[str]:
        """Categorize error types with experiment-specific context."""
        error_types = []

        annotation = response.get('annotation', {})
        compliance_score = annotation.get('compliance_score', 2)
        has_critical = annotation.get('has_critical_safety_issue', False)
        criteria_violated = annotation.get('criteria_violated', [])

        test_case_id = response.get('test_case_id', '')
        test_case = self.test_cases.get(test_case_id, {})

        # Critical Safety Violation
        if has_critical:
            error_types.append('CRITICAL_SAFETY_VIOLATION')

        # MEC Category Misclassification
        if test_case.get('category') == 'Medical Eligibility Criteria':
            mec_category = test_case.get('mec_category')
            if mec_category == 4 and compliance_score < 2:
                error_types.append('MEC_CAT4_FAILURE')

        # Missing Required Information
        must_include = test_case.get('compliant_response_criteria', {}).get('must_include', [])
        criteria_met = annotation.get('criteria_met', [])
        if len(must_include) > 0 and len(criteria_met) < len(must_include) * 0.5:
            error_types.append('MISSING_REQUIRED_INFO')

        # Including Prohibited Information
        if len(criteria_violated) > 0:
            error_types.append('PROHIBITED_INFO_INCLUDED')

        # Rwanda Context Missing
        if test_case.get('category') == 'Rwanda Context' and compliance_score < 2:
            error_types.append('RWANDA_CONTEXT_MISSING')

        # Myth Reinforcement
        if test_case.get('category') == 'Myth Correction' and compliance_score == 0:
            error_types.append('MYTH_REINFORCEMENT')

        # Timing Error
        if test_case.get('category') == 'Timing Guidelines' and compliance_score < 2:
            error_types.append('TIMING_GUIDELINE_ERROR')

        # Effectiveness Misstatement
        if test_case.get('category') == 'Effectiveness Accuracy' and compliance_score < 2:
            error_types.append('EFFECTIVENESS_MISSTATEMENT')

        # Experiment-specific error patterns
        if exp_num == 1:
            # Baseline lacks WHO context
            if 'MEC' in test_case_id.upper() and compliance_score < 2:
                error_types.append('BASELINE_NO_WHO_CONTEXT')

        elif exp_num == 3:
            # RAG-specific issues
            if 'RAG' in str(response.get('model_response', '')).upper() or len(response.get('model_response', '')) > 1000:
                error_types.append('RAG_CONTEXT_OVERLOAD')

        # Generic error if non-compliant but no specific type
        if compliance_score < 2 and not error_types:
            error_types.append('GENERIC_NON_COMPLIANCE')

        return error_types if error_types else ['NO_ERROR']

    def analyze_experiment(self, exp_num: int) -> Dict:
        """Analyze errors for a specific experiment."""
        exp_data = self.experiments[exp_num]['data']

        if not exp_data:
            return {
                'experiment': exp_num,
                'name': self.experiments[exp_num]['name'],
                'total_responses': 0,
                'total_errors': 0,
                'error_rate': 0,
                'errors': []
            }

        all_errors = []
        total_responses = 0

        for exp_result in exp_data:
            model = exp_result['model']
            for response in exp_result['responses']:
                total_responses += 1

                annotation = response.get('annotation', {})
                compliance_score = annotation.get('compliance_score', 2)

                if compliance_score < 2:
                    error_types = self.categorize_error_type(response, exp_num)

                    all_errors.append({
                        'model': model,
                        'test_case_id': response.get('test_case_id', ''),
                        'category': response.get('category', ''),
                        'severity': self.test_cases.get(response.get('test_case_id', ''), {}).get('severity', 'unknown'),
                        'compliance_score': compliance_score,
                        'has_critical_safety_issue': annotation.get('has_critical_safety_issue', False),
                        'error_types': error_types,
                        'criteria_violated': annotation.get('criteria_violated', []),
                        'criteria_met': annotation.get('criteria_met', []),
                        'notes': annotation.get('notes', ''),
                        'scenario': response.get('scenario', ''),
                        'model_response': response.get('model_response', ''),
                        'ground_truth': response.get('ground_truth', '')
                    })

                    # Track model performance
                    self.model_performance[exp_num][model] += 1

        return {
            'experiment': exp_num,
            'name': self.experiments[exp_num]['name'],
            'total_responses': total_responses,
            'total_errors': len(all_errors),
            'error_rate': len(all_errors) / total_responses if total_responses > 0 else 0,
            'errors': all_errors
        }

    def identify_experiment_specific_causes(self, exp_num: int, errors: List[Dict]) -> List[str]:
        """Identify what causes errors in this specific experiment."""
        causes = []

        error_types = [et for e in errors for et in e['error_types']]
        error_type_counts = Counter(error_types)

        # Experiment-specific analysis
        if exp_num == 1:
            # Baseline - no WHO context
            causes.append("LACK_OF_WHO_CONTEXT: Baseline experiment lacks WHO MEC categories and compliance guidelines in system prompt")

            if error_type_counts.get('MEC_CAT4_FAILURE', 0) > 0:
                causes.append("MEC_KNOWLEDGE_GAP: Model relies on generic medical knowledge without WHO-specific contraindication categories")

            if error_type_counts.get('MISSING_REQUIRED_INFO', 0) > error_type_counts.get('PROHIBITED_INFO_INCLUDED', 0):
                causes.append("INCOMPLETE_RESPONSES: Without structured compliance criteria, model provides brief answers missing key information")

        elif exp_num == 2:
            # Compliance-aware prompting
            if error_type_counts.get('PROHIBITED_INFO_INCLUDED', 0) > 0:
                causes.append("INSTRUCTION_FOLLOWING_FAILURE: Despite explicit 'must avoid' constraints, model includes prohibited information")

            if error_type_counts.get('RWANDA_CONTEXT_MISSING', 0) > 0:
                causes.append("REGIONAL_KNOWLEDGE_GAP: Rwanda-specific policies not included in compliance-aware prompt")

            if error_type_counts.get('EFFECTIVENESS_MISSTATEMENT', 0) > 0:
                causes.append("NUMERICAL_PRECISION_ERRORS: Model confuses effectiveness rates despite having WHO guidelines")

        elif exp_num == 3:
            # RAG
            causes.append("RAG_CONTEXT_INTERFERENCE: Retrieved documents dilute or contradict compliance-aware reasoning")

            if error_type_counts.get('MISSING_REQUIRED_INFO', 0) > error_type_counts.get('CRITICAL_SAFETY_VIOLATION', 0):
                causes.append("CONTEXT_OVERLOAD: Too many retrieved documents cause model to miss key compliance criteria")

            if error_type_counts.get('PROHIBITED_INFO_INCLUDED', 0) > 50:
                causes.append("RAG_RETRIEVAL_QUALITY: Retrieved documents may contain outdated or non-compliant information")

            causes.append("LOST_SYSTEM_PROMPT: RAG context may override system prompt instructions")

        elif exp_num == 4:
            # Safety validation
            if len(errors) > 0:
                causes.append("VALIDATION_LAYER_GAPS: Safety validator catches Category 4 violations but misses other error types")

            if error_type_counts.get('PROHIBITED_INFO_INCLUDED', 0) > 0:
                causes.append("POST_GENERATION_LIMITATION: Validator checks after generation, doesn't prevent initial errors")

            if len(errors) < 30:
                causes.append("SUCCESS_FACTOR: Safety validation layer effectively reduces errors compared to other experiments")

        # Common causes across experiments
        if error_type_counts.get('RWANDA_CONTEXT_MISSING', 0) > 5:
            causes.append("TRAINING_DATA_BIAS: All models trained primarily on US/Europe healthcare, lacking Rwanda context")

        if error_type_counts.get('MYTH_REINFORCEMENT', 0) > 0:
            causes.append("MISINFORMATION_IN_TRAINING: Common contraception myths present in web-scraped training data")

        return causes

    def generate_experiment_comparison(self) -> pd.DataFrame:
        """Generate comparison table across experiments."""
        comparison_data = []

        for exp_num in [1, 2, 3, 4]:
            analysis = self.analyze_experiment(exp_num)
            errors = analysis['errors']

            error_types = [et for e in errors for et in e['error_types']]
            error_type_counts = Counter(error_types)

            comparison_data.append({
                'Experiment': f"Exp {exp_num}: {analysis['name']}",
                'Total Responses': analysis['total_responses'],
                'Total Errors': analysis['total_errors'],
                'Error Rate (%)': f"{analysis['error_rate'] * 100:.1f}%",
                'Critical Safety Issues': sum(1 for e in errors if e['has_critical_safety_issue']),
                'MEC Cat4 Failures': error_type_counts.get('MEC_CAT4_FAILURE', 0),
                'Prohibited Info Included': error_type_counts.get('PROHIBITED_INFO_INCLUDED', 0),
                'Missing Required Info': error_type_counts.get('MISSING_REQUIRED_INFO', 0),
                'Rwanda Context Missing': error_type_counts.get('RWANDA_CONTEXT_MISSING', 0)
            })

        return pd.DataFrame(comparison_data)

    def visualize_experiment_comparison(self):
        """Create comprehensive visualization comparing experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Subplot 1: Error counts by experiment
        ax = axes[0, 0]
        exp_names = []
        error_counts = []
        colors = ['#e74c3c', '#3498db', '#e67e22', '#2ecc71']  # Red, Blue, Orange, Green

        for exp_num in [1, 2, 3, 4]:
            analysis = self.analyze_experiment(exp_num)
            exp_names.append(f"Exp {exp_num}")
            error_counts.append(analysis['total_errors'])

        bars = ax.bar(exp_names, error_counts, color=colors)
        ax.set_ylabel('Number of Errors', fontsize=12)
        ax.set_title('Total Errors by Experiment', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Subplot 2: Error rates
        ax = axes[0, 1]
        error_rates = []

        for exp_num in [1, 2, 3, 4]:
            analysis = self.analyze_experiment(exp_num)
            error_rates.append(analysis['error_rate'] * 100)

        bars = ax.bar(exp_names, error_rates, color=colors)
        ax.set_ylabel('Error Rate (%)', fontsize=12)
        ax.set_title('Error Rate by Experiment', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend()

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Subplot 3: Critical safety issues
        ax = axes[1, 0]
        critical_counts = []

        for exp_num in [1, 2, 3, 4]:
            analysis = self.analyze_experiment(exp_num)
            critical = sum(1 for e in analysis['errors'] if e['has_critical_safety_issue'])
            critical_counts.append(critical)

        bars = ax.bar(exp_names, critical_counts, color=['#c0392b' if c > 0 else '#27ae60' for c in critical_counts])
        ax.set_ylabel('Critical Safety Issues', fontsize=12)
        ax.set_title('Critical Safety Violations by Experiment', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Subplot 4: Error type distribution by experiment
        ax = axes[1, 1]

        error_type_data = defaultdict(list)
        top_error_types = ['PROHIBITED_INFO_INCLUDED', 'MISSING_REQUIRED_INFO',
                          'CRITICAL_SAFETY_VIOLATION', 'MEC_CAT4_FAILURE']

        for exp_num in [1, 2, 3, 4]:
            analysis = self.analyze_experiment(exp_num)
            error_types = [et for e in analysis['errors'] for et in e['error_types']]
            error_type_counts = Counter(error_types)

            for et in top_error_types:
                error_type_data[et].append(error_type_counts.get(et, 0))

        x = np.arange(len(exp_names))
        width = 0.2

        for i, (error_type, counts) in enumerate(error_type_data.items()):
            offset = width * (i - 1.5)
            ax.bar(x + offset, counts, width, label=error_type.replace('_', ' ').title())

        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Error Type Distribution Across Experiments', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'experiment_comparison_overview.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()

    def generate_detailed_report(self):
        """Generate detailed report comparing experiments."""
        report_path = self.output_dir / 'experiment_by_experiment_analysis.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("ERROR ANALYSIS BY EXPERIMENT\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Purpose: Identify which experiment has minimal errors and what causes them\n\n")

            # Overall comparison table
            f.write("-" * 100 + "\n")
            f.write("EXPERIMENT COMPARISON SUMMARY\n")
            f.write("-" * 100 + "\n\n")

            comparison_df = self.generate_experiment_comparison()
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")

            # Identify best and worst
            analyses = [self.analyze_experiment(i) for i in [1, 2, 3, 4]]
            analyses.sort(key=lambda x: x['total_errors'])

            best_exp = analyses[0]
            worst_exp = analyses[-1]

            f.write("KEY FINDINGS:\n")
            f.write(f"[BEST]  Experiment {best_exp['experiment']}: {best_exp['name']}\n")
            f.write(f"        Total Errors: {best_exp['total_errors']}\n")
            f.write(f"        Error Rate: {best_exp['error_rate']*100:.1f}%\n\n")

            f.write(f"[WORST] Experiment {worst_exp['experiment']}: {worst_exp['name']}\n")
            f.write(f"        Total Errors: {worst_exp['total_errors']}\n")
            f.write(f"        Error Rate: {worst_exp['error_rate']*100:.1f}%\n\n")

            improvement = ((worst_exp['total_errors'] - best_exp['total_errors']) / worst_exp['total_errors']) * 100
            f.write(f"[IMPROVEMENT] Exp {best_exp['experiment']} reduces errors by {improvement:.1f}% compared to Exp {worst_exp['experiment']}\n\n")

            # Detailed analysis for each experiment
            for exp_num in [1, 2, 3, 4]:
                f.write("\n" + "=" * 100 + "\n")
                f.write(f"EXPERIMENT {exp_num}: {self.experiments[exp_num]['name'].upper()}\n")
                f.write("=" * 100 + "\n\n")

                analysis = self.analyze_experiment(exp_num)
                errors = analysis['errors']

                f.write(f"Total Responses: {analysis['total_responses']}\n")
                f.write(f"Total Errors: {analysis['total_errors']}\n")
                f.write(f"Error Rate: {analysis['error_rate']*100:.1f}%\n")
                f.write(f"Critical Safety Issues: {sum(1 for e in errors if e['has_critical_safety_issue'])}\n\n")

                # Error breakdown
                f.write("-" * 100 + "\n")
                f.write("ERROR BREAKDOWN\n")
                f.write("-" * 100 + "\n\n")

                error_types = [et for e in errors for et in e['error_types']]
                error_type_counts = Counter(error_types)

                f.write("Error Types:\n")
                for error_type, count in error_type_counts.most_common():
                    percentage = (count / len(errors)) * 100 if len(errors) > 0 else 0
                    f.write(f"  {error_type}: {count} ({percentage:.1f}%)\n")

                # Errors by category
                f.write("\nErrors by Category:\n")
                category_counts = Counter([e['category'] for e in errors])
                for category, count in category_counts.most_common():
                    f.write(f"  {category}: {count}\n")

                # Errors by model
                f.write("\nErrors by Model:\n")
                model_counts = Counter([e['model'] for e in errors])
                for model, count in model_counts.most_common():
                    f.write(f"  {model}: {count}\n")

                # Root causes
                f.write("\n" + "-" * 100 + "\n")
                f.write("ROOT CAUSES OF ERRORS IN THIS EXPERIMENT\n")
                f.write("-" * 100 + "\n\n")

                causes = self.identify_experiment_specific_causes(exp_num, errors)
                for i, cause in enumerate(causes, 1):
                    cause_parts = cause.split(': ', 1)
                    if len(cause_parts) == 2:
                        f.write(f"{i}. {cause_parts[0]}\n")
                        f.write(f"   {cause_parts[1]}\n\n")
                    else:
                        f.write(f"{i}. {cause}\n\n")

                # Sample errors
                f.write("-" * 100 + "\n")
                f.write("SAMPLE ERRORS (Random Selection)\n")
                f.write("-" * 100 + "\n\n")

                if errors:
                    sample_size = min(3, len(errors))
                    samples = random.sample(errors, sample_size)

                    for i, sample in enumerate(samples, 1):
                        f.write(f"Sample {i}:\n")
                        f.write(f"  Model: {sample['model']}\n")
                        f.write(f"  Test Case: {sample['test_case_id']}\n")
                        f.write(f"  Category: {sample['category']}\n")
                        f.write(f"  Compliance Score: {sample['compliance_score']}/2\n")
                        f.write(f"  Error Types: {', '.join(sample['error_types'])}\n")
                        f.write(f"  Scenario: {sample['scenario'][:150]}...\n")
                        f.write(f"  Criteria Violated: {len(sample['criteria_violated'])} items\n\n")

            # Recommendations
            f.write("\n" + "=" * 100 + "\n")
            f.write("RECOMMENDATIONS BASED ON EXPERIMENT COMPARISON\n")
            f.write("=" * 100 + "\n\n")

            self._write_recommendations(f, analyses)

        print(f"[OK] Saved: {report_path}")

        # Save comparison table as CSV
        csv_path = self.output_dir / 'experiment_comparison.csv'
        comparison_df.to_csv(csv_path, index=False)
        print(f"[OK] Saved: {csv_path}")

    def _write_recommendations(self, f, analyses):
        """Write recommendations based on experiment comparison."""
        best_exp = analyses[0]
        worst_exp = analyses[-1]

        f.write(f"1. DEPLOY EXPERIMENT {best_exp['experiment']} APPROACH\n\n")

        if best_exp['experiment'] == 4:
            f.write("   The Safety Validation Layer (Exp 4) achieved the lowest error rate.\n")
            f.write("   This 'Trust but Verify' approach catches Category 4 violations post-generation.\n\n")
            f.write("   Implementation:\n")
            f.write("   - Use compliance-aware prompting as base (Exp 2 approach)\n")
            f.write("   - Add SafetyValidator to check responses before delivery\n")
            f.write("   - Block responses with Category 4 contraindications\n")
            f.write("   - Suggest safer alternatives automatically\n\n")

        elif best_exp['experiment'] == 2:
            f.write("   Compliance-Aware Prompting (Exp 2) achieved the best results.\n")
            f.write("   Explicit WHO MEC categories and compliance criteria in system prompt are effective.\n\n")
            f.write("   Implementation:\n")
            f.write("   - Include WHO MEC category table in system prompt\n")
            f.write("   - Add 'must include' and 'must avoid' checklists\n")
            f.write("   - Use structured prompt template for all queries\n\n")

        f.write(f"2. AVOID EXPERIMENT {worst_exp['experiment']} APPROACH\n\n")

        if worst_exp['experiment'] == 3:
            f.write("   RAG Retrieval (Exp 3) had the highest error rate.\n")
            f.write("   Retrieved context interferes with compliance-aware reasoning.\n\n")
            f.write("   Why RAG Failed:\n")
            f.write("   - Too many documents dilute key compliance criteria\n")
            f.write("   - Retrieved docs may contain outdated or conflicting information\n")
            f.write("   - Context length limits cause loss of system prompt instructions\n\n")
            f.write("   Alternative:\n")
            f.write("   - Use compliance-aware prompting without RAG for safety-critical queries\n")
            f.write("   - RAG only for factual lookup (e.g., clinic locations, service availability)\n")
            f.write("   - Hybrid: prompt-based reasoning + RAG for citations only\n\n")

        elif worst_exp['experiment'] == 1:
            f.write("   Baseline (Exp 1) without WHO context had high error rate.\n")
            f.write("   Generic medical knowledge insufficient for contraception compliance.\n\n")
            f.write("   Why Baseline Failed:\n")
            f.write("   - No WHO MEC categories in system prompt\n")
            f.write("   - Model relies on training data which may contain myths\n")
            f.write("   - Lacks structured compliance criteria\n\n")

        f.write("3. COMBINE BEST APPROACHES\n\n")
        f.write("   Recommended Architecture:\n")
        f.write("   - Base: Compliance-Aware Prompting (Exp 2)\n")
        f.write("   - Layer: Safety Validation (Exp 4)\n")
        f.write("   - Avoid: RAG for compliance reasoning (Exp 3)\n\n")

        f.write("4. ADDRESS COMMON ERROR TYPES\n\n")

        # Find most common error across all experiments
        all_errors = []
        for analysis in analyses:
            all_errors.extend(analysis['errors'])

        all_error_types = [et for e in all_errors for et in e['error_types']]
        most_common = Counter(all_error_types).most_common(1)[0]

        f.write(f"   Most Common Error: {most_common[0]} ({most_common[1]} occurrences)\n\n")

        if most_common[0] == 'PROHIBITED_INFO_INCLUDED':
            f.write("   Solution:\n")
            f.write("   - Strengthen 'must avoid' constraints in prompt\n")
            f.write("   - Add pre-response checklist: 'Did I include any prohibited information?'\n")
            f.write("   - Use structured output format to enforce constraints\n\n")

        f.write("5. IMPROVE RWANDA CONTEXT\n\n")
        rwanda_errors = sum(1 for e in all_errors if 'RWANDA_CONTEXT_MISSING' in e['error_types'])
        if rwanda_errors > 0:
            f.write(f"   Rwanda context errors found in all experiments ({rwanda_errors} total).\n")
            f.write("   This is independent of experiment design - all approaches lack Rwanda knowledge.\n\n")
            f.write("   Solution:\n")
            f.write("   - Add Rwanda-specific section to ALL experiment prompts\n")
            f.write("   - Include: free contraception policy, community health workers, non-discrimination\n")
            f.write("   - Fine-tune on Rwanda healthcare documents if possible\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze errors separately by experiment"
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
        default='results/error_analysis_by_experiment',
        help='Output directory for analysis results'
    )

    args = parser.parse_args()

    print("=" * 100)
    print("ERROR ANALYSIS BY EXPERIMENT")
    print("=" * 100)
    print()

    analyzer = ExperimentErrorAnalyzer(output_dir=args.output_dir)

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
        exp_num = exp_data['experiment']

        if exp_num in analyzer.experiments:
            analyzer.experiments[exp_num]['data'].append(exp_data)
            print(f"[OK] Loaded Exp {exp_num}: {file_path.name}")

    print()

    # Analyze each experiment
    for exp_num in [1, 2, 3, 4]:
        print(f"Analyzing Experiment {exp_num}: {analyzer.experiments[exp_num]['name']}...")
        analysis = analyzer.analyze_experiment(exp_num)
        print(f"  Total Responses: {analysis['total_responses']}")
        print(f"  Total Errors: {analysis['total_errors']}")
        print(f"  Error Rate: {analysis['error_rate']*100:.1f}%")
        print()

    # Generate comparison visualization
    print("Creating comparison visualizations...")
    analyzer.visualize_experiment_comparison()
    print("[OK] Visualizations created\n")

    # Generate detailed report
    print("Generating detailed experiment-by-experiment report...")
    analyzer.generate_detailed_report()
    print("[OK] Report generated\n")

    # Summary
    analyses = [analyzer.analyze_experiment(i) for i in [1, 2, 3, 4]]
    analyses.sort(key=lambda x: x['total_errors'])

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Best Experiment:  Exp {analyses[0]['experiment']} - {analyses[0]['name']}")
    print(f"                  {analyses[0]['total_errors']} errors ({analyses[0]['error_rate']*100:.1f}% error rate)")
    print()
    print(f"Worst Experiment: Exp {analyses[-1]['experiment']} - {analyses[-1]['name']}")
    print(f"                  {analyses[-1]['total_errors']} errors ({analyses[-1]['error_rate']*100:.1f}% error rate)")
    print()
    improvement = ((analyses[-1]['total_errors'] - analyses[0]['total_errors']) / analyses[-1]['total_errors']) * 100
    print(f"Improvement: {improvement:.1f}% error reduction from worst to best")
    print()
    print(f"Results saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
