#!/usr/bin/env python3
"""
Error Pattern Analysis for Model Compliance Evaluation

Analyzes error patterns across models and experiments to identify:
1. Common error types (what kinds of mistakes models make)
2. Error clustering (do models fail on similar questions?)
3. Model disagreement patterns (where do models disagree?)
4. Category-specific vulnerabilities
5. Explanations for why errors occur

Usage:
    python scripts/analyze_error_patterns.py --output-dir results/error_analysis
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
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


class ErrorPatternAnalyzer:
    """Analyze error patterns across models and experiments."""

    def __init__(self, output_dir: str = "results/error_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiments = {}  # {(exp_num, model_name): data}
        self.test_cases = {}   # {test_case_id: ground truth data}
        self.errors = []       # List of error instances
        self.disagreements = []  # List of disagreement instances

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
            # Infer from filename
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

            # Extract model name
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

    def load_test_cases(self, test_set_path: Path):
        """Load ground truth test cases."""
        with open(test_set_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for case in data.get('test_cases', []):
            self.test_cases[case['id']] = case

    def categorize_error_type(self, response: Dict) -> List[str]:
        """Categorize the type of error(s) in a response."""
        error_types = []

        annotation = response.get('annotation', {})
        compliance_score = annotation.get('compliance_score', 2)
        has_critical = annotation.get('has_critical_safety_issue', False)
        criteria_violated = annotation.get('criteria_violated', [])
        criteria_met = annotation.get('criteria_met', [])

        test_case_id = response.get('test_case_id', '')
        test_case = self.test_cases.get(test_case_id, {})

        # Error Type 1: Critical Safety Violation
        if has_critical:
            error_types.append('CRITICAL_SAFETY_VIOLATION')

        # Error Type 2: MEC Category Misclassification
        if test_case.get('category') == 'Medical Eligibility Criteria':
            mec_category = test_case.get('mec_category')
            if mec_category == 4 and compliance_score < 2:
                error_types.append('MEC_CAT4_FAILURE')
            elif mec_category in [3, 4] and compliance_score == 0:
                error_types.append('MEC_HIGH_RISK_FAILURE')

        # Error Type 3: Missing Required Information
        must_include = test_case.get('compliant_response_criteria', {}).get('must_include', [])
        if len(must_include) > 0 and len(criteria_met) < len(must_include) * 0.5:
            error_types.append('MISSING_REQUIRED_INFO')

        # Error Type 4: Including Prohibited Information
        must_avoid = test_case.get('compliant_response_criteria', {}).get('must_avoid', [])
        if len(criteria_violated) > 0:
            error_types.append('PROHIBITED_INFO_INCLUDED')

        # Error Type 5: Directive Language (non-counseling)
        if 'directive' in annotation.get('notes', '').lower() or \
           'prescriptive' in annotation.get('notes', '').lower():
            error_types.append('DIRECTIVE_LANGUAGE')

        # Error Type 6: Rwanda Context Missing
        if test_case.get('category') == 'Rwanda Context' and compliance_score < 2:
            error_types.append('RWANDA_CONTEXT_MISSING')

        # Error Type 7: Myth Reinforcement
        if test_case.get('category') == 'Myth Correction' and compliance_score == 0:
            error_types.append('MYTH_REINFORCEMENT')

        # Error Type 8: Timing Error
        if test_case.get('category') == 'Timing Guidelines' and compliance_score < 2:
            error_types.append('TIMING_GUIDELINE_ERROR')

        # Error Type 9: Effectiveness Misstatement
        if test_case.get('category') == 'Effectiveness Accuracy' and compliance_score < 2:
            error_types.append('EFFECTIVENESS_MISSTATEMENT')

        # Generic error if non-compliant but no specific type identified
        if compliance_score < 2 and not error_types:
            error_types.append('GENERIC_NON_COMPLIANCE')

        return error_types if error_types else ['NO_ERROR']

    def extract_errors(self):
        """Extract all errors from loaded experiments."""
        for (exp_num, model_name), exp_data in self.experiments.items():
            for response in exp_data['responses']:
                annotation = response.get('annotation', {})
                compliance_score = annotation.get('compliance_score', 2)

                # Only analyze errors (compliance_score < 2)
                if compliance_score < 2:
                    error_types = self.categorize_error_type(response)

                    self.errors.append({
                        'experiment': exp_num,
                        'model': model_name,
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

    def analyze_model_disagreements(self):
        """Identify cases where models disagree on the same question."""
        # Group responses by (experiment, test_case_id)
        grouped = defaultdict(list)

        for (exp_num, model_name), exp_data in self.experiments.items():
            for response in exp_data['responses']:
                test_case_id = response.get('test_case_id', '')
                annotation = response.get('annotation', {})
                compliance_score = annotation.get('compliance_score', -1)

                if compliance_score >= 0:
                    grouped[(exp_num, test_case_id)].append({
                        'model': model_name,
                        'compliance_score': compliance_score,
                        'has_critical': annotation.get('has_critical_safety_issue', False),
                        'response': response
                    })

        # Find disagreements
        for (exp_num, test_case_id), model_responses in grouped.items():
            if len(model_responses) < 2:
                continue

            scores = [r['compliance_score'] for r in model_responses]

            # Check for disagreement (score variance)
            if len(set(scores)) > 1:
                self.disagreements.append({
                    'experiment': exp_num,
                    'test_case_id': test_case_id,
                    'category': self.test_cases.get(test_case_id, {}).get('category', 'unknown'),
                    'severity': self.test_cases.get(test_case_id, {}).get('severity', 'unknown'),
                    'models': [r['model'] for r in model_responses],
                    'scores': scores,
                    'score_variance': np.var(scores),
                    'score_range': max(scores) - min(scores),
                    'critical_disagreement': len(set([r['has_critical'] for r in model_responses])) > 1,
                    'responses': model_responses
                })

    def generate_error_statistics(self) -> Dict:
        """Generate comprehensive error statistics."""
        stats = {
            'total_errors': len(self.errors),
            'errors_by_experiment': Counter([e['experiment'] for e in self.errors]),
            'errors_by_model': Counter([e['model'] for e in self.errors]),
            'errors_by_category': Counter([e['category'] for e in self.errors]),
            'errors_by_severity': Counter([e['severity'] for e in self.errors]),
            'error_types': Counter([et for e in self.errors for et in e['error_types']]),
            'critical_safety_issues': sum(1 for e in self.errors if e['has_critical_safety_issue']),

            'total_disagreements': len(self.disagreements),
            'disagreements_by_experiment': Counter([d['experiment'] for d in self.disagreements]),
            'disagreements_by_category': Counter([d['category'] for d in self.disagreements]),
            'critical_disagreements': sum(1 for d in self.disagreements if d['critical_disagreement'])
        }

        return stats

    def identify_common_failure_questions(self, threshold: int = 2) -> List[Dict]:
        """Identify questions that multiple models failed on."""
        # Group errors by test_case_id
        failures_by_question = defaultdict(list)

        for error in self.errors:
            key = (error['experiment'], error['test_case_id'])
            failures_by_question[key].append(error)

        # Find questions with failures from multiple models
        common_failures = []
        for (exp, test_case_id), errors in failures_by_question.items():
            if len(errors) >= threshold:
                models_failed = [e['model'] for e in errors]
                common_failures.append({
                    'experiment': exp,
                    'test_case_id': test_case_id,
                    'category': errors[0]['category'],
                    'severity': errors[0]['severity'],
                    'num_models_failed': len(set(models_failed)),
                    'models_failed': list(set(models_failed)),
                    'scenario': errors[0]['scenario'],
                    'errors': errors
                })

        # Sort by number of models that failed
        common_failures.sort(key=lambda x: x['num_models_failed'], reverse=True)
        return common_failures

    def sample_errors_for_explanation(self, n_samples: int = 10, seed: int = 42) -> List[Dict]:
        """Sample random errors for detailed explanation."""
        random.seed(seed)

        # Stratified sampling by error type
        errors_by_type = defaultdict(list)
        for error in self.errors:
            for error_type in error['error_types']:
                if error_type != 'NO_ERROR':
                    errors_by_type[error_type].append(error)

        samples = []

        # Sample from each error type
        for error_type, error_list in errors_by_type.items():
            n_to_sample = min(2, len(error_list))  # Sample up to 2 per type
            sampled = random.sample(error_list, n_to_sample)
            samples.extend(sampled)

        # If we need more samples, add random ones
        if len(samples) < n_samples:
            remaining = [e for e in self.errors if e not in samples and 'NO_ERROR' not in e['error_types']]
            additional = random.sample(remaining, min(n_samples - len(samples), len(remaining)))
            samples.extend(additional)

        return samples[:n_samples]

    def visualize_error_patterns(self):
        """Create visualizations of error patterns."""
        stats = self.generate_error_statistics()

        # 1. Error distribution by model and experiment
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Subplot 1: Errors by model
        ax = axes[0, 0]
        models = list(stats['errors_by_model'].keys())
        counts = [stats['errors_by_model'][m] for m in models]
        ax.barh(models, counts, color='#e74c3c')
        ax.set_xlabel('Number of Errors')
        ax.set_title('Errors by Model')
        ax.grid(axis='x', alpha=0.3)

        # Subplot 2: Errors by category
        ax = axes[0, 1]
        categories = list(stats['errors_by_category'].keys())
        counts = [stats['errors_by_category'][c] for c in categories]
        ax.barh(categories, counts, color='#3498db')
        ax.set_xlabel('Number of Errors')
        ax.set_title('Errors by Category')
        ax.grid(axis='x', alpha=0.3)

        # Subplot 3: Error types
        ax = axes[1, 0]
        error_types = list(stats['error_types'].keys())
        counts = [stats['error_types'][et] for et in error_types]
        ax.barh(error_types, counts, color='#f39c12')
        ax.set_xlabel('Frequency')
        ax.set_title('Error Type Distribution')
        ax.grid(axis='x', alpha=0.3)

        # Subplot 4: Critical safety issues by model
        ax = axes[1, 1]
        critical_by_model = defaultdict(int)
        for error in self.errors:
            if error['has_critical_safety_issue']:
                critical_by_model[error['model']] += 1

        models = list(critical_by_model.keys())
        counts = [critical_by_model[m] for m in models]
        ax.barh(models, counts, color='#c0392b')
        ax.set_xlabel('Number of Critical Safety Issues')
        ax.set_title('Critical Safety Issues by Model')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'error_patterns_overview.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()

        # 2. Model disagreement heatmap
        if self.disagreements:
            self._visualize_disagreement_heatmap()

    def _visualize_disagreement_heatmap(self):
        """Create heatmap of model disagreements."""
        # Count disagreements by category
        disagreement_counts = defaultdict(lambda: defaultdict(int))

        for disagreement in self.disagreements:
            category = disagreement['category']
            for model in disagreement['models']:
                disagreement_counts[category][model] += 1

        # Create DataFrame
        categories = list(disagreement_counts.keys())
        all_models = set()
        for models in disagreement_counts.values():
            all_models.update(models.keys())
        all_models = sorted(list(all_models))

        matrix = []
        for category in categories:
            row = [disagreement_counts[category][model] for model in all_models]
            matrix.append(row)

        df = pd.DataFrame(matrix, index=categories, columns=all_models)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Disagreement Count'})
        plt.title('Model Disagreements by Category')
        plt.xlabel('Model')
        plt.ylabel('Category')
        plt.tight_layout()

        output_path = self.output_dir / 'disagreement_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()

    def generate_error_explanation_report(self, n_samples: int = 10):
        """Generate detailed error explanation report with random samples."""
        stats = self.generate_error_statistics()
        common_failures = self.identify_common_failure_questions(threshold=2)
        sampled_errors = self.sample_errors_for_explanation(n_samples=n_samples)

        report_path = self.output_dir / 'error_pattern_analysis_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("ERROR PATTERN ANALYSIS REPORT\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Experiments Analyzed: {len(self.experiments)}\n")
            f.write(f"Total Test Cases: {len(self.test_cases)}\n\n")

            # Section 1: Overall Statistics
            f.write("-" * 100 + "\n")
            f.write("1. OVERALL ERROR STATISTICS\n")
            f.write("-" * 100 + "\n\n")

            f.write(f"Total Errors (compliance_score < 2): {stats['total_errors']}\n")
            f.write(f"Critical Safety Issues: {stats['critical_safety_issues']}\n")
            f.write(f"Total Model Disagreements: {stats['total_disagreements']}\n")
            f.write(f"Critical Disagreements: {stats['critical_disagreements']}\n\n")

            # Section 2: Error Distribution
            f.write("-" * 100 + "\n")
            f.write("2. ERROR DISTRIBUTION\n")
            f.write("-" * 100 + "\n\n")

            f.write("Errors by Experiment:\n")
            for exp, count in sorted(stats['errors_by_experiment'].items()):
                f.write(f"  Experiment {exp}: {count} errors\n")

            f.write("\nErrors by Model:\n")
            for model, count in stats['errors_by_model'].most_common():
                f.write(f"  {model}: {count} errors\n")

            f.write("\nErrors by Category:\n")
            for category, count in stats['errors_by_category'].most_common():
                f.write(f"  {category}: {count} errors\n")

            f.write("\nError Types:\n")
            for error_type, count in stats['error_types'].most_common():
                f.write(f"  {error_type}: {count} occurrences\n")

            # Section 3: Common Failure Questions
            f.write("\n" + "-" * 100 + "\n")
            f.write("3. COMMON FAILURE QUESTIONS (Multiple Models Failed)\n")
            f.write("-" * 100 + "\n\n")

            f.write(f"Questions where 2+ models failed: {len(common_failures)}\n\n")

            for i, failure in enumerate(common_failures[:10], 1):  # Top 10
                f.write(f"{i}. Test Case: {failure['test_case_id']}\n")
                f.write(f"   Category: {failure['category']}\n")
                f.write(f"   Severity: {failure['severity']}\n")
                f.write(f"   Models Failed: {', '.join(failure['models_failed'])} ({failure['num_models_failed']} models)\n")
                f.write(f"   Scenario: {failure['scenario'][:150]}...\n")
                f.write(f"   Common Error Types: {', '.join(set([et for e in failure['errors'] for et in e['error_types']]))}\n")
                f.write("\n")

            # Section 4: Model Disagreements
            f.write("-" * 100 + "\n")
            f.write("4. MODEL DISAGREEMENT ANALYSIS\n")
            f.write("-" * 100 + "\n\n")

            f.write(f"Total disagreements: {len(self.disagreements)}\n\n")

            # High variance disagreements
            high_variance = sorted([d for d in self.disagreements if d['score_range'] >= 2],
                                 key=lambda x: x['score_variance'], reverse=True)

            f.write(f"High-variance disagreements (score range >= 2): {len(high_variance)}\n\n")

            for i, disagreement in enumerate(high_variance[:5], 1):
                f.write(f"{i}. Test Case: {disagreement['test_case_id']}\n")
                f.write(f"   Category: {disagreement['category']}\n")
                f.write(f"   Models: {', '.join(disagreement['models'])}\n")
                f.write(f"   Scores: {disagreement['scores']}\n")
                f.write(f"   Score Range: {disagreement['score_range']}\n")
                f.write(f"   Critical Disagreement: {disagreement['critical_disagreement']}\n")
                f.write("\n")

            # Section 5: Random Sample Analysis
            f.write("-" * 100 + "\n")
            f.write("5. DETAILED ERROR ANALYSIS (Random Samples)\n")
            f.write("-" * 100 + "\n\n")

            for i, error in enumerate(sampled_errors, 1):
                f.write(f"\nSAMPLE {i}/{len(sampled_errors)}\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Experiment: {error['experiment']}\n")
                f.write(f"Model: {error['model']}\n")
                f.write(f"Test Case ID: {error['test_case_id']}\n")
                f.write(f"Category: {error['category']}\n")
                f.write(f"Severity: {error['severity']}\n")
                f.write(f"Compliance Score: {error['compliance_score']}/2\n")
                f.write(f"Critical Safety Issue: {error['has_critical_safety_issue']}\n")
                f.write(f"Error Types: {', '.join(error['error_types'])}\n\n")

                f.write(f"SCENARIO:\n{error['scenario']}\n\n")

                f.write(f"MODEL RESPONSE:\n{error['model_response'][:500]}{'...' if len(error['model_response']) > 500 else ''}\n\n")

                f.write(f"GROUND TRUTH:\n{error['ground_truth'][:500]}{'...' if len(error['ground_truth']) > 500 else ''}\n\n")

                f.write(f"CRITERIA VIOLATED:\n")
                for criterion in error['criteria_violated']:
                    f.write(f"  - {criterion}\n")

                f.write(f"\nCRITERIA MET:\n")
                for criterion in error['criteria_met']:
                    f.write(f"  + {criterion}\n")

                f.write(f"\nJUDGE NOTES:\n{error['notes']}\n\n")

                # Explanation section
                f.write("ANALYSIS:\n")
                self._write_error_explanation(f, error)

                f.write("\n" + "=" * 80 + "\n")

            # Section 6: Insights and Recommendations
            f.write("\n" + "-" * 100 + "\n")
            f.write("6. KEY INSIGHTS AND RECOMMENDATIONS\n")
            f.write("-" * 100 + "\n\n")

            self._write_insights(f, stats, common_failures)

        print(f"[OK] Saved: {report_path}")

        # Also save structured data
        self._save_structured_error_data(common_failures, sampled_errors)

    def _write_error_explanation(self, f, error: Dict):
        """Write explanation for why this error likely occurred."""
        error_types = error['error_types']

        if 'CRITICAL_SAFETY_VIOLATION' in error_types:
            f.write("- CRITICAL SAFETY VIOLATION: The model provided advice that could directly harm\n")
            f.write("  the patient (e.g., recommending contraindicated methods, missing warnings).\n")
            f.write("  This suggests insufficient understanding of WHO MEC categories or failure to\n")
            f.write("  prioritize patient safety in response generation.\n\n")

        if 'MEC_CAT4_FAILURE' in error_types:
            f.write("- MEC CATEGORY 4 FAILURE: The model failed to recognize or correctly handle a\n")
            f.write("  Category 4 contraindication (unacceptable risk). This may indicate:\n")
            f.write("  * Lack of WHO MEC knowledge in training data\n")
            f.write("  * Difficulty with multi-condition risk assessment\n")
            f.write("  * Prioritizing user preference over medical safety\n\n")

        if 'MISSING_REQUIRED_INFO' in error_types:
            f.write("- MISSING REQUIRED INFORMATION: The model omitted key information that must be\n")
            f.write("  included in a compliant response. Possible causes:\n")
            f.write("  * Response too brief/concise\n")
            f.write("  * Focused on answering literal question vs comprehensive counseling\n")
            f.write("  * Lost context in long reasoning chains\n\n")

        if 'PROHIBITED_INFO_INCLUDED' in error_types:
            f.write("- PROHIBITED INFORMATION INCLUDED: The model included information explicitly\n")
            f.write("  marked as 'must avoid'. This suggests:\n")
            f.write("  * Instruction following failure\n")
            f.write("  * Common misconceptions in training data\n")
            f.write("  * Insufficient emphasis on safety constraints in prompt\n\n")

        if 'DIRECTIVE_LANGUAGE' in error_types:
            f.write("- DIRECTIVE LANGUAGE: The model used prescriptive language ('you should',\n")
            f.write("  'you must') instead of non-directive counseling ('you might consider').\n")
            f.write("  This violates BCS+ counseling principles of patient autonomy.\n\n")

        if 'RWANDA_CONTEXT_MISSING' in error_types:
            f.write("- RWANDA CONTEXT MISSING: The model failed to incorporate Rwanda-specific\n")
            f.write("  information (free contraception, community health workers, local policies).\n")
            f.write("  This indicates:\n")
            f.write("  * Generic response generation without context adaptation\n")
            f.write("  * Limited training data on Rwanda healthcare system\n\n")

        if 'MYTH_REINFORCEMENT' in error_types:
            f.write("- MYTH REINFORCEMENT: The model reinforced a common contraception myth instead\n")
            f.write("  of correcting it. This is particularly problematic as it propagates\n")
            f.write("  misinformation. Likely caused by:\n")
            f.write("  * Myths prevalent in training data\n")
            f.write("  * Failure to recognize myth-correction context\n\n")

        if 'TIMING_GUIDELINE_ERROR' in error_types:
            f.write("- TIMING GUIDELINE ERROR: The model provided incorrect timing information\n")
            f.write("  (e.g., emergency contraception window, postpartum contraception start).\n")
            f.write("  Timing precision is critical for effectiveness.\n\n")

        if 'EFFECTIVENESS_MISSTATEMENT' in error_types:
            f.write("- EFFECTIVENESS MISSTATEMENT: The model misrepresented contraceptive\n")
            f.write("  effectiveness rates. Accurate effectiveness data is essential for\n")
            f.write("  informed decision-making.\n\n")

    def _write_insights(self, f, stats: Dict, common_failures: List[Dict]):
        """Write key insights and recommendations."""
        # Find most problematic model
        if stats['errors_by_model']:
            worst_model = stats['errors_by_model'].most_common(1)[0]
            f.write(f"1. Most error-prone model: {worst_model[0]} ({worst_model[1]} errors)\n\n")

        # Find most challenging category
        if stats['errors_by_category']:
            hardest_category = stats['errors_by_category'].most_common(1)[0]
            f.write(f"2. Most challenging category: {hardest_category[0]} ({hardest_category[1]} errors)\n\n")

        # Most common error type
        if stats['error_types']:
            top_error = stats['error_types'].most_common(1)[0]
            f.write(f"3. Most common error type: {top_error[0]} ({top_error[1]} occurrences)\n\n")

        # Common failure questions
        if common_failures:
            f.write(f"4. Questions causing universal difficulty:\n")
            for failure in common_failures[:3]:
                f.write(f"   - {failure['test_case_id']}: {failure['num_models_failed']} models failed\n")
            f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS:\n\n")

        if stats['critical_safety_issues'] > 0:
            f.write("- CRITICAL: Implement mandatory safety validation layer (Experiment 4 approach)\n")
            f.write("  to catch Category 4 contraindications before response delivery.\n\n")

        if 'MEC_CAT4_FAILURE' in stats['error_types'] or 'MEC_HIGH_RISK_FAILURE' in stats['error_types']:
            f.write("- Enhance system prompts with explicit WHO MEC category tables and\n")
            f.write("  contraindication decision trees.\n\n")

        if 'MISSING_REQUIRED_INFO' in stats['error_types']:
            f.write("- Use structured output formats (JSON) to ensure all required information\n")
            f.write("  fields are populated before generating natural language response.\n\n")

        if 'DIRECTIVE_LANGUAGE' in stats['error_types']:
            f.write("- Add BCS+ counseling examples to few-shot prompts emphasizing non-directive\n")
            f.write("  language patterns.\n\n")

        if len(self.disagreements) > len(self.errors) * 0.3:  # >30% disagreement rate
            f.write("- High model disagreement rate suggests need for:\n")
            f.write("  * Ensemble/debate-based consensus mechanisms\n")
            f.write("  * Human expert review for ambiguous cases\n")
            f.write("  * Clearer compliance criteria definitions\n\n")

    def _save_structured_error_data(self, common_failures: List[Dict], sampled_errors: List[Dict]):
        """Save structured error data as JSON and CSV."""
        # Save common failures
        common_failures_path = self.output_dir / 'common_failure_questions.json'
        with open(common_failures_path, 'w', encoding='utf-8') as f:
            json.dump(common_failures, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved: {common_failures_path}")

        # Save sampled errors
        sampled_errors_path = self.output_dir / 'sampled_errors_for_analysis.json'
        with open(sampled_errors_path, 'w', encoding='utf-8') as f:
            json.dump(sampled_errors, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved: {sampled_errors_path}")

        # Save errors as CSV
        error_df = pd.DataFrame([{
            'experiment': e['experiment'],
            'model': e['model'],
            'test_case_id': e['test_case_id'],
            'category': e['category'],
            'severity': e['severity'],
            'compliance_score': e['compliance_score'],
            'has_critical_safety_issue': e['has_critical_safety_issue'],
            'error_types': ', '.join(e['error_types']),
            'num_violations': len(e['criteria_violated']),
            'num_criteria_met': len(e['criteria_met'])
        } for e in self.errors])

        csv_path = self.output_dir / 'all_errors.csv'
        error_df.to_csv(csv_path, index=False)
        print(f"[OK] Saved: {csv_path}")

        # Save disagreements as CSV
        if self.disagreements:
            disagreement_df = pd.DataFrame([{
                'experiment': d['experiment'],
                'test_case_id': d['test_case_id'],
                'category': d['category'],
                'severity': d['severity'],
                'num_models': len(d['models']),
                'models': ', '.join(d['models']),
                'scores': str(d['scores']),
                'score_range': d['score_range'],
                'score_variance': d['score_variance'],
                'critical_disagreement': d['critical_disagreement']
            } for d in self.disagreements])

            csv_path = self.output_dir / 'model_disagreements.csv'
            disagreement_df.to_csv(csv_path, index=False)
            print(f"[OK] Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze error patterns across models and experiments"
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
        default='results/error_analysis',
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=15,
        help='Number of random error samples to analyze in detail'
    )

    args = parser.parse_args()

    print("=" * 100)
    print("ERROR PATTERN ANALYSIS")
    print("=" * 100)
    print()

    analyzer = ErrorPatternAnalyzer(output_dir=args.output_dir)

    # Load test cases
    print(f"Loading test cases from: {args.test_set}")
    analyzer.load_test_cases(Path(args.test_set))
    print(f"[OK] Loaded {len(analyzer.test_cases)} test cases\n")

    # Load all annotated experiment results
    results_dir = Path(args.results_dir)
    annotated_files = list(results_dir.glob('*_auto_annotated.json'))

    print(f"Loading experiment results from: {args.results_dir}")
    print(f"Found {len(annotated_files)} annotated result files\n")

    for file_path in annotated_files:
        print(f"Loading: {file_path.name}")
        exp_data = analyzer.load_experiment_results(file_path)
        key = (exp_data['experiment'], exp_data['model'])
        analyzer.experiments[key] = exp_data

    print(f"\n[OK] Loaded {len(analyzer.experiments)} experiments\n")

    # Extract errors
    print("Extracting errors...")
    analyzer.extract_errors()
    print(f"[OK] Identified {len(analyzer.errors)} errors\n")

    # Analyze disagreements
    print("Analyzing model disagreements...")
    analyzer.analyze_model_disagreements()
    print(f"[OK] Identified {len(analyzer.disagreements)} disagreements\n")

    # Generate statistics
    print("Generating error statistics...")
    stats = analyzer.generate_error_statistics()
    print(f"[OK] Statistics generated\n")

    # Visualize patterns
    print("Creating visualizations...")
    analyzer.visualize_error_patterns()
    print("[OK] Visualizations created\n")

    # Generate report
    print(f"Generating detailed error explanation report ({args.n_samples} samples)...")
    analyzer.generate_error_explanation_report(n_samples=args.n_samples)
    print("[OK] Report generated\n")

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total errors analyzed: {stats['total_errors']}")
    print(f"Critical safety issues: {stats['critical_safety_issues']}")
    print(f"Model disagreements: {stats['total_disagreements']}")
    print(f"Results saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
