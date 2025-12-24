"""
Base Analyzer - Shared functionality for all analysis scripts.

This module consolidates duplicate code from analyze_error_patterns.py,
analyze_errors_by_experiment.py, and analyze_errors_by_model.py.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class BaseAnalyzer:
    """Shared functionality for all analysis scripts."""

    def __init__(self, output_dir: str = "results/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiments = {}  # {(exp_num, model_name): data}
        self.test_cases = {}   # {test_case_id: ground truth data}
        self.errors = []       # List of error instances

    def load_experiment_results(self, file_path: Path) -> Dict:
        """
        Load annotated experiment results.

        Single source of truth for loading result files.
        """
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

    def categorize_error_type(self, response: Dict, exp_num: Optional[int] = None) -> List[str]:
        """
        Categorize the type of error(s) in a response.

        Single source of truth for error categorization logic.
        """
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

        # Error Type 2: MEC Category 4 Failure
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

        # Error Type 8: Timing Guideline Error
        if test_case.get('category') == 'Timing Guidelines' and compliance_score < 2:
            error_types.append('TIMING_GUIDELINE_ERROR')

        # Error Type 9: Effectiveness Misstatement
        if test_case.get('category') == 'Effectiveness Accuracy' and compliance_score < 2:
            error_types.append('EFFECTIVENESS_MISSTATEMENT')

        # Experiment-specific error patterns
        if exp_num == 1:
            # Baseline lacks WHO context
            if 'MEC' in test_case_id.upper() and compliance_score < 2:
                error_types.append('BASELINE_NO_WHO_CONTEXT')
        elif exp_num == 3:
            # RAG-specific issues
            if 'RAG' in str(response.get('model_response', '')).upper() or \
               len(response.get('model_response', '')) > 1000:
                error_types.append('RAG_CONTEXT_OVERLOAD')

        # Generic error if non-compliant but no specific type identified
        if compliance_score < 2 and not error_types:
            error_types.append('GENERIC_NON_COMPLIANCE')

        return error_types if error_types else ['NO_ERROR']

    def extract_errors_from_experiment(self, exp_num: int, model_name: str, responses: List[Dict]) -> List[Dict]:
        """
        Extract all errors from a single experiment.

        Args:
            exp_num: Experiment number
            model_name: Model name
            responses: List of response dictionaries

        Returns:
            List of error dictionaries
        """
        errors = []

        for response in responses:
            annotation = response.get('annotation', {})
            compliance_score = annotation.get('compliance_score', 2)

            # Only analyze errors (compliance_score < 2)
            if compliance_score < 2:
                error_types = self.categorize_error_type(response, exp_num)

                errors.append({
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

        return errors

    def load_all_experiments(self, results_dir: Path) -> Dict:
        """
        Load all annotated experiment results from a directory.

        Args:
            results_dir: Directory containing *_auto_annotated.json files

        Returns:
            Dictionary mapping (exp_num, model_name) to experiment data
        """
        experiments = {}

        for file_path in results_dir.glob("*_auto_annotated.json"):
            exp_data = self.load_experiment_results(file_path)
            key = (exp_data['experiment'], exp_data['model'])
            experiments[key] = exp_data

        return experiments

    def generate_markdown_report(self, title: str, sections: List[Dict], output_path: Path):
        """
        Generate a markdown report.

        Args:
            title: Report title
            sections: List of section dictionaries with 'heading' and 'content' keys
            output_path: Output file path
        """
        lines = [
            f"# {title}",
            "",
            f"Generated: {self._get_timestamp()}",
            ""
        ]

        for section in sections:
            lines.append(f"## {section['heading']}")
            lines.append("")
            lines.append(section['content'])
            lines.append("")

        output_path.write_text('\n'.join(lines), encoding='utf-8')

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def compute_summary_stats(self, errors: List[Dict]) -> Dict:
        """
        Compute summary statistics for a list of errors.

        Args:
            errors: List of error dictionaries

        Returns:
            Dictionary with summary statistics
        """
        from collections import Counter

        total_errors = len(errors)
        critical_errors = sum(1 for e in errors if e['has_critical_safety_issue'])

        # Error type distribution
        error_type_counts = Counter()
        for error in errors:
            for error_type in error['error_types']:
                error_type_counts[error_type] += 1

        # Category distribution
        category_counts = Counter(e['category'] for e in errors)

        # Model distribution
        model_counts = Counter(e['model'] for e in errors)

        return {
            'total_errors': total_errors,
            'critical_errors': critical_errors,
            'error_type_distribution': dict(error_type_counts),
            'category_distribution': dict(category_counts),
            'model_distribution': dict(model_counts)
        }
