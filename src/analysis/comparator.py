"""
Unified Comparison Module - Statistical comparison and visualization.

Consolidates functionality from:
- compare_experiments.py
- compare_models.py
- compute_statistical_significance.py
- visualize_experiments.py
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy import stats
import pandas as pd
import json


class ExperimentComparator:
    """Statistical comparison between experiments."""

    def __init__(self):
        self.results = {}  # {exp_num: {model: data}}
        self.outcomes = defaultdict(lambda: defaultdict(dict))

    def load_results(self, results_dir: Path):
        """Load all annotated experiment results."""
        for file_path in results_dir.glob("*_auto_annotated.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract metadata
            if 'experiment_metadata' in data:
                exp_num = data['experiment_metadata']['experiment_number']
                model = data['experiment_metadata']['model']
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
                    continue

                parts = filename.split('_')
                model = parts[1] if len(parts) >= 2 else "unknown"

            # Store results
            if exp_num not in self.results:
                self.results[exp_num] = {}
            self.results[exp_num][model] = data

            # Extract binary outcomes
            responses = data.get('responses', [])
            for response in responses:
                test_case_id = response.get('test_case_id')
                compliance_score = response.get('annotation', {}).get('compliance_score', 2)
                # Binary: 1 if fully compliant (score=2), 0 otherwise
                is_compliant = 1 if compliance_score == 2 else 0
                self.outcomes[exp_num][model][test_case_id] = is_compliant

    def mcnemar_test(self, exp1: int, exp2: int, model: str) -> Dict:
        """
        McNemar's test for paired samples.

        Tests whether the proportion of errors differs significantly
        between two experiments for the same model.

        Args:
            exp1: First experiment number
            exp2: Second experiment number
            model: Model name

        Returns:
            Dictionary with test results
        """
        if model not in self.outcomes[exp1] or model not in self.outcomes[exp2]:
            return {
                'test': 'mcnemar',
                'statistic': None,
                'p_value': None,
                'significant': None,
                'note': f'Model {model} not found in both experiments'
            }

        exp1_outcomes = self.outcomes[exp1][model]
        exp2_outcomes = self.outcomes[exp2][model]

        # Find common test cases
        common_cases = set(exp1_outcomes.keys()) & set(exp2_outcomes.keys())

        if len(common_cases) < 10:
            return {
                'test': 'mcnemar',
                'statistic': None,
                'p_value': None,
                'significant': None,
                'note': f'Insufficient paired samples (n={len(common_cases)})'
            }

        # Create contingency table
        both_correct = 0
        exp1_only = 0  # exp1 correct, exp2 wrong
        exp2_only = 0  # exp2 correct, exp1 wrong
        both_wrong = 0

        for case_id in common_cases:
            e1 = exp1_outcomes[case_id]
            e2 = exp2_outcomes[case_id]

            if e1 == 1 and e2 == 1:
                both_correct += 1
            elif e1 == 1 and e2 == 0:
                exp1_only += 1
            elif e1 == 0 and e2 == 1:
                exp2_only += 1
            else:
                both_wrong += 1

        # McNemar's test focuses on discordant pairs
        b = exp1_only  # exp1 correct, exp2 wrong
        c = exp2_only  # exp2 correct, exp1 wrong

        if b + c == 0:
            return {
                'test': 'mcnemar',
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'n_cases': len(common_cases),
                'note': 'No discordant pairs'
            }

        # McNemar's statistic with continuity correction
        statistic = (abs(b - c) - 1)**2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, 1)

        return {
            'test': 'mcnemar',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_cases': len(common_cases),
            'contingency_table': {
                'both_correct': both_correct,
                'exp1_only_correct': exp1_only,
                'exp2_only_correct': exp2_only,
                'both_wrong': both_wrong
            },
            'interpretation': f"{'Significant' if p_value < 0.05 else 'Not significant'} difference (p={p_value:.4f})"
        }

    def chi_square_test(self, exp1: int, exp2: int, model: str) -> Dict:
        """
        Chi-square test for independence.

        Tests whether experiment type and compliance are independent.

        Args:
            exp1: First experiment number
            exp2: Second experiment number
            model: Model name

        Returns:
            Dictionary with test results
        """
        if model not in self.outcomes[exp1] or model not in self.outcomes[exp2]:
            return {
                'test': 'chi_square',
                'statistic': None,
                'p_value': None,
                'significant': None,
                'note': f'Model {model} not found in both experiments'
            }

        exp1_outcomes = self.outcomes[exp1][model]
        exp2_outcomes = self.outcomes[exp2][model]

        # Count compliant/non-compliant for each experiment
        exp1_compliant = sum(exp1_outcomes.values())
        exp1_total = len(exp1_outcomes)
        exp2_compliant = sum(exp2_outcomes.values())
        exp2_total = len(exp2_outcomes)

        # Create contingency table
        table = np.array([
            [exp1_compliant, exp1_total - exp1_compliant],
            [exp2_compliant, exp2_total - exp2_compliant]
        ])

        chi2, p_value, dof, expected = stats.chi2_contingency(table)

        return {
            'test': 'chi_square',
            'statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05,
            'contingency_table': {
                'exp1': {'compliant': exp1_compliant, 'non_compliant': exp1_total - exp1_compliant},
                'exp2': {'compliant': exp2_compliant, 'non_compliant': exp2_total - exp2_compliant}
            },
            'interpretation': f"{'Significant' if p_value < 0.05 else 'Not significant'} association (p={p_value:.4f})"
        }

    def cohens_h(self, exp1: int, exp2: int, model: str) -> Dict:
        """
        Cohen's h effect size for proportions.

        Measures the magnitude of difference between compliance rates.

        Args:
            exp1: First experiment number
            exp2: Second experiment number
            model: Model name

        Returns:
            Dictionary with effect size results
        """
        if model not in self.outcomes[exp1] or model not in self.outcomes[exp2]:
            return {
                'test': 'cohens_h',
                'effect_size': None,
                'magnitude': None,
                'note': f'Model {model} not found in both experiments'
            }

        exp1_outcomes = self.outcomes[exp1][model]
        exp2_outcomes = self.outcomes[exp2][model]

        # Calculate proportions
        p1 = sum(exp1_outcomes.values()) / len(exp1_outcomes)
        p2 = sum(exp2_outcomes.values()) / len(exp2_outcomes)

        # Cohen's h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
        h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

        # Interpret magnitude
        abs_h = abs(h)
        if abs_h < 0.2:
            magnitude = 'negligible'
        elif abs_h < 0.5:
            magnitude = 'small'
        elif abs_h < 0.8:
            magnitude = 'medium'
        else:
            magnitude = 'large'

        return {
            'test': 'cohens_h',
            'effect_size': h,
            'abs_effect_size': abs_h,
            'magnitude': magnitude,
            'p1': p1,
            'p2': p2,
            'interpretation': f"{magnitude.capitalize()} effect (h={h:.3f})"
        }

    def compare_experiments_pairwise(
        self,
        exp1: int,
        exp2: int,
        model: str
    ) -> Dict:
        """
        Comprehensive pairwise comparison between two experiments.

        Args:
            exp1: First experiment number
            exp2: Second experiment number
            model: Model name

        Returns:
            Dictionary with all comparison results
        """
        return {
            'exp1': exp1,
            'exp2': exp2,
            'model': model,
            'mcnemar': self.mcnemar_test(exp1, exp2, model),
            'chi_square': self.chi_square_test(exp1, exp2, model),
            'cohens_h': self.cohens_h(exp1, exp2, model)
        }

    def compare_all_experiments(self, model: str) -> List[Dict]:
        """
        Compare all experiment pairs for a given model.

        Args:
            model: Model name

        Returns:
            List of pairwise comparison results
        """
        comparisons = []
        experiments = sorted(self.results.keys())

        for i, exp1 in enumerate(experiments):
            for exp2 in experiments[i+1:]:
                if model in self.outcomes[exp1] and model in self.outcomes[exp2]:
                    comparison = self.compare_experiments_pairwise(exp1, exp2, model)
                    comparisons.append(comparison)

        return comparisons

    def get_summary_statistics(self, exp_num: int, model: str) -> Dict:
        """
        Get summary statistics for an experiment.

        Args:
            exp_num: Experiment number
            model: Model name

        Returns:
            Dictionary with summary statistics
        """
        if exp_num not in self.results or model not in self.results[exp_num]:
            return {}

        data = self.results[exp_num][model]
        responses = data.get('responses', [])

        if not responses:
            return {}

        total = len(responses)
        fully_compliant = sum(
            1 for r in responses
            if r.get('annotation', {}).get('compliance_score', 0) == 2
        )
        partially_compliant = sum(
            1 for r in responses
            if r.get('annotation', {}).get('compliance_score', 0) == 1
        )
        non_compliant = sum(
            1 for r in responses
            if r.get('annotation', {}).get('compliance_score', 0) == 0
        )
        critical_issues = sum(
            1 for r in responses
            if r.get('annotation', {}).get('has_critical_safety_issue', False)
        )

        # Calculate average score
        scores = [
            r.get('annotation', {}).get('compliance_score', 0)
            for r in responses
        ]
        avg_score = np.mean(scores) if scores else 0

        return {
            'experiment': exp_num,
            'model': model,
            'total_responses': total,
            'fully_compliant': fully_compliant,
            'fully_compliant_pct': fully_compliant / total * 100,
            'partially_compliant': partially_compliant,
            'partially_compliant_pct': partially_compliant / total * 100,
            'non_compliant': non_compliant,
            'non_compliant_pct': non_compliant / total * 100,
            'critical_issues': critical_issues,
            'avg_score': avg_score
        }

    def generate_comparison_table(
        self,
        experiments: List[int],
        models: List[str]
    ) -> pd.DataFrame:
        """
        Generate comparison table across experiments and models.

        Args:
            experiments: List of experiment numbers
            models: List of model names

        Returns:
            Pandas DataFrame with comparison data
        """
        data = []

        for exp_num in experiments:
            for model in models:
                stats = self.get_summary_statistics(exp_num, model)
                if stats:
                    data.append({
                        'Experiment': f"Exp{exp_num}",
                        'Model': model,
                        'Compliance Rate (%)': f"{stats['fully_compliant_pct']:.1f}%",
                        'Critical Issues': stats['critical_issues'],
                        'Avg Score': f"{stats['avg_score']:.2f}/2.0"
                    })

        return pd.DataFrame(data)
