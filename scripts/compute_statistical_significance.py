"""
Statistical Significance Testing for Experiment Comparisons
Tests whether differences between experiments are statistically significant using:
1. McNemar's test (paired, same test cases)
2. Chi-square test (independence)
3. Effect size (Cohen's h)
4. Confidence intervals
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats
from scipy.stats import chi2_contingency
import pandas as pd
from typing import Dict, List, Tuple

# Manual McNemar's test implementation
def mcnemar_test(table, exact=False, correction=True):
    """
    McNemar's test for paired nominal data
    table: 2x2 contingency table [[a, b], [c, d]]
    """
    b = table[0, 1]  # exp1 correct, exp2 wrong
    c = table[1, 0]  # exp2 correct, exp1 wrong

    if b + c == 0:
        return type('obj', (object,), {'statistic': 0, 'pvalue': 1.0})

    if correction:
        statistic = (abs(b - c) - 1)**2 / (b + c)
    else:
        statistic = (b - c)**2 / (b + c)

    pvalue = 1 - stats.chi2.cdf(statistic, 1)

    return type('obj', (object,), {'statistic': statistic, 'pvalue': pvalue})

def load_experiment_results():
    """Load all annotated experiment results"""
    results = {}
    base_path = Path("results/compliance_experiments")

    # Map experiments to their models
    exp_models = {
        1: ['claude-opus-4-5-20251101', 'o3-2025-04-16', 'grok-4-1-fast-reasoning'],
        2: ['claude-opus-4-5-20251101', 'o3-2025-04-16', 'grok-4-1-fast-reasoning'],
        3: ['claude-opus-4-5-20251101', 'o3-2025-04-16', 'grok-4-1-fast-reasoning'],
        4: ['claude-opus-4-5-20251101']
    }

    for exp_num, models in exp_models.items():
        results[exp_num] = {}
        for model in models:
            # Find the annotated file for this experiment and model
            pattern = f"exp{exp_num}_{model}*_auto_annotated.json"
            files = list(base_path.glob(pattern))
            if files:
                with open(files[0], 'r', encoding='utf-8') as f:
                    results[exp_num][model] = json.load(f)

    return results

def extract_binary_outcomes(results: Dict) -> Dict:
    """Extract binary outcomes (compliant=1, non-compliant=0) for each test case"""
    outcomes = defaultdict(lambda: defaultdict(dict))

    for exp_num, exp_data in results.items():
        for model, data in exp_data.items():
            responses = data.get('responses', [])
            for response in responses:
                test_case_id = response.get('test_case_id')
                compliance_score = response.get('annotation', {}).get('compliance_score', 2)
                # Binary: 1 if fully compliant (score=2), 0 otherwise
                is_compliant = 1 if compliance_score == 2 else 0
                outcomes[exp_num][model][test_case_id] = is_compliant

    return outcomes

def mcnemar_test_pairwise(outcomes: Dict, exp1: int, exp2: int, model: str) -> Dict:
    """
    Perform McNemar's test for paired samples (same test cases, different experiments)
    Tests whether the proportion of errors differs significantly between experiments
    """
    exp1_outcomes = outcomes[exp1][model]
    exp2_outcomes = outcomes[exp2][model]

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
    # [exp2_success & exp1_success, exp2_success & exp1_fail]
    # [exp2_fail & exp1_success, exp2_fail & exp1_fail]
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
    table = np.array([[both_correct, exp1_only],
                      [exp2_only, both_wrong]])

    try:
        result = mcnemar_test(table, exact=False, correction=True)

        return {
            'test': 'mcnemar',
            'statistic': result.statistic,
            'p_value': result.pvalue,
            'significant': result.pvalue < 0.05,
            'n_cases': len(common_cases),
            'contingency_table': {
                'both_correct': both_correct,
                'exp1_only_correct': exp1_only,
                'exp2_only_correct': exp2_only,
                'both_wrong': both_wrong
            },
            'interpretation': f"{'Significant' if result.pvalue < 0.05 else 'Not significant'} difference (p={result.pvalue:.4f})"
        }
    except Exception as e:
        return {
            'test': 'mcnemar',
            'error': str(e),
            'contingency_table': {
                'both_correct': both_correct,
                'exp1_only_correct': exp1_only,
                'exp2_only_correct': exp2_only,
                'both_wrong': both_wrong
            }
        }

def chi_square_test(outcomes: Dict, exp1: int, exp2: int, model: str) -> Dict:
    """Chi-square test for independence"""
    exp1_outcomes = outcomes[exp1][model]
    exp2_outcomes = outcomes[exp2][model]

    common_cases = set(exp1_outcomes.keys()) & set(exp2_outcomes.keys())

    if len(common_cases) < 10:
        return {'test': 'chi_square', 'note': 'Insufficient samples'}

    exp1_successes = sum(exp1_outcomes[c] for c in common_cases)
    exp1_failures = len(common_cases) - exp1_successes
    exp2_successes = sum(exp2_outcomes[c] for c in common_cases)
    exp2_failures = len(common_cases) - exp2_successes

    contingency_table = np.array([
        [exp1_successes, exp1_failures],
        [exp2_successes, exp2_failures]
    ])

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    return {
        'test': 'chi_square',
        'statistic': chi2,
        'p_value': p_value,
        'dof': dof,
        'significant': p_value < 0.05,
        'contingency_table': contingency_table.tolist(),
        'interpretation': f"{'Significant' if p_value < 0.05 else 'Not significant'} association (p={p_value:.4f})"
    }

def cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for proportions
    h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    Interpretation: 0.2 = small, 0.5 = medium, 0.8 = large
    """
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    return h

def calculate_effect_size(outcomes: Dict, exp1: int, exp2: int, model: str) -> Dict:
    """Calculate effect size (Cohen's h) and confidence intervals"""
    exp1_outcomes = outcomes[exp1][model]
    exp2_outcomes = outcomes[exp2][model]

    common_cases = set(exp1_outcomes.keys()) & set(exp2_outcomes.keys())
    n = len(common_cases)

    if n < 10:
        return {'note': 'Insufficient samples'}

    # Calculate proportions
    p1 = sum(exp1_outcomes[c] for c in common_cases) / n
    p2 = sum(exp2_outcomes[c] for c in common_cases) / n

    # Cohen's h
    h = cohens_h(p1, p2)

    # Effect size interpretation
    if abs(h) < 0.2:
        magnitude = "negligible"
    elif abs(h) < 0.5:
        magnitude = "small"
    elif abs(h) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    # Confidence interval for difference in proportions
    diff = p2 - p1
    se = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se

    return {
        'cohens_h': h,
        'magnitude': magnitude,
        'exp1_success_rate': p1,
        'exp2_success_rate': p2,
        'difference': diff,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'interpretation': f"Effect size: {magnitude} (h={h:.3f}), {abs(diff)*100:.1f}% {'improvement' if diff > 0 else 'degradation'}"
    }

def perform_comprehensive_tests():
    """Run all statistical tests for experiment comparisons"""
    print("Loading experiment results...")
    results = load_experiment_results()

    print("Extracting binary outcomes...")
    outcomes = extract_binary_outcomes(results)

    # Define comparisons of interest
    comparisons = [
        (1, 2, "Baseline vs Compliance-Aware"),
        (1, 3, "Baseline vs RAG"),
        (2, 3, "Compliance-Aware vs RAG"),
        (1, 4, "Baseline vs Safety Validation (Claude only)"),
        (2, 4, "Compliance-Aware vs Safety Validation (Claude only)"),
        (3, 4, "RAG vs Safety Validation (Claude only)")
    ]

    all_results = []

    for exp1, exp2, description in comparisons:
        print(f"\n{'='*80}")
        print(f"Comparing: {description}")
        print(f"{'='*80}")

        # Determine which models to test
        models_exp1 = set(outcomes[exp1].keys())
        models_exp2 = set(outcomes[exp2].keys())
        common_models = models_exp1 & models_exp2

        for model in common_models:
            print(f"\n--- Model: {model} ---")

            # McNemar's test (preferred for paired data)
            mcnemar_result = mcnemar_test_pairwise(outcomes, exp1, exp2, model)
            print(f"McNemar's Test: {mcnemar_result.get('interpretation', mcnemar_result.get('note', 'N/A'))}")

            # Chi-square test
            chi_result = chi_square_test(outcomes, exp1, exp2, model)
            print(f"Chi-Square Test: {chi_result.get('interpretation', chi_result.get('note', 'N/A'))}")

            # Effect size
            effect_result = calculate_effect_size(outcomes, exp1, exp2, model)
            print(f"Effect Size: {effect_result.get('interpretation', effect_result.get('note', 'N/A'))}")

            all_results.append({
                'comparison': description,
                'exp1': exp1,
                'exp2': exp2,
                'model': model,
                'mcnemar': mcnemar_result,
                'chi_square': chi_result,
                'effect_size': effect_result
            })

    return all_results

def generate_summary_table(results: List[Dict]):
    """Generate summary table of statistical significance"""
    rows = []

    for result in results:
        comparison = result['comparison']
        model = result['model'].split('_')[0]  # Shorten model name

        mcnemar = result['mcnemar']
        effect = result['effect_size']

        row = {
            'Comparison': comparison,
            'Model': model,
            'p-value': f"{mcnemar.get('p_value', 'N/A'):.4f}" if mcnemar.get('p_value') else 'N/A',
            'Significant': 'Yes' if mcnemar.get('significant') else 'No',
            "Cohen's h": f"{effect.get('cohens_h', 'N/A'):.3f}" if effect.get('cohens_h') else 'N/A',
            'Effect Size': effect.get('magnitude', 'N/A'),
            'Success Rate Change': f"{effect.get('difference', 0)*100:+.1f}%" if effect.get('difference') is not None else 'N/A',
            'n': mcnemar.get('n_cases', 'N/A')
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def main():
    print("="*80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*80)

    results = perform_comprehensive_tests()

    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    df = generate_summary_table(results)
    # Don't print to console due to Unicode issues, save to file instead
    print("\n[Table saved to CSV file - see results/statistical_analysis/significance_tests_summary.csv]")

    # Save results
    output_dir = Path("results/statistical_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save CSV
    df.to_csv(output_dir / "significance_tests_summary.csv", index=False)
    print(f"\n[SAVED] {output_dir / 'significance_tests_summary.csv'}")

    # Save detailed JSON
    with open(output_dir / "significance_tests_detailed.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[SAVED] {output_dir / 'significance_tests_detailed.json'}")

    # Generate interpretation report
    generate_interpretation_report(results, output_dir)

    print("\n[SUCCESS] Statistical significance analysis complete!")

def generate_interpretation_report(results: List[Dict], output_dir: Path):
    """Generate human-readable interpretation report"""
    report = []
    report.append("# Statistical Significance Analysis Report\n")
    report.append("## Summary of Findings\n")

    significant_comparisons = [r for r in results if r['mcnemar'].get('significant')]
    report.append(f"**Total Comparisons:** {len(results)}\n")
    report.append(f"**Statistically Significant Differences:** {len(significant_comparisons)}/{len(results)}\n\n")

    report.append("## Key Findings\n\n")

    # Group by comparison
    by_comparison = defaultdict(list)
    for r in results:
        by_comparison[r['comparison']].append(r)

    for comparison, comp_results in by_comparison.items():
        report.append(f"### {comparison}\n\n")

        for result in comp_results:
            model = result['model']
            mcnemar = result['mcnemar']
            effect = result['effect_size']

            report.append(f"**Model: {model}**\n\n")

            if mcnemar.get('significant'):
                report.append(f"- **Statistical Significance:** YES (p={mcnemar.get('p_value', 'N/A'):.4f})\n")
            else:
                report.append(f"- **Statistical Significance:** NO (p={mcnemar.get('p_value', 'N/A'):.4f})\n")

            if effect.get('cohens_h'):
                report.append(f"- **Effect Size:** {effect['magnitude']} (Cohen's h={effect['cohens_h']:.3f})\n")
                report.append(f"- **Success Rate Change:** {effect['difference']*100:+.1f}%\n")
                report.append(f"- **95% CI:** [{effect['ci_95_lower']*100:.1f}%, {effect['ci_95_upper']*100:.1f}%]\n")

            report.append(f"- **Sample Size:** {mcnemar.get('n_cases', 'N/A')} test cases\n\n")

            # Interpretation
            if mcnemar.get('significant') and effect.get('magnitude') in ['medium', 'large']:
                report.append("**Interpretation:** This difference is both statistically significant AND practically meaningful.\n\n")
            elif mcnemar.get('significant'):
                report.append("**Interpretation:** This difference is statistically significant but the effect size is small.\n\n")
            else:
                report.append("**Interpretation:** This difference is not statistically significant.\n\n")

    # Save report
    with open(output_dir / "significance_interpretation.md", 'w') as f:
        f.writelines(report)
    print(f"[SAVED] {output_dir / 'significance_interpretation.md'}")

if __name__ == "__main__":
    main()
