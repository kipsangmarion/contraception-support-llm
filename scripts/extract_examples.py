#!/usr/bin/env python3
"""
Extract illustrative examples from experiment results
"""
import json
import sys
from pathlib import Path

# Force UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def get_example(file_path, test_case_id):
    """Get specific test case from results."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for r in data['responses']:
        if r.get('test_case_id') == test_case_id:
            ann = r.get('annotation', {})
            return {
                'test_case_id': r.get('test_case_id'),
                'scenario': r.get('scenario', ''),
                'model_response': r.get('model_response', ''),
                'compliance_score': ann.get('compliance_score', r.get('compliance_score', 0)),
                'critical_issue': ann.get('has_critical_safety_issue', r.get('critical_safety_issue', False)),
                'reasoning': ann.get('reasoning', '')
            }
    return None

def main():
    results_dir = Path('results/compliance_experiments')

    examples = {}

    # Example 1: Fully compliant (Exp1 - o3, breastfeeding)
    ex1 = get_example(
        results_dir / 'exp1_o3-2025-04-16_20251208_110608_auto_annotated.json',
        'mec_001'
    )
    if ex1:
        examples['exp1_fully_compliant'] = ex1
        print("Example 1: Fully Compliant (Exp1 - o3)")
        print(f"  Scenario: {ex1['scenario'][:80]}...")
        print(f"  Score: {ex1['compliance_score']}/2")
        print()

    # Example 2: Same scenario with RAG (Exp3 - o3)
    ex2 = get_example(
        results_dir / 'exp3_o3-2025-04-16_rag_20251209_004927_auto_annotated.json',
        'mec_001'
    )
    if ex2:
        examples['exp3_rag_degraded'] = ex2
        print("Example 2: RAG Degraded (Exp3 - o3, same scenario)")
        print(f"  Scenario: {ex2['scenario'][:80]}...")
        print(f"  Score: {ex2['compliance_score']}/2")
        print()

    # Example 3: Grok Exp2 improvement
    # Find a case where Grok improved from Exp1 to Exp2
    with open(results_dir / 'exp1_grok-4-1-fast-reasoning_20251208_053400_auto_annotated.json', 'r', encoding='utf-8') as f:
        exp1_grok = json.load(f)
    with open(results_dir / 'exp2_grok-4-1-fast-reasoning_20251208_220605_auto_annotated.json', 'r', encoding='utf-8') as f:
        exp2_grok = json.load(f)

    # Find critical issue in Exp1 that was fixed in Exp2
    for r1 in exp1_grok['responses']:
        ann1 = r1.get('annotation', {})
        if ann1.get('has_critical_safety_issue', False):
            test_id = r1.get('test_case_id')
            # Check if fixed in Exp2
            for r2 in exp2_grok['responses']:
                if r2.get('test_case_id') == test_id:
                    ann2 = r2.get('annotation', {})
                    if not ann2.get('has_critical_safety_issue', False):
                        examples['grok_improvement'] = {
                            'exp1': {
                                'test_case_id': test_id,
                                'scenario': r1.get('scenario', ''),
                                'model_response': r1.get('model_response', ''),
                                'compliance_score': ann1.get('compliance_score', 0),
                                'critical_issue': True
                            },
                            'exp2': {
                                'test_case_id': test_id,
                                'scenario': r2.get('scenario', ''),
                                'model_response': r2.get('model_response', ''),
                                'compliance_score': ann2.get('compliance_score', 0),
                                'critical_issue': False
                            }
                        }
                        print(f"Example 3: Grok Improvement (Exp1→Exp2)")
                        print(f"  Test Case: {test_id}")
                        print(f"  Exp1: Score {ann1.get('compliance_score')}/2, Critical: True")
                        print(f"  Exp2: Score {ann2.get('compliance_score')}/2, Critical: False")
                        print()
                        break
            if 'grok_improvement' in examples:
                break

    # Save examples to JSON
    output_file = Path('results/illustrative_examples.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Saved examples to: {output_file}")
    return examples

if __name__ == '__main__':
    main()
