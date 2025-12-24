"""
Test script for unified analysis pipeline.
Tests all analysis modes without requiring full dataset.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.visualizer import UnifiedVisualizer


# Create mock test data
def create_mock_test_set():
    """Create minimal test set for testing."""
    return {
        'test_cases': [
            {
                'id': 'test_001',
                'category': 'Medical Eligibility Criteria',
                'mec_category': 4,
                'severity': 'high',
                'compliant_response_criteria': {
                    'must_include': ['Category 4', 'contraindicated'],
                    'must_avoid': ['safe', 'recommended']
                }
            },
            {
                'id': 'test_002',
                'category': 'Effectiveness Accuracy',
                'severity': 'medium',
                'compliant_response_criteria': {
                    'must_include': ['99% effective'],
                    'must_avoid': ['100% effective']
                }
            }
        ]
    }


def create_mock_experiment_results(exp_num, model_name):
    """Create mock annotated results."""
    return {
        'experiment_metadata': {
            'experiment_number': exp_num,
            'model': model_name,
            'timestamp': datetime.now().isoformat()
        },
        'responses': [
            {
                'test_case_id': 'test_001',
                'category': 'Medical Eligibility Criteria',
                'model_response': 'Mock response 1',
                'annotation': {
                    'compliance_score': 0,  # Error
                    'has_critical_safety_issue': True,
                    'criteria_violated': ['safe', 'recommended'],
                    'criteria_met': [],
                    'notes': 'Failed to warn about Category 4 contraindication'
                }
            },
            {
                'test_case_id': 'test_002',
                'category': 'Effectiveness Accuracy',
                'model_response': 'Mock response 2',
                'annotation': {
                    'compliance_score': 2,  # No error
                    'has_critical_safety_issue': False,
                    'criteria_violated': [],
                    'criteria_met': ['99% effective'],
                    'notes': 'Correct effectiveness information'
                }
            }
        ]
    }


def test_base_analyzer():
    """Test BaseAnalyzer functionality."""
    print("\n" + "="*80)
    print("TEST: BaseAnalyzer")
    print("="*80)

    try:
        # Create mock data directory
        test_dir = Path("test_data/mock_results")
        test_dir.mkdir(parents=True, exist_ok=True)

        # Write mock test set
        test_set_path = test_dir / "test_set.json"
        with open(test_set_path, 'w') as f:
            json.dump(create_mock_test_set(), f)

        # Write mock results
        for exp_num in [1, 2]:
            for model in ['claude-opus-4-5', 'o3-2025']:
                result_file = test_dir / f"exp{exp_num}_{model}_auto_annotated.json"
                with open(result_file, 'w') as f:
                    json.dump(create_mock_experiment_results(exp_num, model), f)

        # Test analyzer
        analyzer = BaseAnalyzer(output_dir="test_data/output")

        # Test loading test cases
        analyzer.load_test_cases(test_set_path)
        assert len(analyzer.test_cases) == 2, "Should load 2 test cases"
        print("[OK] load_test_cases works")

        # Test loading experiment results
        exp_data = analyzer.load_experiment_results(
            test_dir / "exp1_claude-opus-4-5_auto_annotated.json"
        )
        assert exp_data['experiment'] == 1, "Should extract exp number"
        assert 'claude-opus' in exp_data['model'], "Should extract model name"
        print("[OK] load_experiment_results works")

        # Test error categorization
        response = exp_data['responses'][0]  # The error response
        error_types = analyzer.categorize_error_type(response)
        assert 'MEC_CAT4_FAILURE' in error_types, "Should categorize MEC error"
        assert 'CRITICAL_SAFETY_VIOLATION' in error_types, "Should identify critical"
        print("[OK] categorize_error_type works")

        # Test error extraction
        errors = analyzer.extract_errors_from_experiment(
            exp_num=1,
            model_name='claude-opus-4-5',
            responses=exp_data['responses']
        )
        assert len(errors) == 1, "Should extract 1 error"
        assert errors[0]['has_critical_safety_issue'], "Should flag critical"
        print("[OK] extract_errors_from_experiment works")

        # Test summary stats
        stats = analyzer.compute_summary_stats(errors)
        assert stats['total_errors'] == 1, "Should count 1 error"
        assert stats['critical_errors'] == 1, "Should count 1 critical"
        print("[OK] compute_summary_stats works")

        print("\n[PASS] BaseAnalyzer test passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] BaseAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualizer():
    """Test UnifiedVisualizer functionality."""
    print("\n" + "="*80)
    print("TEST: UnifiedVisualizer")
    print("="*80)

    try:
        visualizer = UnifiedVisualizer(output_dir=Path("test_data/viz"))

        # Create mock error data
        errors = [
            {
                'model': 'claude-opus-4-5',
                'experiment': 1,
                'error_types': ['MEC_CAT4_FAILURE'],
                'has_critical_safety_issue': True,
                'category': 'Medical Eligibility Criteria'
            },
            {
                'model': 'o3-2025',
                'experiment': 1,
                'error_types': ['EFFECTIVENESS_MISSTATEMENT'],
                'has_critical_safety_issue': False,
                'category': 'Effectiveness Accuracy'
            }
        ]

        # Test error distribution plot
        visualizer.plot_error_distribution(
            errors,
            group_by='model',
            title="Test: Error by Model",
            output_name="test_error_by_model.png"
        )
        assert (visualizer.output_dir / "test_error_by_model.png").exists()
        print("[OK] plot_error_distribution works")

        # Test category breakdown
        visualizer.plot_category_breakdown(
            errors,
            title="Test: Category Breakdown",
            output_name="test_category.png"
        )
        assert (visualizer.output_dir / "test_category.png").exists()
        print("[OK] plot_category_breakdown works")

        # Test model trajectories
        model_data = {
            'claude-opus-4-5': {
                'experiments': [1, 2, 3],
                'error_rate': [30, 25, 40],
                'error_counts': [24, 20, 32],
                'critical_counts': [1, 0, 8]
            }
        }
        visualizer.plot_model_trajectories(
            model_data,
            metric='error_rate',
            title="Test: Model Trajectory",
            output_name="test_trajectory.png"
        )
        assert (visualizer.output_dir / "test_trajectory.png").exists()
        print("[OK] plot_model_trajectories works")

        print("\n[PASS] UnifiedVisualizer test passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] UnifiedVisualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_analyzer():
    """Test full UnifiedAnalyzer workflow."""
    print("\n" + "="*80)
    print("TEST: UnifiedAnalyzer (Full Workflow)")
    print("="*80)

    try:
        from scripts.analyze_results import UnifiedAnalyzer

        analyzer = UnifiedAnalyzer(output_dir="test_data/unified")

        # Use mock data from test_base_analyzer
        test_dir = Path("test_data/mock_results")
        test_set_path = test_dir / "test_set.json"

        # Test error pattern analysis (limited output)
        print("\nTesting error pattern analysis...")
        analyzer.load_test_cases(test_set_path)
        analyzer.experiments = analyzer.load_all_experiments(test_dir)

        all_errors = []
        for (exp_num, model_name), exp_data in analyzer.experiments.items():
            errors = analyzer.extract_errors_from_experiment(
                exp_num, model_name, exp_data['responses']
            )
            all_errors.extend(errors)

        assert len(all_errors) >= 2, "Should have errors from 2 models x 2 experiments"
        print(f"[OK] Extracted {len(all_errors)} errors")

        # Test stats computation
        stats = analyzer.compute_summary_stats(all_errors)
        assert stats['total_errors'] >= 2, "Should have at least 2 errors"
        print("[OK] Computed summary stats")

        # Test CSV export
        analyzer._save_error_csv(all_errors, "test_errors.csv")
        assert (analyzer.output_dir / "test_errors.csv").exists()
        print("[OK] Exported errors to CSV")

        print("\n[PASS] UnifiedAnalyzer test passed!")
        return True

    except Exception as e:
        print(f"\n[FAIL] UnifiedAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("UNIFIED ANALYSIS PIPELINE TESTS")
    print("="*80)

    results = {
        'base_analyzer': test_base_analyzer(),
        'visualizer': test_visualizer(),
        'unified_analyzer': test_unified_analyzer()
    }

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {test_name}: {status}")

    print("\n" + "="*80)

    if passed == total:
        print("All tests passed! The unified analysis pipeline is ready to use.")
        print("\nUsage:")
        print("  python scripts/analyze_results.py --mode comprehensive")
        print("  python scripts/analyze_results.py --mode error-patterns")
        print("  python scripts/analyze_results.py --mode by-experiment")
        print("  python scripts/analyze_results.py --mode by-model")
        return 0
    else:
        print("Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
