"""
Visualize Experiment Results

Creates comparison charts for all experiments:
- Compliance rates
- Critical issues
- Average scores
- Latency comparison
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_analysis_results(results_dir: Path):
    """Load the latest comprehensive analysis results."""
    # Find the latest analysis file
    analysis_files = list(results_dir.glob("comprehensive_analysis_*.json"))
    if not analysis_files:
        print("ERROR: No analysis results found. Run analyze_all_experiments.py first.")
        sys.exit(1)

    # Sort by timestamp in filename and get latest
    latest_file = sorted(analysis_files)[-1]

    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_compliance_comparison(data: dict, output_dir: Path):
    """Create bar chart comparing compliance rates."""
    experiments = ['Exp1\nBaseline', 'Exp2\nCompliance-\nAware', 'Exp3\nRAG', 'Exp4\nSafety\nValidation']
    compliance_rates = [
        data['exp1_baseline']['compliance_pct'],
        data['exp2_compliance_aware']['compliance_pct'],
        data['exp3_rag']['compliance_pct'],
        data['exp4_safety_validation']['compliance_pct']
    ]

    colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(experiments, compliance_rates, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight best approach
    best_idx = compliance_rates.index(max(compliance_rates))
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(3)

    ax.set_ylabel('Compliance Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Compliance Rate Comparison Across Experiments', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'compliance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def create_critical_issues_comparison(data: dict, output_dir: Path):
    """Create bar chart comparing critical safety issues."""
    experiments = ['Exp1\nBaseline', 'Exp2\nCompliance-\nAware', 'Exp3\nRAG', 'Exp4\nSafety\nValidation']
    critical_issues = [
        data['exp1_baseline']['critical_issues'],
        data['exp2_compliance_aware']['critical_issues'],
        data['exp3_rag']['critical_issues'],
        data['exp4_safety_validation']['critical_issues']
    ]

    # Red for issues, green for no issues
    colors = ['red' if issues > 0 else 'green' for issues in critical_issues]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(experiments, critical_issues, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Number of Critical Safety Issues', fontsize=12, fontweight='bold')
    ax.set_title('Critical Safety Issues Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(critical_issues) * 1.2 if max(critical_issues) > 0 else 10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'critical_issues_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def create_score_comparison(data: dict, output_dir: Path):
    """Create bar chart comparing average compliance scores."""
    experiments = ['Exp1\nBaseline', 'Exp2\nCompliance-\nAware', 'Exp3\nRAG', 'Exp4\nSafety\nValidation']
    scores = [
        data['exp1_baseline']['avg_score'],
        data['exp2_compliance_aware']['avg_score'],
        data['exp3_rag']['avg_score'],
        data['exp4_safety_validation']['avg_score']
    ]

    colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(experiments, scores, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight best approach
    best_idx = scores.index(max(scores))
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(3)

    ax.set_ylabel('Average Compliance Score (out of 2.0)', fontsize=12, fontweight='bold')
    ax.set_title('Average Compliance Score Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 2.0)
    ax.grid(axis='y', alpha=0.3)

    # Add horizontal line at 2.0 (perfect score)
    ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Perfect Score')
    ax.legend()

    plt.tight_layout()
    output_path = output_dir / 'score_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def create_latency_comparison(data: dict, output_dir: Path):
    """Create bar chart comparing latency."""
    experiments = ['Exp1\nBaseline', 'Exp2\nCompliance-\nAware', 'Exp3\nRAG', 'Exp4\nSafety\nValidation']
    latencies = [
        data['exp1_baseline']['avg_latency_s'],
        data['exp2_compliance_aware']['avg_latency_s'],
        data['exp3_rag']['avg_latency_s'],
        data['exp4_safety_validation']['avg_latency_s']
    ]

    colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(experiments, latencies, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Highlight fastest approach
    fastest_idx = latencies.index(min(latencies))
    bars[fastest_idx].set_edgecolor('blue')
    bars[fastest_idx].set_linewidth(3)

    ax.set_ylabel('Average Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Response Latency Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(latencies) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'latency_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def create_comprehensive_dashboard(data: dict, output_dir: Path):
    """Create a comprehensive dashboard with all metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    experiments = ['Exp1', 'Exp2', 'Exp3', 'Exp4']
    colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']

    # 1. Compliance Rate
    compliance_rates = [
        data['exp1_baseline']['compliance_pct'],
        data['exp2_compliance_aware']['compliance_pct'],
        data['exp3_rag']['compliance_pct'],
        data['exp4_safety_validation']['compliance_pct']
    ]
    bars1 = ax1.bar(experiments, compliance_rates, color=colors, edgecolor='black')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax1.set_ylabel('Compliance Rate (%)', fontweight='bold')
    ax1.set_title('Compliance Rate', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Critical Issues
    critical_issues = [
        data['exp1_baseline']['critical_issues'],
        data['exp2_compliance_aware']['critical_issues'],
        data['exp3_rag']['critical_issues'],
        data['exp4_safety_validation']['critical_issues']
    ]
    issue_colors = ['red' if issues > 0 else 'green' for issues in critical_issues]
    bars2 = ax2.bar(experiments, critical_issues, color=issue_colors, edgecolor='black', alpha=0.7)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    ax2.set_ylabel('Critical Issues', fontweight='bold')
    ax2.set_title('Critical Safety Issues', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(critical_issues) * 1.2 if max(critical_issues) > 0 else 10)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Average Score
    scores = [
        data['exp1_baseline']['avg_score'],
        data['exp2_compliance_aware']['avg_score'],
        data['exp3_rag']['avg_score'],
        data['exp4_safety_validation']['avg_score']
    ]
    bars3 = ax3.bar(experiments, scores, color=colors, edgecolor='black')
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    ax3.set_ylabel('Avg Score (out of 2.0)', fontweight='bold')
    ax3.set_title('Average Compliance Score', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 2.0)
    ax3.axhline(y=2.0, color='green', linestyle='--', alpha=0.5)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Latency
    latencies = [
        data['exp1_baseline']['avg_latency_s'],
        data['exp2_compliance_aware']['avg_latency_s'],
        data['exp3_rag']['avg_latency_s'],
        data['exp4_safety_validation']['avg_latency_s']
    ]
    bars4 = ax4.bar(experiments, latencies, color=colors, edgecolor='black')
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')
    ax4.set_ylabel('Latency (seconds)', fontweight='bold')
    ax4.set_title('Response Latency', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(latencies) * 1.2)
    ax4.grid(axis='y', alpha=0.3)

    # Overall title
    fig.suptitle('Comprehensive Experiment Comparison Dashboard',
                 fontsize=16, fontweight='bold', y=0.995)

    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc='#ff9999', edgecolor='black', label='Exp1: Baseline'),
        plt.Rectangle((0,0),1,1, fc='#66b3ff', edgecolor='black', label='Exp2: Compliance-Aware'),
        plt.Rectangle((0,0),1,1, fc='#ffcc99', edgecolor='black', label='Exp3: RAG'),
        plt.Rectangle((0,0),1,1, fc='#99ff99', edgecolor='black', label='Exp4: Safety Validation')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'comprehensive_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def create_improvement_chart(data: dict, output_dir: Path):
    """Create chart showing improvements vs baseline."""
    metrics = ['Compliance\nRate', 'Score', 'Critical\nIssues\n(inverted)']

    # Calculate improvements vs Exp1
    exp2_improvements = [
        data['exp2_compliance_aware']['improvement_vs_exp1']['compliance_delta'],
        data['exp2_compliance_aware']['improvement_vs_exp1']['score_delta'] * 50,  # Scale to %
        data['exp2_compliance_aware']['improvement_vs_exp1']['critical_issues_delta'] * 10  # Scale
    ]

    exp3_changes = [
        data['exp3_rag']['degradation_vs_exp2']['compliance_delta'],
        (data['exp3_rag']['avg_score'] - data['exp1_baseline']['avg_score']) * 50,
        (data['exp1_baseline']['critical_issues'] - data['exp3_rag']['critical_issues']) * 10
    ]

    exp4_changes = [
        data['exp4_safety_validation']['compliance_pct'] - data['exp1_baseline']['compliance_pct'],
        (data['exp4_safety_validation']['avg_score'] - data['exp1_baseline']['avg_score']) * 50,
        (data['exp1_baseline']['critical_issues'] - data['exp4_safety_validation']['critical_issues']) * 10
    ]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, exp2_improvements, width, label='Exp2 vs Exp1', color='#66b3ff', edgecolor='black')
    bars2 = ax.bar(x, exp3_changes, width, label='Exp3 vs Exp1', color='#ffcc99', edgecolor='black')
    bars3 = ax.bar(x + width, exp4_changes, width, label='Exp4 vs Exp1', color='#99ff99', edgecolor='black')

    ax.set_ylabel('Improvement (%)', fontweight='bold')
    ax.set_title('Performance Improvements vs Baseline (Exp1)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'improvement_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def main():
    """Main visualization function."""
    print("="*80)
    print("EXPERIMENT RESULTS VISUALIZATION")
    print("="*80)

    results_dir = Path("results/compliance_experiments")
    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading analysis results...")
    data = load_analysis_results(results_dir)

    print("\nGenerating visualizations...")

    # Create individual charts
    create_compliance_comparison(data, output_dir)
    create_critical_issues_comparison(data, output_dir)
    create_score_comparison(data, output_dir)
    create_latency_comparison(data, output_dir)

    # Create comprehensive dashboard
    create_comprehensive_dashboard(data, output_dir)

    # Create improvement chart
    create_improvement_chart(data, output_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll charts saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - compliance_comparison.png")
    print("  - critical_issues_comparison.png")
    print("  - score_comparison.png")
    print("  - latency_comparison.png")
    print("  - comprehensive_dashboard.png")
    print("  - improvement_comparison.png")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
