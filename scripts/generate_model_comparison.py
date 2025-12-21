"""
Generate model comparison overview visualization for Claude, o3, and Grok.
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Load data from summary files
results_dir = Path("results/compliance_experiments")

# Experiment 1: Baseline
exp1_claude = json.load(open(results_dir / "exp1_claude-opus-4-5-20251101_20251208_050610_auto_summary.json"))
exp1_o3 = json.load(open(results_dir / "exp1_o3-2025-04-16_20251208_110608_auto_summary.json"))
exp1_grok = json.load(open(results_dir / "exp1_grok-4-1-fast-reasoning_20251208_053400_auto_summary.json"))

# Experiment 2: Compliance-Aware
exp2_claude = json.load(open(results_dir / "exp2_claude-opus-4-5-20251101_20251208_214528_auto_summary.json"))
exp2_o3 = json.load(open(results_dir / "exp2_o3-2025-04-16_20251208_212508_auto_summary.json"))
exp2_grok = json.load(open(results_dir / "exp2_grok-4-1-fast-reasoning_20251208_220605_auto_summary.json"))

# Experiment 3: RAG - need to calculate from annotated files
exp3_claude_file = results_dir / "exp3_claude-opus-4-5-20251101_rag_20251209_010156_auto_annotated.json"
exp3_o3_file = results_dir / "exp3_o3-2025-04-16_rag_20251209_004927_auto_annotated.json"
exp3_grok_file = results_dir / "exp3_grok-4-1-fast-reasoning_rag_20251209_010840_auto_annotated.json"

def calculate_exp3_stats(filepath):
    """Calculate stats from annotated file."""
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)

    responses = data['responses']
    total = len(responses)
    fully_compliant = sum(1 for r in responses if r['annotation']['compliance_score'] == 2)
    partially_compliant = sum(1 for r in responses if r['annotation']['compliance_score'] == 1)
    non_compliant = sum(1 for r in responses if r['annotation']['compliance_score'] == 0)
    critical_issues = sum(1 for r in responses if r['annotation'].get('has_critical_safety_issue', False))

    avg_score = sum(r['annotation']['compliance_score'] for r in responses) / total
    avg_latency = sum(r.get('latency_seconds', 0) for r in responses) / total

    return {
        'fully_compliant': fully_compliant,
        'partially_compliant': partially_compliant,
        'non_compliant': non_compliant,
        'critical_issues': critical_issues,
        'avg_compliance_score': avg_score,
        'avg_latency': avg_latency,
        'total_responses': total
    }

exp3_claude = calculate_exp3_stats(exp3_claude_file)
exp3_o3 = calculate_exp3_stats(exp3_o3_file)
exp3_grok = calculate_exp3_stats(exp3_grok_file)

# Experiment 4: Safety Validation (All three models)
exp4_claude_file = results_dir / "exp4_claude-opus-4-5-20251101_safety_validation_20251213_090153_auto_annotated.json"
exp4_claude = calculate_exp3_stats(exp4_claude_file)

exp4_o3_file = results_dir / "exp4_o3-2025-04-16_safety_validation_20251213_091630_auto_annotated.json"
exp4_o3 = calculate_exp3_stats(exp4_o3_file)

exp4_grok_file = results_dir / "exp4_grok-4-1-fast-reasoning_safety_validation_20251213_093115_auto_annotated.json"
exp4_grok = calculate_exp3_stats(exp4_grok_file)

# Organize data
experiments = ['Exp 1\n(Baseline)', 'Exp 2\n(Compliance)', 'Exp 3\n(RAG)', 'Exp 4\n(Safety Val)']

# Error rates (1 - avg_score/2)
claude_error_rates = [
    (2 - exp1_claude['avg_compliance_score']) / 2 * 100,
    (2 - exp2_claude['avg_compliance_score']) / 2 * 100,
    (2 - exp3_claude['avg_compliance_score']) / 2 * 100,
    (2 - exp4_claude['avg_compliance_score']) / 2 * 100
]

o3_error_rates = [
    (2 - exp1_o3['avg_compliance_score']) / 2 * 100,
    (2 - exp2_o3['avg_compliance_score']) / 2 * 100,
    (2 - exp3_o3['avg_compliance_score']) / 2 * 100,
    (2 - exp4_o3['avg_compliance_score']) / 2 * 100
]

grok_error_rates = [
    (2 - exp1_grok['avg_compliance_score']) / 2 * 100,
    (2 - exp2_grok['avg_compliance_score']) / 2 * 100,
    (2 - exp3_grok['avg_compliance_score']) / 2 * 100,
    (2 - exp4_grok['avg_compliance_score']) / 2 * 100
]

# Critical issues
claude_critical = [exp1_claude['critical_issues'], exp2_claude['critical_issues'],
                   exp3_claude['critical_issues'], exp4_claude['critical_issues']]
o3_critical = [exp1_o3['critical_issues'], exp2_o3['critical_issues'],
               exp3_o3['critical_issues'], exp4_o3['critical_issues']]
grok_critical = [exp1_grok['critical_issues'], exp2_grok['critical_issues'],
                 exp3_grok['critical_issues'], exp4_grok['critical_issues']]

# Latency
claude_latency = [exp1_claude['avg_latency'], exp2_claude['avg_latency'],
                  exp3_claude['avg_latency'], exp4_claude['avg_latency']]
o3_latency = [exp1_o3['avg_latency'], exp2_o3['avg_latency'],
              exp3_o3['avg_latency'], exp4_o3['avg_latency']]
grok_latency = [exp1_grok['avg_latency'], exp2_grok['avg_latency'],
                exp3_grok['avg_latency'], exp4_grok['avg_latency']]

# Create 4-panel figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Comparison: Claude Opus 4.5, OpenAI o3, Grok 4.1 Fast',
             fontsize=16, fontweight='bold', y=0.995)

# Colors
claude_color = '#1E3A8A'  # Deep blue
o3_color = '#F97316'      # Orange
grok_color = '#10B981'    # Green

# Panel A: Error Rates
ax1 = axes[0, 0]
x = np.arange(len(experiments))
width = 0.25

# Only plot where data exists
claude_bars = ax1.bar(x - width, claude_error_rates, width, label='Claude Opus 4.5', color=claude_color, alpha=0.8)
o3_vals = [v if v is not None else 0 for v in o3_error_rates]
o3_bars = ax1.bar(x, o3_vals, width, label='OpenAI o3', color=o3_color, alpha=0.8)
grok_vals = [v if v is not None else 0 for v in grok_error_rates]
grok_bars = ax1.bar(x + width, grok_vals, width, label='Grok 4.1 Fast', color=grok_color, alpha=0.8)

ax1.set_ylabel('Error Rate (%)', fontweight='bold')
ax1.set_xlabel('Experiment', fontweight='bold')
ax1.set_title('A. Error Rates by Experiment', fontweight='bold', loc='left')
ax1.set_xticks(x)
ax1.set_xticklabels(experiments)
ax1.legend(loc='upper left')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [claude_bars, o3_bars, grok_bars]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)

# Panel B: Critical Safety Violations
ax2 = axes[0, 1]
claude_bars_crit = ax2.bar(x - width, claude_critical, width, label='Claude Opus 4.5',
                           color=claude_color, alpha=0.8)
o3_bars_crit = ax2.bar(x, o3_critical[:3] + [0], width, label='OpenAI o3',
                       color=o3_color, alpha=0.8)
grok_bars_crit = ax2.bar(x + width, grok_critical[:3] + [0], width, label='Grok 4.1 Fast',
                         color=grok_color, alpha=0.8)

ax2.set_ylabel('Critical Safety Violations', fontweight='bold')
ax2.set_xlabel('Experiment', fontweight='bold')
ax2.set_title('B. Critical Safety Violations', fontweight='bold', loc='left')
ax2.set_xticks(x)
ax2.set_xticklabels(experiments)
ax2.legend(loc='upper left')
ax2.set_ylim(0, max(max(claude_critical), max(o3_critical[:3]), max(grok_critical[:3])) + 2)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [claude_bars_crit, o3_bars_crit, grok_bars_crit]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)

# Panel C: Compliance Scores (Bar Chart)
ax3 = axes[1, 0]

# Compliance scores
claude_compliance_scores = [
    exp1_claude['avg_compliance_score'],
    exp2_claude['avg_compliance_score'],
    exp3_claude['avg_compliance_score'],
    exp4_claude['avg_compliance_score']
]

o3_compliance_scores = [
    exp1_o3['avg_compliance_score'],
    exp2_o3['avg_compliance_score'],
    exp3_o3['avg_compliance_score'],
    exp4_o3['avg_compliance_score']
]

grok_compliance_scores = [
    exp1_grok['avg_compliance_score'],
    exp2_grok['avg_compliance_score'],
    exp3_grok['avg_compliance_score'],
    exp4_grok['avg_compliance_score']
]

claude_bars_comp = ax3.bar(x - width, claude_compliance_scores, width,
                           label='Claude Opus 4.5', color=claude_color, alpha=0.8)
o3_bars_comp = ax3.bar(x, o3_compliance_scores, width,
                       label='OpenAI o3', color=o3_color, alpha=0.8)
grok_bars_comp = ax3.bar(x + width, grok_compliance_scores, width,
                         label='Grok 4.1 Fast', color=grok_color, alpha=0.8)

ax3.set_ylabel('Average Compliance Score (0-2)', fontweight='bold')
ax3.set_xlabel('Experiment', fontweight='bold')
ax3.set_title('C. Compliance Score Comparison', fontweight='bold', loc='left')
ax3.set_xticks(x)
ax3.set_xticklabels(experiments)
ax3.legend(loc='upper right')
ax3.set_ylim(0, 2.2)
ax3.axhline(y=2.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax3.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [claude_bars_comp, o3_bars_comp, grok_bars_comp]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)

# Panel D: Average Latency
ax4 = axes[1, 1]
claude_bars_lat = ax4.bar(x - width, claude_latency, width, label='Claude Opus 4.5',
                          color=claude_color, alpha=0.8)
o3_bars_lat = ax4.bar(x, o3_latency, width, label='OpenAI o3',
                      color=o3_color, alpha=0.8)
grok_bars_lat = ax4.bar(x + width, grok_latency, width, label='Grok 4.1 Fast',
                        color=grok_color, alpha=0.8)

ax4.set_ylabel('Average Latency (seconds)', fontweight='bold')
ax4.set_xlabel('Experiment', fontweight='bold')
ax4.set_title('D. Response Latency Comparison', fontweight='bold', loc='left')
ax4.set_xticks(x)
ax4.set_xticklabels(experiments)
ax4.legend(loc='upper left')
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [claude_bars_lat, o3_bars_lat, grok_bars_lat]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s',
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# Save figure
output_dir = Path("results/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "model_comparison_overview.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {output_path}")

plt.show()
