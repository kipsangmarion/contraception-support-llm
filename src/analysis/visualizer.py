"""
Unified Visualizer - Consistent visualization component for all analysis.

Consolidates duplicate visualization code from multiple scripts:
- visualize_experiments.py
- visualize_model_trajectories.py
- generate_model_comparison.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class UnifiedVisualizer:
    """Single visualization component for all analysis."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color palette for consistency
        self.model_colors = {
            'claude-opus-4-5-20251101': '#FF6B6B',
            'o3-2025-04-16': '#4ECDC4',
            'grok-4-1-fast-reasoning': '#95E1D3',
            'gemini-2.5-pro': '#F38181',
            'gemini-3-pro-preview': '#AA96DA'
        }

        self.experiment_colors = {
            1: '#3498db',  # Blue
            2: '#2ecc71',  # Green
            3: '#e74c3c',  # Red
            4: '#f39c12'   # Orange
        }

    def plot_error_distribution(
        self,
        errors: List[Dict],
        group_by: str = 'model',
        title: str = "Error Distribution",
        output_name: str = "error_distribution.png"
    ):
        """
        Plot error distribution grouped by model, experiment, or error type.

        Args:
            errors: List of error dictionaries
            group_by: Group by 'model', 'experiment', or 'error_type'
            title: Plot title
            output_name: Output filename
        """
        if not errors:
            print(f"No errors to plot for {title}")
            return

        # Count errors by group
        error_counts = defaultdict(int)
        critical_counts = defaultdict(int)

        for error in errors:
            if group_by == 'model':
                key = error['model']
            elif group_by == 'experiment':
                key = f"Exp{error['experiment']}"
            elif group_by == 'error_type':
                for error_type in error['error_types']:
                    error_counts[error_type] += 1
                    if error['has_critical_safety_issue']:
                        critical_counts[error_type] += 1
                continue
            else:
                raise ValueError(f"Invalid group_by: {group_by}")

            error_counts[key] += 1
            if error['has_critical_safety_issue']:
                critical_counts[key] += 1

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))

        groups = sorted(error_counts.keys())
        x = np.arange(len(groups))
        width = 0.35

        total_errors = [error_counts[g] for g in groups]
        critical_errors = [critical_counts[g] for g in groups]

        ax.bar(x - width/2, total_errors, width, label='Total Errors', color='#e74c3c', alpha=0.7)
        ax.bar(x + width/2, critical_errors, width, label='Critical Errors', color='#c0392b')

        ax.set_xlabel(group_by.replace('_', ' ').title())
        ax.set_ylabel('Error Count')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir / output_name}")

    def plot_model_trajectories(
        self,
        model_data: Dict,
        metric: str = 'error_rate',
        title: str = "Model Performance Across Experiments",
        output_name: str = "model_trajectories.png"
    ):
        """
        Plot model performance trajectories across experiments.

        Args:
            model_data: Dictionary mapping model names to trajectory data
                       Each value should have 'experiments' and metric keys
            metric: Metric to plot ('error_rate', 'error_counts', 'critical_counts')
            title: Plot title
            output_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        for model_name, data in model_data.items():
            if metric not in data:
                continue

            experiments = data['experiments']
            values = data[metric]

            # Get color
            color = self.model_colors.get(model_name, '#95a5a6')

            # Plot line
            ax.plot(experiments, values, marker='o', linewidth=2, markersize=8,
                   label=self._format_model_name(model_name), color=color)

        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['Exp1\n(Baseline)', 'Exp2\n(Compliance)', 'Exp3\n(RAG)', 'Exp4\n(Safety)'])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir / output_name}")

    def plot_error_heatmap(
        self,
        disagreements: List[Dict],
        title: str = "Model Disagreement Heatmap",
        output_name: str = "disagreement_heatmap.png"
    ):
        """
        Plot heatmap of model disagreements.

        Args:
            disagreements: List of disagreement instances
            title: Plot title
            output_name: Output filename
        """
        if not disagreements:
            print(f"No disagreements to plot for {title}")
            return

        # Build disagreement matrix
        models = sorted(set(d['model'] for d in disagreements))
        model_pairs = defaultdict(int)

        for disagreement in disagreements:
            model = disagreement['model']
            for other_model in models:
                if other_model != model:
                    pair = tuple(sorted([model, other_model]))
                    model_pairs[pair] += 1

        # Create matrix
        n = len(models)
        matrix = np.zeros((n, n))

        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    pair = tuple(sorted([model1, model2]))
                    matrix[i][j] = model_pairs[pair]

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Reds',
                   xticklabels=[self._format_model_name(m) for m in models],
                   yticklabels=[self._format_model_name(m) for m in models],
                   ax=ax, cbar_kws={'label': 'Disagreement Count'})

        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir / output_name}")

    def plot_experiment_comparison(
        self,
        experiment_stats: Dict,
        title: str = "Experiment Comparison",
        output_name: str = "experiment_comparison.png"
    ):
        """
        Plot comparison across experiments.

        Args:
            experiment_stats: Dictionary mapping experiment numbers to stats
                             Each value should have 'error_rate', 'critical_rate', etc.
            title: Plot title
            output_name: Output filename
        """
        experiments = sorted(experiment_stats.keys())
        x = np.arange(len(experiments))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        # Extract metrics
        error_rates = [experiment_stats[exp].get('error_rate', 0) for exp in experiments]
        critical_rates = [experiment_stats[exp].get('critical_rate', 0) for exp in experiments]
        compliance_rates = [100 - experiment_stats[exp].get('error_rate', 0) for exp in experiments]

        # Plot bars
        ax.bar(x - width, error_rates, width, label='Error Rate (%)', color='#e74c3c', alpha=0.7)
        ax.bar(x, critical_rates, width, label='Critical Rate (%)', color='#c0392b')
        ax.bar(x + width, compliance_rates, width, label='Compliance Rate (%)', color='#2ecc71', alpha=0.7)

        ax.set_xlabel('Experiment')
        ax.set_ylabel('Rate (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Exp{e}" for e in experiments])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir / output_name}")

    def plot_category_breakdown(
        self,
        errors: List[Dict],
        title: str = "Errors by Category",
        output_name: str = "category_breakdown.png"
    ):
        """
        Plot error distribution by category.

        Args:
            errors: List of error dictionaries
            title: Plot title
            output_name: Output filename
        """
        from collections import Counter

        if not errors:
            print(f"No errors to plot for {title}")
            return

        # Count by category
        category_counts = Counter(e['category'] for e in errors if e['category'])

        # Sort by count
        categories = [cat for cat, _ in category_counts.most_common()]
        counts = [category_counts[cat] for cat in categories]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        ax.barh(categories, counts, color=colors)

        ax.set_xlabel('Error Count')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir / output_name}")

    def plot_compliance_comparison(
        self,
        experiment_stats: Dict,
        title: str = "Compliance Rate Comparison",
        output_name: str = "compliance_comparison.png"
    ):
        """
        Plot compliance rate comparison across experiments.

        Args:
            experiment_stats: Dict mapping exp numbers to stats with 'fully_compliant_pct'
            title: Plot title
            output_name: Output filename
        """
        experiments = sorted(experiment_stats.keys())
        exp_labels = [f"Exp{e}\n{self._get_exp_name(e)}" for e in experiments]
        compliance_rates = [experiment_stats[e].get('fully_compliant_pct', 0) for e in experiments]

        colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(exp_labels, compliance_rates, color=colors[:len(experiments)],
                     edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Highlight best approach
        if compliance_rates:
            best_idx = compliance_rates.index(max(compliance_rates))
            bars[best_idx].set_edgecolor('green')
            bars[best_idx].set_linewidth(3)

        ax.set_ylabel('Compliance Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir / output_name}")

    def plot_critical_issues_comparison(
        self,
        experiment_stats: Dict,
        title: str = "Critical Safety Issues Comparison",
        output_name: str = "critical_issues_comparison.png"
    ):
        """
        Plot critical safety issues comparison across experiments.

        Args:
            experiment_stats: Dict mapping exp numbers to stats with 'critical_issues'
            title: Plot title
            output_name: Output filename
        """
        experiments = sorted(experiment_stats.keys())
        exp_labels = [f"Exp{e}\n{self._get_exp_name(e)}" for e in experiments]
        critical_issues = [experiment_stats[e].get('critical_issues', 0) for e in experiments]

        # Red for issues, green for no issues
        colors = ['red' if issues > 0 else 'green' for issues in critical_issues]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(exp_labels, critical_issues, color=colors,
                     edgecolor='black', linewidth=1.5, alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Number of Critical Safety Issues', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        max_issues = max(critical_issues) if critical_issues else 10
        ax.set_ylim(0, max_issues * 1.2 if max_issues > 0 else 10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir / output_name}")

    def plot_score_comparison(
        self,
        experiment_stats: Dict,
        title: str = "Average Compliance Score Comparison",
        output_name: str = "score_comparison.png"
    ):
        """
        Plot average compliance score comparison across experiments.

        Args:
            experiment_stats: Dict mapping exp numbers to stats with 'avg_score'
            title: Plot title
            output_name: Output filename
        """
        experiments = sorted(experiment_stats.keys())
        exp_labels = [f"Exp{e}\n{self._get_exp_name(e)}" for e in experiments]
        scores = [experiment_stats[e].get('avg_score', 0) for e in experiments]

        colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(exp_labels, scores, color=colors[:len(experiments)],
                     edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Highlight best approach
        if scores:
            best_idx = scores.index(max(scores))
            bars[best_idx].set_edgecolor('green')
            bars[best_idx].set_linewidth(3)

        ax.set_ylabel('Average Compliance Score (out of 2.0)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 2.0)
        ax.grid(axis='y', alpha=0.3)

        # Add horizontal line at 2.0 (perfect score)
        ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Perfect Score')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir / output_name}")

    def _get_exp_name(self, exp_num: int) -> str:
        """Get experiment name from number."""
        exp_names = {
            1: "Baseline",
            2: "Compliance",
            3: "RAG",
            4: "Safety"
        }
        return exp_names.get(exp_num, f"Exp{exp_num}")

    def _format_model_name(self, model_name: str) -> str:
        """Format model name for display."""
        if 'claude-opus' in model_name:
            return 'Claude Opus 4.5'
        elif 'o3-2025' in model_name:
            return 'OpenAI o3'
        elif 'grok-4-1' in model_name:
            return 'Grok 4.1'
        elif 'gemini-2.5' in model_name:
            return 'Gemini 2.5 Pro'
        elif 'gemini-3' in model_name:
            return 'Gemini 3 Pro'
        else:
            return model_name
