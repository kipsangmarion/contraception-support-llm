# Experiment Results Visualizations

**Generated**: December 2, 2025
**Source**: Contraception Support LLM Experiment Results
**Model**: llama3.2 (Ollama)

---

## Overview

This directory contains publication-ready visualizations and tables generated from the completed experiment results. All charts are saved at 300 DPI for thesis/publication quality.

---

## Generated Files

### Charts (PNG, 300 DPI)

1. **plot1_rag_comparison.png**
   - Bar chart comparing Baseline vs RAG performance
   - Shows unexpected 7.6% F1 score decrease with RAG
   - Statistical significance: p < 0.001 ***
   - Key finding: RAG degraded performance

2. **plot3_rl_cumulative_reward.png**
   - Line chart showing cumulative rewards over 1,000 rounds
   - Three strategies: LinUCB (adaptive), Fixed, Random
   - Shows LinUCB achieving 18.9% improvement over fixed
   - 50.9% improvement over random baseline (p < 0.001)

3. **plot4_cross_experiment_f1.png**
   - Bar chart comparing F1 scores across experiments
   - Baseline: 79.3%, Anchored: 78.7%, RAG: 73.2%
   - Shows consistency of baseline performance
   - Highlights RAG degradation issue

4. **plot5_latency_comparison.png**
   - Bar chart comparing response times
   - Baseline: 10.1s, Anchored: 8.5s, RAG: 39.0s
   - Shows 15.8% speedup with anchored prompts
   - RAG adds 29% latency overhead

5. **plot6_safety_fallback.png**
   - Pie chart showing safety fallback rate
   - 70% of questions triggered "insufficient information" response
   - Indicates overly conservative prompt design
   - 30% answered directly

### Tables (CSV)

1. **table1_experiment_overview.csv**
   - Summary of all 5 completed experiments
   - Test cases, success rates, key metrics, timestamps
   - Total: 1,541 test cases across 6.5 hours runtime

2. **table2_performance_summary.csv**
   - Detailed performance metrics for Experiments 1-3
   - BERTScore precision, recall, F1
   - Latency comparison

3. **table3_statistical_tests.csv**
   - Statistical significance tests
   - RAG vs Baseline, LinUCB comparisons
   - t-statistics, p-values, significance flags

---

## Quick Results Summary

### Experiment 1: Baseline Knowledge
- **100 questions** | **F1: 79.3%** | **Latency: 10.1s**
- Strong intrinsic knowledge without RAG
- High recall (86.8%), moderate precision (73.1%)

### Experiment 2: Anchored Prompts
- **100 questions** | **F1: 78.7%** | **Latency: 8.5s** | **Safety Fallback: 70%**
- Maintains quality with stricter prompts
- 15.8% faster than baseline
- High fallback rate indicates over-conservatism

### Experiment 3: RAG Comparison
- **100 questions** | **Baseline F1: 79.2%** | **RAG F1: 73.2%**
- RAG decreased F1 by 7.6% (p < 0.001) - unexpected finding
- Zero citation rate indicates potential pipeline issue
- RAG adds 8.8s latency (29% increase)

### Experiment 4A: Long-Session Memory
- **10 conversations** | **331 total turns** | **Recall: 38%**
- High contradiction rate (45.3%)
- Poor long-context memory retention
- Motivates explicit memory mechanism

### Experiment 4B: Multi-Session Memory
- **Incomplete** - Summarized memory results missing
- Full memory: 18.3% recall accuracy
- No memory: 2.5% recall accuracy
- 533% improvement with memory (p = 0.046)

### Experiment 5: Adherence RL
- **1,000 rounds** | **LinUCB: 0.605** | **Fixed: 0.509** | **Random: 0.401**
- LinUCB +18.9% vs Fixed (p = 0.111, trending)
- LinUCB +50.9% vs Random (p < 0.001, significant)
- Adaptive learning shows practical benefit

---

## Key Findings

1. **RAG Paradox**: RAG unexpectedly decreased performance by 7.6%
   - Requires investigation: retrieval quality, chunk relevance, context contamination
   - Model's intrinsic knowledge may be sufficient for contraception domain

2. **Memory is Critical**: 533% improvement with full memory across sessions
   - Essential for multi-session counseling scenarios
   - Stateless LLMs inadequate without explicit memory

3. **Prompt Engineering**: Minimal quality loss (-0.6% F1) with anchored prompts
   - 70% safety fallback rate too conservative
   - Need balanced prompts for safety + informativeness

4. **Adaptive Learning**: LinUCB shows 18.9% improvement trend
   - Personalization improves adherence outcomes
   - Larger sample needed for statistical significance

---

## Visualization Usage

### For Thesis/Papers
- All PNG files are 300 DPI, suitable for publication
- Figures use serif fonts and clear labeling
- Statistical significance annotated with p-values

### For Presentations
- Charts use high-contrast colors (colorblind-friendly)
- Large font sizes (12-14pt) readable from distance
- Key metrics annotated directly on charts

### For Reports
- CSV tables easily importable to LaTeX, Word, Google Docs
- Formatted for readability in spreadsheet software
- Can be regenerated with updated data

---

## Regenerating Visualizations

To regenerate all visualizations with updated data:

```bash
python visualize_results.py
```

This will:
1. Load all experiment results from `results/tables/`
2. Generate 5 PNG charts at 300 DPI
3. Generate 3 CSV summary tables
4. Save all files to `results/visualizations/`

Requirements:
- Python 3.7+
- pandas, matplotlib, seaborn, numpy

Install dependencies:
```bash
pip install pandas matplotlib seaborn numpy
```

---

## Notes

- **Incomplete Data**: Experiment 4B (Multi-Session Memory) missing summarized memory results
  - Re-run needed after fixing JSON serialization issue
  - Plot 2 (Memory Strategies) will generate once data is complete

- **Citation Rate**: Experiment 3 shows 0% citation rate
  - Indicates RAG pipeline not recording sources
  - Need to investigate RAGGenerator citation tracking

- **Statistical Power**: Some tests underpowered
  - Experiment 4B: Only 10 scenarios
  - Experiment 5: LinUCB vs Fixed trending but not significant (p = 0.111)
  - Consider larger sample sizes for final thesis

---

## Contact

For questions about these visualizations or to request additional charts, please refer to the main experiment documentation in `docs/EXPERIMENT_RESULTS_ANALYSIS.md`.
