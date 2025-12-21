# Statistical Significance Analysis Report
## Summary of Findings
**Total Comparisons:** 12
**Statistically Significant Differences:** 7/12

## Key Findings

### Baseline vs Compliance-Aware

**Model: grok-4-1-fast-reasoning**

- **Statistical Significance:** NO (p=0.1530)
- **Effect Size:** small (Cohen's h=0.208)
- **Success Rate Change:** -10.0%
- **95% CI:** [-24.8%, 4.8%]
- **Sample Size:** 80 test cases

**Interpretation:** This difference is not statistically significant.

**Model: claude-opus-4-5-20251101**

- **Statistical Significance:** NO (p=0.1356)
- **Effect Size:** small (Cohen's h=-0.222)
- **Success Rate Change:** +10.0%
- **95% CI:** [-3.9%, 23.9%]
- **Sample Size:** 80 test cases

**Interpretation:** This difference is not statistically significant.

**Model: o3-2025-04-16**

- **Statistical Significance:** NO (p=0.8312)
- **Effect Size:** negligible (Cohen's h=-0.055)
- **Success Rate Change:** +2.5%
- **95% CI:** [-11.7%, 16.7%]
- **Sample Size:** 80 test cases

**Interpretation:** This difference is not statistically significant.

### Baseline vs RAG

**Model: grok-4-1-fast-reasoning**

- **Statistical Significance:** YES (p=0.0000)
- **Effect Size:** large (Cohen's h=1.638)
- **Success Rate Change:** -66.2%
- **95% CI:** [-77.0%, -55.5%]
- **Sample Size:** 80 test cases

**Interpretation:** This difference is both statistically significant AND practically meaningful.

**Model: claude-opus-4-5-20251101**

- **Statistical Significance:** YES (p=0.0000)
- **Effect Size:** large (Cohen's h=1.006)
- **Success Rate Change:** -47.5%
- **95% CI:** [-60.9%, -34.1%]
- **Sample Size:** 80 test cases

**Interpretation:** This difference is both statistically significant AND practically meaningful.

**Model: o3-2025-04-16**

- **Statistical Significance:** YES (p=0.0000)
- **Effect Size:** large (Cohen's h=0.967)
- **Success Rate Change:** -46.2%
- **95% CI:** [-59.9%, -32.6%]
- **Sample Size:** 80 test cases

**Interpretation:** This difference is both statistically significant AND practically meaningful.

### Compliance-Aware vs RAG

**Model: grok-4-1-fast-reasoning**

- **Statistical Significance:** YES (p=0.0000)
- **Effect Size:** large (Cohen's h=1.429)
- **Success Rate Change:** -56.2%
- **95% CI:** [-67.6%, -44.9%]
- **Sample Size:** 80 test cases

**Interpretation:** This difference is both statistically significant AND practically meaningful.

**Model: claude-opus-4-5-20251101**

- **Statistical Significance:** YES (p=0.0000)
- **Effect Size:** large (Cohen's h=1.228)
- **Success Rate Change:** -57.5%
- **95% CI:** [-70.2%, -44.8%]
- **Sample Size:** 80 test cases

**Interpretation:** This difference is both statistically significant AND practically meaningful.

**Model: o3-2025-04-16**

- **Statistical Significance:** YES (p=0.0000)
- **Effect Size:** large (Cohen's h=1.021)
- **Success Rate Change:** -48.8%
- **95% CI:** [-62.2%, -35.3%]
- **Sample Size:** 80 test cases

**Interpretation:** This difference is both statistically significant AND practically meaningful.

### Baseline vs Safety Validation (Claude only)

**Model: claude-opus-4-5-20251101**

- **Statistical Significance:** NO (p=1.0000)
- **Effect Size:** negligible (Cohen's h=-0.027)
- **Success Rate Change:** +1.3%
- **95% CI:** [-13.3%, 15.8%]
- **Sample Size:** 80 test cases

**Interpretation:** This difference is not statistically significant.

### Compliance-Aware vs Safety Validation (Claude only)

**Model: claude-opus-4-5-20251101**

- **Statistical Significance:** NO (p=0.1456)
- **Effect Size:** negligible (Cohen's h=0.195)
- **Success Rate Change:** -8.7%
- **95% CI:** [-22.6%, 5.1%]
- **Sample Size:** 80 test cases

**Interpretation:** This difference is not statistically significant.

### RAG vs Safety Validation (Claude only)

**Model: claude-opus-4-5-20251101**

- **Statistical Significance:** YES (p=0.0000)
- **Effect Size:** large (Cohen's h=-1.033)
- **Success Rate Change:** +48.8%
- **95% CI:** [35.4%, 62.1%]
- **Sample Size:** 80 test cases

**Interpretation:** This difference is both statistically significant AND practically meaningful.

