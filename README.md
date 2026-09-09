# CSE Research Dataset and Analysis

This repository contains data preparation and econometric analysis notebooks for a Colombo Stock Exchange (CSE) research project.

## Repository Structure

- `Raw Data/` - source CSV files used to build analysis tables
- `master_dataset.csv` - merged and cleaned working dataset
- `sector_liquidity_report.csv` - zero-return ratio output for liquidity checks
- `combine_data.ipynb` - combines source files into a master dataset
- `liquidity_check.ipynb` - computes illiquidity using zero-return ratio (ZRR)
- `Analysis/` - exploratory and econometric analysis notebooks

## Analysis Directories

| Directory | Produces |
| --- | --- |
| `Analysis/EDA/` | descriptive statistics, normality tests, correlation heatmaps |
| `Analysis/Structural_Breaks/` | regime detection: Bai-Perron dynamic programming with BIC selection, sup-F test with simulated critical values, two-state Markov-switching model |
| `Analysis/VECM/` | Johansen cointegration test, VECM estimation, error-correction terms, residual diagnostics |
| `Analysis/IRF_VD/` | impulse response functions and forecast-error variance decomposition |
| `Analysis/EGARCH/` | EGARCH(1,1) with Student-t errors, fitted per series per regime |
| `Analysis/Contagion/` | Forbes-Rigobon heteroskedasticity-adjusted correlations, defensive vs cyclical sector decomposition |

### Regime definitions

Regimes are estimated from the ASPI return series, not assigned from the policy
calendar. `Analysis/Structural_Breaks/break_tests.py` produces `detected_breaks.csv`,
which gives breaks on 2020-02-11, 2022-01-25 and 2022-12-06. Every downstream
analysis uses those dates:

| Regime | Window |
| --- | --- |
| Pre-COVID | start to 2020-02-10 |
| COVID | 2020-02-11 to 2022-01-24 |
| Default | 2022-01-25 to 2022-12-05 |
| Recovery | 2022-12-06 to end |

### Alternative-specification files

Files ending in `_backup` are not stale outputs. They hold results under
segmentations other than the detected one, and support the robustness comparison
reported alongside the main results:

- `Analysis/EGARCH/egarch_policy_dates_backup.csv` - EGARCH under the policy-calendar
  segmentation (default dated 2022-03-01 to 2023-02-28)
- `Analysis/EGARCH/egarch_detected3_backup.csv` - EGARCH under a three-regime split,
  before the COVID period was separated from the pre-default window
- `Analysis/Contagion/fr_policy_backup.csv`, `Analysis/Contagion/defensive_policy_backup.csv` -
  the corresponding contagion and sector-decomposition results under policy dates

### Scripts

Directories containing `.py` files are run directly rather than as notebooks, from
within their own directory:

```bash
cd Analysis/Structural_Breaks && python break_tests.py
cd Analysis/Contagion && python contagion_tests.py
cd Analysis/Contagion && python defensive_sector_tests.py
cd Analysis/VECM && python regime_vecm.py
```

`Analysis/Structural_Breaks/supf_critical_values.py` simulates the null distribution
of the sup-F statistic (5,000 replications) rather than relying on published tables.
`Analysis/VECM/regime_vecm.py` re-estimates the cointegrating relation within each
regime and reports whether each regime supports one at all.

## Main Workflows

1. Build master dataset
   - Open `combine_data.ipynb`
   - Run all cells to produce `master_dataset.csv`

2. Run liquidity diagnostics
   - Open `liquidity_check.ipynb`
   - Run all cells to generate `sector_liquidity_report.csv`

3. Run analysis notebooks
   - Open notebooks under `Analysis/`
   - Execute cells in order for EDA, correlation, and EGARCH outputs

## Python Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Notes

- Notebook relative paths assume execution from each notebook's directory.
- Column names are generated from file stems with spaces converted to `_` and `&` converted to `and`.
- Some names may include punctuation from original source files (for example, `Food,_Beverage_and_Tobacco`).
