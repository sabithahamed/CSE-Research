# CSE Research Dataset and Analysis

This repository contains data preparation and econometric analysis notebooks for a Colombo Stock Exchange (CSE) research project.

## Repository Structure

- `Raw Data/` - source CSV files used to build analysis tables
- `master_dataset.csv` - merged and cleaned working dataset
- `sector_liquidity_report.csv` - zero-return ratio output for liquidity checks
- `combine_data.ipynb` - combines source files into a master dataset
- `liquidity_check.ipynb` - computes illiquidity using zero-return ratio (ZRR)
- `Analysis/` - exploratory and econometric analysis notebooks

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
