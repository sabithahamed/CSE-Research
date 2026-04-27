## 1. Data Sources

This project uses a high-frequency market-macro panel built for the study titled *Algorithmic Detection of Volatility Regimes and Market Contagion During Sovereign Default*. The final modeling table contains **1,559 daily observations** (up to 2025-08-31) and combines:
- Colombo Stock Exchange (CSE) market series from the **CSE portal** (ASPI plus sector indices), and
- Macroeconomic series from the **Central Bank of Sri Lanka (CBSL) eResearch portal**.

The financial panel used in estimation includes ASPI and the following 13 sectors retained after liquidity diagnostics and consistency checks:
- Banks
- Capital Goods
- Consumer Durables and Apparel
- Consumer Services
- Diversified Financials
- Energy
- Food, Beverage and Tobacco
- Healthcare Equipment and Services
- Insurance
- Materials
- Real Estate Management and Development
- Retailing
- Telecommunication Services

### 1.1 Log-Transformed Macroeconomic Indicators

The integrated macro block (from CBSL eResearch) includes:
- Inflation proxy: CCPI-based series (harmonized to `CCPI_2021_Base`)
- Interest-rate proxies: Standing deposit and standing lending facility rates
- Exchange-rate proxy: Monthly average LKR/USD exchange rate

These macro indicators are transformed (and aligned) for econometric modeling, including log transformations where applicable in downstream VECM/EGARCH workflows.

Source link placeholders for final submission:
- CBSL eResearch portal: https://www.cbsl.lk/eResearch/
- CSE historical data portal: https://www.cse.lk/listed-entities/indices

## 2. Assumptions

The preprocessing and merge pipeline applies the following explicit assumptions:

- Trading calendar anchor:
  Daily stock-market observations (ASPI and sector indices) define the reference timeline to which macro variables are aligned.

- Missing-value handling and continuity:
  For variables not observed on every trading day, the latest available value is carried forward to preserve continuity.

- Mixed-frequency harmonization:
  Monthly macro series are mapped to daily records via month-key merging so each trading day inherits the relevant monthly macro state.

- Rate and price transformations:
  Financial series are converted to returns in analysis notebooks, and macro series are transformed (including log transformations where appropriate) to stabilize scale and support interpretable dynamic modeling.

- Modeling suitability:
  The resulting aligned panel is treated as adequate for VECM cointegration analysis and EGARCH volatility estimation under the study design.

## 3. Limitations

Despite structured cleaning and alignment, the dataset has important limitations:

- Liquidity-based exclusion effects:
  The pipeline excludes highly illiquid or weak-coverage segments identified through zero-return ratio diagnostics (`sector_liquidity_report.csv`) and related screening decisions. This improves model stability but can reduce representation of thinly traded market segments.

- Regime-window scope:
  Results are conditioned on the study's fixed windows:
  - Pre-Crisis: 2019-01-01 to 2022-02-28
  - Crisis: 2022-03-01 to 2023-02-28
  - Recovery: 2023-03-01 to 2025-08-31

- Daily-monthly integration trade-off:
  Mapping monthly macro indicators to daily observations may smooth intramonth shocks and can attenuate very short-run macro transmission effects.

- Context-specific generalizability:
  The panel is specific to Sri Lanka and the CSE during a sovereign-default episode; external validity to other markets or non-crisis periods is limited.

## 4. Ethical Considerations

All variables in this repository are drawn from publicly available aggregate datasets obtained from the CSE portal and the CBSL eResearch portal. The data contain no Personally Identifiable Information (PII), no client-level records, and no private transactional identifiers.

Accordingly, the project follows standard ethical practice for secondary quantitative finance research: transparent sourcing, non-intrusive collection, reproducible transformation, and analysis limited to public market and macroeconomic indicators.