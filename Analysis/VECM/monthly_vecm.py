"""Frequency-aligned VECM, requested in review.

The main VECM is estimated on daily data, with the monthly macroeconomic series
(CCPI, USD/LKR) carried forward between releases. Holding a value constant for a
month builds step-like persistence into those series. Here the same system is
re-estimated with every variable at monthly frequency, so no value is carried
forward: the ASPI and the policy rate are averaged within each month, matching
the exchange rate, which is published as a monthly average.

The specification mirrors VECM.ipynb and irf_vd.ipynb as closely as the shorter
sample allows: lag order chosen by BIC on the differenced VAR, Johansen trace
test with det_order=0, VECM with deterministic='colo', and forecast-error
variance decomposition from the equivalent levels VAR with the ASPI ordered
first (so macroeconomic shares are lower bounds, as in the daily model).
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

warnings.filterwarnings('ignore')

COLS = ['ln_ASPI', 'ln_Exchange_Rate', 'Interest_Rate', 'ln_CCPI']

df = pd.read_csv('../../master_dataset.csv', parse_dates=['date'], index_col='date')
df = df.loc['2019-01-01':'2025-08-31']
raw = df[['ASPI', 'Reverse_Repo_Standing_Lending_Facility_Rate',
          'CCPI_2021_Base', 'Monthly_Average_Exchange_Rates']].dropna()


def to_monthly(how):
    m = pd.DataFrame({
        'ln_ASPI': np.log(raw['ASPI'].resample('ME').agg(how)),
        'ln_Exchange_Rate': np.log(raw['Monthly_Average_Exchange_Rates'].resample('ME').mean()),
        'Interest_Rate': raw['Reverse_Repo_Standing_Lending_Facility_Rate'].resample('ME').mean(),
        'ln_CCPI': np.log(raw['CCPI_2021_Base'].resample('ME').mean()),
    })[COLS]
    return m.dropna()


def run(label, m, maxlags=6):
    sel = VAR(m.diff().dropna()).select_order(maxlags=maxlags)
    p = sel.bic if sel.bic > 0 else 2
    k = max(1, p - 1)
    jo = coint_johansen(m, det_order=0, k_ar_diff=k)
    rank = next((i for i in range(len(jo.lr1)) if jo.lr1[i] < jo.cvt[i, 1]), len(jo.lr1))
    print(f'\n=== {label}: {len(m)} months, BIC lag order {p} (k_ar_diff={k}) ===')
    for i in range(len(jo.lr1)):
        print(f'  trace r<={i}: {jo.lr1[i]:7.2f}   5% cv {jo.cvt[i, 1]:6.2f}   '
              f'{"reject" if jo.lr1[i] > jo.cvt[i, 1] else "do not reject"}')
    print(f'  cointegrating rank at 5%: {rank}')
    out = {'Specification': label, 'Months': len(m), 'BIC lags': p, 'Rank (5%)': rank,
           'Trace r=0': jo.lr1[0], 'CV 5% r=0': jo.cvt[0, 1]}
    if rank >= 1:
        vr = VECM(m, k_ar_diff=k, coint_rank=1, deterministic='colo').fit()
        a, pv = vr.alpha[0, 0], vr.pvalues_alpha[0, 0]
        hl = np.log(0.5) / np.log(1 + a) if -1 < a < 0 else np.nan
        print(f'  ASPI error-correction coefficient {a:+.4f} (p={pv:.3f}), half-life {hl:.1f} months')
        out.update({'ASPI ECT': a, 'ASPI ECT p': pv, 'Half-life (months)': hl})
    fe = VAR(m).fit(k + 1).fevd(12).decomp[0] * 100           # rows: horizon, cols: shock
    for h in (1, 2, 3, 6, 12):
        macro = 100 - fe[h - 1, 0]
        out[f'Macro share h={h}'] = macro
    print('  macro share of ASPI forecast-error variance: ' +
          ', '.join(f'{h}m {out[f"Macro share h={h}"]:.1f}%' for h in (1, 2, 3, 6, 12)))
    return out


rows = [run('Monthly average ASPI', to_monthly('mean')),
        run('Month-end ASPI', to_monthly('last'))]
pd.DataFrame(rows).to_csv('monthly_vecm_results.csv', index=False)
print('\nSaved: monthly_vecm_results.csv')
print('Daily model for comparison: macro share 1.1% at 5 days, 3.9% at 30 days; '
      'ECT -0.00125 (p=0.039), half-life ~2.2 years')
