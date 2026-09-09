"""Descriptive summary of the four estimated regimes.

The regime boundaries come from break_tests.py, which detects breaks in the ASPI
return variance at 2020-02-11, 2022-01-25 and 2022-12-06. This script reports
the window, sample size and realised volatility of each resulting regime, so
the descriptive figures quoted in the paper can be checked without rerunning
the break search or any model.

Volatility is the sample standard deviation of daily ASPI log returns within the
regime, expressed in percent per day.
"""
import numpy as np
import pandas as pd

REGIMES = [
    ('Pre-COVID', None, '2020-02-10'),
    ('COVID', '2020-02-11', '2022-01-24'),
    ('Default', '2022-01-25', '2022-12-05'),
    ('Recovery', '2022-12-06', None),
]

df = pd.read_csv('../../master_dataset.csv', parse_dates=['date'], index_col='date')
ret = (np.log(df['ASPI']).diff() * 100).dropna()

rows = []
for name, a, b in REGIMES:
    r = ret.loc[(a or ret.index[0]):(b or ret.index[-1])]
    rows.append({
        'Regime': name,
        'Start': r.index[0].date(),
        'End': r.index[-1].date(),
        'N': len(r),
        'Volatility (%/day)': round(r.std(), 3),
        'Mean return (%/day)': round(r.mean(), 4),
        'Min (%)': round(r.min(), 2),
        'Max (%)': round(r.max(), 2),
    })
out = pd.DataFrame(rows)

print("=" * 88)
print("ESTIMATED REGIMES: DESCRIPTIVE SUMMARY")
print("=" * 88)
print(out.to_string(index=False))

pre = out.loc[out.Regime == 'Pre-COVID', 'Volatility (%/day)'].iloc[0]
for name in ('COVID', 'Default', 'Recovery'):
    v = out.loc[out.Regime == name, 'Volatility (%/day)'].iloc[0]
    print("\n  %-9s volatility is %.2fx the pre-COVID level (%.2f%% vs %.2f%%)"
          % (name, v / pre, v, pre))

print("\n  Note: N counts return observations, so it is one fewer than the number")
print("  of price observations in the first regime. Non-trading days, including")
print("  the April 2022 exchange closure, are absent rather than imputed.")

out.to_csv('regime_summary.csv', index=False)
print("\nSaved: regime_summary.csv")
