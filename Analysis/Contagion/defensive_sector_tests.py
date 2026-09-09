"""Did traditionally defensive sectors behave differently during the default?

The claim under test is a proposition about a sector *class*, not about
individual sectors: standard portfolio theory holds that defensive sectors
provide protection during broad sell-offs. We therefore classify sectors as
defensive or cyclical a priori, then ask whether the defensive group differs
systematically from the cyclical group on three dimensions:

  (i)   protection      - crisis-period mean return and total volatility
  (ii)  decoupling      - share of variance explained by a market factor
  (iii) idiosyncratic   - residual (sector-specific) volatility

The market factor is the equal-weighted mean return of the other twelve
sectors, so the reference series never contains the sector being tested.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu

REGIMES = {'Pre-Crisis': ('2019-01-01', '2022-01-24'),
           'Crisis': ('2022-01-25', '2022-12-05'),
           'Recovery': ('2022-12-06', '2025-08-31')}

# A priori classification. Healthcare, food/staples and telecommunications are
# the conventional non-cyclical (defensive) groups; the remainder are cyclical.
# Insurance is classified as cyclical: it sits in Financials under GICS, and
# treating it as defensive is not standard.
DEFENSIVE = ['Healthcare_Equipment_and_Services',
             'Food,_Beverage_and_Tobacco',
             'Telecommunication_Services']
CYCLICAL = ['Banks', 'Capital_Goods', 'Consumer_Durables_and_Apparel', 'Consumer_Services',
            'Diversified_Financials', 'Energy', 'Insurance', 'Materials',
            'Real_Estate_Management_and_Development', 'Retailing']
SECTORS = DEFENSIVE + CYCLICAL

df = pd.read_csv('../../master_dataset.csv', parse_dates=['date'], index_col='date')
rets = pd.DataFrame({c: np.log(df[c] / df[c].shift(1)) * 100 for c in SECTORS}).dropna()
seg = {k: rets.loc[a:b] for k, (a, b) in REGIMES.items()}

rows = []
for c in SECTORS:
    others = [s for s in SECTORS if s != c]
    rec = {'Sector': c, 'Class': 'Defensive' if c in DEFENSIVE else 'Cyclical'}
    for k, d in seg.items():
        mkt = d[others].mean(axis=1)
        fit = sm.OLS(d[c], sm.add_constant(mkt)).fit()
        rec['R2_' + k] = fit.rsquared
        rec['beta_' + k] = fit.params.iloc[1]
        rec['idio_' + k] = float(np.std(fit.resid, ddof=2))
        rec['totvol_' + k] = float(d[c].std())
        rec['meanret_' + k] = float(d[c].mean())
    rec['dR2_crisis'] = rec['R2_Crisis'] - rec['R2_Pre-Crisis']
    rec['idio_ratio_crisis'] = rec['idio_Crisis'] / rec['idio_Pre-Crisis'] - 1.0
    rec['pattern'] = (rec['dR2_crisis'] < 0) and (rec['idio_ratio_crisis'] > 0)
    rows.append(rec)
res = pd.DataFrame(rows)

print("=" * 92)
print("SECTOR-LEVEL DECOMPOSITION, CRISIS vs PRE-CRISIS")
print("=" * 92)
print("%-40s%-11s%9s%9s%11s%9s" % ('Sector', 'Class', 'R2 pre', 'R2 cri', 'idio chg', 'pattern'))
for _, r in res.sort_values(['Class', 'Sector']).iterrows():
    print("%-40s%-11s%9.2f%9.2f%10.1f%%%9s"
          % (r['Sector'][:38], r['Class'], r['R2_Pre-Crisis'], r['R2_Crisis'],
             100 * r['idio_ratio_crisis'], 'yes' if r['pattern'] else 'no'))

d_pat = res[res['Class'] == 'Defensive']['pattern'].sum()
c_pat = res[res['Class'] == 'Cyclical']['pattern'].sum()
print("\n  sectors showing decoupling + higher idiosyncratic volatility:")
print("     Defensive: %d/%d      Cyclical: %d/%d" % (d_pat, len(DEFENSIVE), c_pat, len(CYCLICAL)))

print("\n" + "=" * 92)
print("DOES THE DEFENSIVE GROUP DIFFER SYSTEMATICALLY? (Mann-Whitney, two-sided)")
print("=" * 92)
tests = [('dR2_crisis', 'change in market-explained variance'),
         ('idio_ratio_crisis', 'change in idiosyncratic volatility'),
         ('meanret_Crisis', 'crisis mean daily return'),
         ('totvol_Crisis', 'crisis total volatility')]
for col, lab in tests:
    a = res[res['Class'] == 'Defensive'][col].values
    b = res[res['Class'] == 'Cyclical'][col].values
    u, p = mannwhitneyu(a, b, alternative='two-sided')
    print("  %-38s defensive median %+8.3f | cyclical median %+8.3f | p = %.3f"
          % (lab, np.median(a), np.median(b), p))

print("\n" + "=" * 92)
print("DID DEFENSIVE SECTORS PROTECT DURING THE CRISIS?")
print("=" * 92)
for cls in ['Defensive', 'Cyclical']:
    g = res[res['Class'] == cls]
    print("  %-10s crisis mean return %+.4f%%/day | crisis volatility %.3f%% | crisis beta %.2f"
          % (cls, g['meanret_Crisis'].mean(), g['totvol_Crisis'].mean(), g['beta_Crisis'].mean()))

res.to_csv('defensive_sector_decomposition.csv', index=False)
print("\nSaved: defensive_sector_decomposition.csv")
