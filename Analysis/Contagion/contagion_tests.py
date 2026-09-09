"""Heteroskedasticity-adjusted contagion tests for the CSE sector panel.

Forbes and Rigobon (2002) show that cross-market correlation rises mechanically
when the source market becomes more volatile, even with no change in the
underlying transmission mechanism. Comparing raw crisis and pre-crisis
correlations therefore cannot distinguish contagion from interdependence.

This script applies their adjustment. Because the ASPI contains the sector
indices, correlating a sector against the ASPI is mechanically inflated; the
primary specification therefore uses a leave-one-out market factor (the
equal-weighted mean return of the other twelve sectors). Results against the
ASPI and across all sector pairs are reported as alternatives.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import itertools

REGIMES = {'Pre-Crisis': ('2019-01-01', '2022-01-24'),
           'Crisis': ('2022-01-25', '2022-12-05'),
           'Recovery': ('2022-12-06', '2025-08-31')}
SECTORS = ['Banks', 'Capital_Goods', 'Consumer_Durables_and_Apparel', 'Consumer_Services',
           'Diversified_Financials', 'Energy', 'Food,_Beverage_and_Tobacco',
           'Healthcare_Equipment_and_Services', 'Insurance', 'Materials',
           'Real_Estate_Management_and_Development', 'Retailing', 'Telecommunication_Services']

df = pd.read_csv('../../master_dataset.csv', parse_dates=['date'], index_col='date')
rets = pd.DataFrame({c: np.log(df[c] / df[c].shift(1)) * 100
                     for c in ['ASPI'] + SECTORS}).dropna()
seg = {k: rets.loc[a:b] for k, (a, b) in REGIMES.items()}


def fr_adjust(rho_crisis, delta):
    """Forbes-Rigobon heteroskedasticity-adjusted correlation."""
    return rho_crisis / np.sqrt(1.0 + delta * (1.0 - rho_crisis ** 2))


def fisher_z(r1, n1, r2, n2):
    z1, z2 = np.arctanh(np.clip(r1, -0.999, 0.999)), np.arctanh(np.clip(r2, -0.999, 0.999))
    z = (z1 - z2) / np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    return z, 2.0 * (1.0 - norm.cdf(abs(z)))


print("=" * 78)
print("1. RAW CROSS-SECTOR CORRELATION AND VOLATILITY BY REGIME")
print("=" * 78)
iu = np.triu_indices(len(SECTORS), 1)
summary = []
for k, d in seg.items():
    cm = d[SECTORS].corr().values
    summary.append({'Regime': k, 'N': len(d),
                    'Mean pairwise corr': round(float(cm[iu].mean()), 3),
                    'ASPI variance': round(float(d['ASPI'].var()), 3)})
    print("  %-11s n=%4d  mean pairwise rho = %.3f   ASPI variance = %.3f"
          % (k, len(d), cm[iu].mean(), d['ASPI'].var()))
pd.DataFrame(summary).to_csv('correlation_summary.csv', index=False)

print("\n" + "=" * 78)
print("2. FORBES-RIGOBON ADJUSTMENT, LEAVE-ONE-OUT MARKET FACTOR (primary)")
print("=" * 78)
rows = []
for target in ['Crisis', 'Recovery']:
    a, b = seg[target], seg['Pre-Crisis']
    n_cont = n_dec = n_flat = 0
    for c in SECTORS:
        others = [s for s in SECTORS if s != c]
        mb, ma = b[others].mean(axis=1), a[others].mean(axis=1)
        delta = ma.var() / mb.var() - 1.0
        rho_b, rho_a = b[c].corr(mb), a[c].corr(ma)
        rho_adj = fr_adjust(rho_a, delta)
        _, p = fisher_z(rho_adj, len(a), rho_b, len(b))
        if rho_adj > rho_b and p < 0.05:
            verdict, n_cont = 'contagion', n_cont + 1
        elif rho_adj < rho_b and p < 0.05:
            verdict, n_dec = 'decoupling', n_dec + 1
        else:
            verdict, n_flat = 'no change', n_flat + 1
        rows.append({'Comparison': target + ' vs Pre-Crisis', 'Sector': c,
                     'rho_pre': round(rho_b, 3), 'rho_target': round(rho_a, 3),
                     'rho_adjusted': round(rho_adj, 3), 'p': round(p, 4),
                     'verdict': verdict})
    print("\n  %s vs Pre-Crisis   (market variance ratio delta = %+.2f)" % (target, delta))
    print("  %-40s%9s%9s%9s%8s   %s" % ('Sector', 'rho_pre', 'rho_tgt', 'FR-adj', 'p', 'verdict'))
    for r in [x for x in rows if x['Comparison'].startswith(target)]:
        print("  %-40s%9.3f%9.3f%9.3f%8.3f   %s"
              % (r['Sector'], r['rho_pre'], r['rho_target'], r['rho_adjusted'], r['p'], r['verdict']))
    print("  --> contagion %d/13, decoupling %d/13, no change %d/13" % (n_cont, n_dec, n_flat))
pd.DataFrame(rows).to_csv('forbes_rigobon_leave_one_out.csv', index=False)

print("\n" + "=" * 78)
print("3. ALTERNATIVE SPECIFICATIONS")
print("=" * 78)
for target in ['Crisis', 'Recovery']:
    a, b = seg[target], seg['Pre-Crisis']
    # (a) sector vs ASPI
    delta = a['ASPI'].var() / b['ASPI'].var() - 1.0
    n_cont = 0
    for c in SECTORS:
        rho_b, rho_a = b[c].corr(b['ASPI']), a[c].corr(a['ASPI'])
        radj = fr_adjust(rho_a, delta)
        _, p = fisher_z(radj, len(a), rho_b, len(b))
        n_cont += int(radj > rho_b and p < 0.05)
    print("  %-9s sector vs ASPI            : contagion in %2d/13" % (target, n_cont))
    # (b) all directed sector pairs
    n_c = n_t = 0
    for x, y in itertools.combinations(SECTORS, 2):
        for src in (x, y):
            d2 = a[src].var() / b[src].var() - 1.0
            rho_b, rho_a = b[x].corr(b[y]), a[x].corr(a[y])
            radj = fr_adjust(rho_a, d2)
            _, p = fisher_z(radj, len(a), rho_b, len(b))
            n_c += int(radj > rho_b and p < 0.05); n_t += 1
    print("  %-9s all directed sector pairs : contagion in %2d/%d" % (target, n_c, n_t))

print("\nSaved: correlation_summary.csv, forbes_rigobon_leave_one_out.csv")
