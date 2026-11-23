"""False-discovery-rate control for the contagion tests, requested in review.

contagion_tests.py reports how many Fisher z tests reject at the 5% level. With
13 sector tests and 156 directed pair tests per comparison, several rejections
are expected by chance alone. This script re-runs the same tests (identical
data, regimes and Forbes-Rigobon adjustment) and applies the Benjamini-Hochberg
procedure at FDR 5% within each family, reporting increases (contagion) and
decreases (decoupling) separately.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm

REGIMES = {'Pre-Crisis': ('2019-01-01', '2022-01-24'),
           'Crisis': ('2022-01-25', '2022-12-05'),
           'Recovery': ('2022-12-06', '2025-08-31')}
SECTORS = ['Banks', 'Capital_Goods', 'Consumer_Durables_and_Apparel', 'Consumer_Services',
           'Diversified_Financials', 'Energy', 'Food,_Beverage_and_Tobacco',
           'Healthcare_Equipment_and_Services', 'Insurance', 'Materials',
           'Real_Estate_Management_and_Development', 'Retailing', 'Telecommunication_Services']


def fr_adjust(rho, delta):
    return rho / np.sqrt(1.0 + delta * (1.0 - rho ** 2))


def fisher_p(r1, n1, r2, n2):
    z1, z2 = np.arctanh(np.clip(r1, -0.999, 0.999)), np.arctanh(np.clip(r2, -0.999, 0.999))
    return 2.0 * (1.0 - norm.cdf(abs((z1 - z2) / np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3)))))


def bh_pass(p, q=0.05):
    p = np.asarray(p, float)
    n = len(p)
    rank = np.empty(n, int)
    rank[np.argsort(p)] = np.arange(1, n + 1)
    ok = p <= rank / n * q
    return rank <= rank[ok].max() if ok.any() else np.zeros(n, bool)


df = pd.read_csv('../../master_dataset.csv', parse_dates=['date'], index_col='date')
rets = pd.DataFrame({c: np.log(df[c] / df[c].shift(1)) * 100 for c in ['ASPI'] + SECTORS}).dropna()
seg = {k: rets.loc[a:b] for k, (a, b) in REGIMES.items()}

rows = []
for target in ['Crisis', 'Recovery']:
    a, b = seg[target], seg['Pre-Crisis']
    for c in SECTORS:                                            # leave-one-out factor
        others = [s for s in SECTORS if s != c]
        mb, ma = b[others].mean(axis=1), a[others].mean(axis=1)
        radj = fr_adjust(a[c].corr(ma), ma.var() / mb.var() - 1.0)
        rb = b[c].corr(mb)
        rows.append((target, 'Leave-one-out factor', c, radj > rb, fisher_p(radj, len(a), rb, len(b))))
    delta = a['ASPI'].var() / b['ASPI'].var() - 1.0              # ASPI as market proxy
    for c in SECTORS:
        radj, rb = fr_adjust(a[c].corr(a['ASPI']), delta), b[c].corr(b['ASPI'])
        rows.append((target, 'ASPI as market proxy', c, radj > rb, fisher_p(radj, len(a), rb, len(b))))
    for x, y in itertools.combinations(SECTORS, 2):                # all directed pairs
        for src in (x, y):
            radj = fr_adjust(a[x].corr(a[y]), a[src].var() / b[src].var() - 1.0)
            rb = b[x].corr(b[y])
            rows.append((target, 'All directed sector pairs', f'{x}<-{src}', radj > rb,
                         fisher_p(radj, len(a), rb, len(b))))

res = pd.DataFrame(rows, columns=['Regime', 'Specification', 'Test', 'Increase', 'p'])
res['BH'] = False
for key, idx in res.groupby(['Regime', 'Specification']).groups.items():
    res.loc[idx, 'BH'] = bh_pass(res.loc[idx, 'p'])
res.to_csv('contagion_fdr.csv', index=False)

print('Significant adjusted change vs the pre-default window: raw 5% -> Benjamini-Hochberg FDR 5%')
print(f"{'Regime':<9} {'Specification':<27} {'n':>4} {'increase':>15} {'decrease':>15}")
for (reg, spec), g in res.groupby(['Regime', 'Specification'], sort=False):
    inc, dec = g['Increase'], ~g['Increase']
    raw = g['p'] < 0.05
    print(f"{'Default' if reg == 'Crisis' else reg:<9} {spec:<27} {len(g):>4} "
          f"{int((raw & inc).sum()):>6} -> {int((g['BH'] & inc).sum()):>4} "
          f"{int((raw & dec).sum()):>8} -> {int((g['BH'] & dec).sum()):>4}")
