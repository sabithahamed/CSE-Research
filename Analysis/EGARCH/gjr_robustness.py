import zlib
import warnings
import numpy as np
import pandas as pd
from arch import arch_model

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# Sensitivity check on the asymmetry result, requested in review.
#
# The main analysis measures asymmetry with EGARCH(1,1)-t. Here every series in
# every regime is re-estimated with a second asymmetric specification,
# GJR-GARCH(1,1)-t, which models asymmetry differently: an extra ARCH term
# switched on only by negative shocks. In GJR a POSITIVE gamma is the leverage
# effect (bad news raises variance more), the opposite sign convention to
# EGARCH. If the default-regime result depends on the EGARCH functional form,
# it should weaken here.
# ----------------------------------------------------------------------------

N_STARTS = 8
REGIMES = {
    'Pre-COVID': (None, '2020-02-10'),
    'COVID': ('2020-02-11', '2022-01-24'),
    'Crisis': ('2022-01-25', '2022-12-05'),
    'Recovery': ('2022-12-06', '2025-08-31'),
}
SECTORS = [
    'Banks', 'Capital_Goods', 'Consumer_Durables_and_Apparel',
    'Consumer_Services', 'Diversified_Financials', 'Energy',
    'Food,_Beverage_and_Tobacco', 'Materials', 'Real_Estate_Management_and_Development',
    'Retailing', 'Telecommunication_Services', 'Healthcare_Equipment_and_Services', 'Insurance'
]
SERIES = ['ASPI'] + SECTORS


def fit_gjr(y, seed):
    """Fit GJR-GARCH(1,1)-t from multiple starts; return the highest-likelihood fit."""
    rng = np.random.default_rng(seed)
    am = arch_model(y, vol='GARCH', p=1, o=1, q=1, dist='t')
    fits = []
    try:
        fits.append(am.fit(disp='off', options={'maxiter': 2000}))
    except Exception:
        pass
    var = float(np.var(y))
    for _ in range(N_STARTS):
        a, g, b = rng.uniform(0.02, 0.15), rng.uniform(0.0, 0.2), rng.uniform(0.6, 0.95)
        b = min(b, 0.98 - a - g / 2)                       # keep the start stationary
        sv = np.array([y.mean(), var * (1 - a - g / 2 - b), a, g, b, rng.uniform(4, 15)])
        try:
            fits.append(am.fit(disp='off', starting_values=sv, options={'maxiter': 2000}))
        except Exception:
            pass
    fits = [f for f in fits if np.isfinite(f.loglikelihood)]
    return max(fits, key=lambda f: f.loglikelihood) if fits else None


def bh_pass(p, q=0.05):
    """Benjamini-Hochberg: True where the test survives FDR control at level q."""
    p = np.asarray(p, float)
    n = len(p)
    rank = np.empty(n, int)
    rank[np.argsort(p)] = np.arange(1, n + 1)
    ok = p <= rank / n * q
    return rank <= rank[ok].max() if ok.any() else np.zeros(n, bool)


def main():
    df = pd.read_csv('../../master_dataset.csv', parse_dates=['date'], index_col='date')
    for c in SERIES:
        df[f'{c}_Return'] = np.log(df[c] / df[c].shift(1)) * 100
    df = df.dropna(subset=[f'{c}_Return' for c in SERIES])

    rows = []
    for reg, (a, b) in REGIMES.items():
        part = df.loc[a:b]
        for c in SERIES:
            y = part[f'{c}_Return'].dropna()
            res = fit_gjr(y, seed=zlib.crc32(f'{reg}|{c}'.encode()))
            if res is None:
                print(f'all starts failed: {reg} {c}')
                continue
            p, se, pv = res.params, res.std_err, res.pvalues
            rows.append({
                'Period': reg, 'Asset': c, 'N': len(y),
                'Alpha': p['alpha[1]'], 'Gamma': p['gamma[1]'], 'Gamma SE': se['gamma[1]'],
                'Gamma P-Value': pv['gamma[1]'], 'Beta': p['beta[1]'], 'Nu': p['nu'],
                'Persistence': p['alpha[1]'] + p['gamma[1]'] / 2 + p['beta[1]'],
                'Log-Likelihood': res.loglikelihood, 'Converged': res.convergence_flag == 0,
                'Any SE < 1e-4': bool((se < 1e-4).any()),
            })
    out = pd.DataFrame(rows)
    out['Leverage (gamma>0, p<0.05)'] = (out['Gamma'] > 0) & (out['Gamma P-Value'] < 0.05)
    out['BH within regime'] = False
    for reg in REGIMES:
        m = out['Period'] == reg
        out.loc[m, 'BH within regime'] = bh_pass(out.loc[m, 'Gamma P-Value']) & (out.loc[m, 'Gamma'] > 0)
    out.to_csv('gjr_1_1_results.csv', index=False)

    eg = pd.read_csv('egarch_1_1_results.csv')
    eg['sig'] = (eg['Gamma (Leverage)'] < 0) & (eg['Gamma P-Value'] < 0.05)
    print('Significant leverage effect by regime (5% level; BH = FDR-controlled within regime)')
    print(f"{'Regime':<10} {'EGARCH':>7} {'GJR':>5} {'GJR BH':>7} {'both':>5}   non-converged / SE<1e-4")
    for reg in REGIMES:
        g = out[out['Period'] == reg].set_index('Asset')
        e = eg[eg['Period'] == reg].set_index('Asset')
        both = int((g['Leverage (gamma>0, p<0.05)'] & e['sig'].reindex(g.index)).sum())
        print(f"{'Default' if reg == 'Crisis' else reg:<10} {int(e['sig'].sum()):>4}/14 "
              f"{int(g['Leverage (gamma>0, p<0.05)'].sum()):>2}/14 {int(g['BH within regime'].sum()):>4}/14 "
              f"{both:>5}   {int((~g['Converged']).sum())} / {int(g['Any SE < 1e-4'].sum())}")
    print('\nDefault regime, GJR gamma by series:')
    print(out[out['Period'] == 'Crisis'][['Asset', 'Gamma', 'Gamma P-Value', 'Persistence', 'Any SE < 1e-4']]
          .sort_values('Gamma P-Value').to_string(index=False))


if __name__ == '__main__':
    main()
