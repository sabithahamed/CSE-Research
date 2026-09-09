"""Re-estimate the cointegrating relation within each detected regime.

The paper claims the long-run relation is not invariant across regimes. This
script produces the evidence for that claim and saves it, so the assertion in
the text is reproducible from the repository.

Two things it checks that a naive re-fit does not:

  1. Whether each regime actually supports a cointegrating relation at all
     (Johansen trace test run separately per regime). If a regime has rank 0,
     "the relation in that regime" is not a well-defined object.

  2. Whether the normalisation is well conditioned. The cointegrating vector is
     only identified up to scale, so it is conventionally normalised to put a
     coefficient of 1 on the index. If the index's own weight in the raw
     eigenvector is near zero, that division blows the other coefficients up and
     the reported numbers are an artefact of the normalisation, not an estimate.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order

warnings.filterwarnings('ignore')

REGIMES = {
    'Pre-COVID': (None, '2020-02-10'),
    'COVID': ('2020-02-11', '2022-01-24'),
    'Default': ('2022-01-25', '2022-12-05'),
    'Recovery': ('2022-12-06', None),
}
COLS = ['ln_ASPI', 'ln_Exchange_Rate', 'Interest_Rate', 'ln_CCPI']


def load():
    d = pd.read_csv('../../master_dataset.csv', parse_dates=['date']).set_index('date')
    return pd.DataFrame({
        'ln_ASPI': np.log(d['ASPI']),
        'ln_Exchange_Rate': np.log(d['Monthly_Average_Exchange_Rates']),
        'Interest_Rate': d['Reverse_Repo_Standing_Lending_Facility_Rate'],
        'ln_CCPI': np.log(d['CCPI_2021_Base']),
    }).dropna()


def analyse(sample, lags):
    """Return rank, normalisation conditioning, and the implied FX coefficient."""
    jres = coint_johansen(sample, det_order=0, k_ar_diff=lags)
    rank = int((jres.lr1 > jres.cvt[:, 1]).sum())          # trace test at 5%

    # Raw eigenvector for the first relation, before any normalisation.
    raw = jres.evec[:, 0]
    weight = abs(raw[0]) / np.linalg.norm(raw)             # index's share of the vector

    fit = VECM(sample, k_ar_diff=lags, coint_rank=1, deterministic='ci').fit()
    beta = fit.beta[:, 0]                                  # statsmodels sets beta[0] = 1
    ci = fit.conf_int_beta()[1][0]
    implied = -beta[1]                                     # ln_ASPI = -beta1*ln_FX + ...
    lo, hi = sorted([-ci[0], -ci[1]])
    return rank, weight, implied, lo, hi


def main():
    D = load()
    full_lags = 5
    rows = []
    for name, (a, b) in REGIMES.items():
        s = D.loc[(a or D.index[0]):(b or D.index[-1])]
        sel = select_order(s, maxlags=8, deterministic='ci')
        own = max(1, (sel.bic or 1) - 1)                   # VAR order -> k_ar_diff
        for label, lags in (('fixed k=5', full_lags), ('BIC per regime', own)):
            rank, weight, implied, lo, hi = analyse(s, lags)
            rows.append(dict(regime=name, N=len(s), lag_rule=label, k_ar_diff=lags,
                             johansen_rank=rank, index_weight=round(weight, 4),
                             implied_fx=round(implied, 3),
                             ci_lo=round(lo, 3), ci_hi=round(hi, 3)))

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv('regime_vecm_estimates.csv', index=False)

    for label in ('fixed k=5', 'BIC per regime'):
        sub = out[out.lag_rule == label].set_index('regime')
        usable = sub[sub.johansen_rank >= 1]
        print('\n--- %s ---' % label)
        print('regimes supporting a cointegrating relation at 5%%: %d of 4 (%s)'
              % (len(usable), ', '.join(usable.index) if len(usable) else 'none'))
        n_dis = 0
        pairs = list(itertools.combinations(sub.index, 2))
        for x, y in pairs:
            rx, ry = sub.loc[x], sub.loc[y]
            disjoint = rx.ci_hi < ry.ci_lo or ry.ci_hi < rx.ci_lo
            n_dis += disjoint
            print('   %-10s vs %-10s %s' % (x, y, 'disjoint' if disjoint else 'OVERLAP'))
        print('   disjoint: %d of %d' % (n_dis, len(pairs)))
        print('   full-sample -9.51 inside each regime CI: %s'
              % {i: bool(r.ci_lo <= -9.51 <= r.ci_hi) for i, r in sub.iterrows()})


if __name__ == '__main__':
    main()
