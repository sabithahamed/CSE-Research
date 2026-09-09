import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ruptures as rpt
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# Data-driven identification of volatility regimes in the ASPI.
#
# Reviewers of an earlier draft noted that regimes assigned from policy-event
# dates are imposed rather than detected. This script identifies regime
# boundaries from the return series itself using three independent procedures,
# then checks whether the EGARCH conclusions depend on which segmentation is
# used.
# ----------------------------------------------------------------------------

POLICY = {'Pre-Crisis': ('2019-01-01', '2022-02-28'),
          'Crisis': ('2022-03-01', '2023-02-28'),
          'Recovery': ('2023-03-01', '2025-08-31')}
MIN_SEG = 126          # about six months: minimum admissible regime length
TRIM = 0.15            # trimming fraction for the sup-F search

df = pd.read_csv('../../master_dataset.csv', parse_dates=['date'], index_col='date')
r = (np.log(df['ASPI'] / df['ASPI'].shift(1)) * 100).dropna()
lv = np.log(r.values ** 2 + 1e-8)          # log squared returns, a log-variance proxy
n = len(lv)
print("ASPI log returns: n = %d, %s to %s\n" % (n, r.index.min().date(), r.index.max().date()))

# --- 1. Sequential break estimation in the log variance (Bai-Perron style) ---
print("=" * 74)
print("1. BREAK ESTIMATION IN LOG VARIANCE (dynamic programming + BIC)")
print("=" * 74)
algo = rpt.Dynp(model="l2", min_size=MIN_SEG, jump=1).fit(lv)
bic_rows = []
for k in range(1, 6):
    bkps = algo.predict(n_bkps=k)
    ssr = sum(((lv[a:b] - lv[a:b].mean()) ** 2).sum()
              for a, b in zip([0] + bkps[:-1], bkps))
    bic = n * np.log(ssr / n) + (2 * k + 1) * np.log(n)
    dates = [str(r.index[i].date()) for i in bkps[:-1]]
    bic_rows.append({'k': k, 'BIC': round(bic, 1), 'break_dates': ", ".join(dates)})
    print("  k=%d  BIC=%9.1f   %s" % (k, bic, dates))
bic_df = pd.DataFrame(bic_rows)
k_star = int(bic_df.loc[bic_df['BIC'].idxmin(), 'k'])
breaks = [r.index[i] for i in algo.predict(n_bkps=k_star)[:-1]]
print("\n  BIC-selected number of breaks: k = %d" % k_star)
print("  Estimated break dates: %s" % [str(d.date()) for d in breaks])

# --- 2. sup-F test for a single break in the variance ---
print("\n" + "=" * 74)
print("2. SUP-F TEST FOR A VARIANCE BREAK (15 percent trimming)")
print("=" * 74)
lo, hi = int(TRIM * n), int((1 - TRIM) * n)
tot = ((lv - lv.mean()) ** 2).sum()
F = []
for i in range(lo, hi):
    s1, s2 = lv[:i], lv[i:]
    ssr = ((s1 - s1.mean()) ** 2).sum() + ((s2 - s2.mean()) ** 2).sum()
    F.append(((tot - ssr) / 1) / (ssr / (n - 2)))
F = np.array(F)
i_star = lo + int(F.argmax())
print("  sup-F = %.1f at %s" % (F.max(), r.index[i_star].date()))
print("  Andrews (1993) 5 percent critical value, one parameter, 15 percent trimming: 8.85")

# --- 3. Markov-switching variance model ---
print("\n" + "=" * 74)
print("3. TWO-STATE MARKOV-SWITCHING VARIANCE MODEL")
print("=" * 74)
ms = MarkovRegression(r.values, k_regimes=2, trend='c', switching_variance=True)
ms_res = ms.fit(search_reps=20, disp=False)
sd = np.sqrt(ms_res.params[-2:])
hi_state = int(np.argmax(sd))
print("  low-volatility state:  sd = %.3f pct, expected duration %.0f days"
      % (sd[1 - hi_state], ms_res.expected_durations[1 - hi_state]))
print("  high-volatility state: sd = %.3f pct, expected duration %.0f days"
      % (sd[hi_state], ms_res.expected_durations[hi_state]))
smoothed = pd.Series(ms_res.smoothed_marginal_probabilities[:, hi_state], index=r.index)
inhi = smoothed > 0.5
grp = (inhi != inhi.shift()).cumsum()
episodes = [(g.index.min().date(), g.index.max().date(), len(g))
            for _, g in smoothed[inhi].groupby(grp[inhi]) if len(g) >= 20]
print("  high-volatility episodes lasting 20 or more days:")
for a, b, L in episodes:
    print("     %s to %s   (%d days)" % (a, b, L))

# --- 4. Do the EGARCH conclusions depend on the segmentation? ---
print("\n" + "=" * 74)
print("4. EGARCH ROBUSTNESS: POLICY-DATE vs DETECTED SEGMENTATION")
print("=" * 74)
SECTORS = ['Banks', 'Capital_Goods', 'Consumer_Durables_and_Apparel', 'Consumer_Services',
           'Diversified_Financials', 'Energy', 'Food,_Beverage_and_Tobacco',
           'Healthcare_Equipment_and_Services', 'Insurance', 'Materials',
           'Real_Estate_Management_and_Development', 'Retailing', 'Telecommunication_Services']
ASSETS = ['ASPI'] + SECTORS
rets = pd.DataFrame({c: np.log(df[c] / df[c].shift(1)) * 100 for c in ASSETS}).dropna()


def fit_egarch(y, seed=0, n_starts=8):
    """EGARCH(1,1)-t fitted from multiple starts; best plausible fit retained."""
    rng = np.random.default_rng(seed)
    am = arch_model(y, vol='EGARCH', p=1, o=1, q=1, dist='t')
    cands = []
    try:
        cands.append(am.fit(disp='off', options={'maxiter': 2000}))
    except Exception:
        pass
    base = np.array([y.mean(), 0.05, 0.15, -0.05, 0.95, 8.0])
    for _ in range(n_starts):
        sv = base * (1 + rng.normal(0, 0.35, size=6))
        sv[2] = np.clip(abs(sv[2]), 0.02, 0.90)
        sv[4] = np.clip(sv[4], 0.30, 0.995)
        sv[5] = np.clip(abs(sv[5]), 4.0, 20.0)
        try:
            cands.append(am.fit(disp='off', starting_values=sv, options={'maxiter': 2000}))
        except Exception:
            pass
    ok = [c for c in cands if abs(c.params['alpha[1]']) <= 3
          and -0.05 <= c.params['beta[1]'] <= 1.02 and abs(c.params['gamma[1]']) <= 1.5]
    return max(ok if ok else cands, key=lambda c: c.loglikelihood)


crisis_detected = (str(breaks[-2].date()),
                   str((breaks[-1] - pd.Timedelta(days=1)).date()))
schemes = [('Policy milestones', POLICY['Crisis']), ('Detected breaks', crisis_detected)]
rows = []
for name, (a, b) in schemes:
    res = fit_egarch(rets.loc[a:b, 'ASPI'], seed=1)
    n_sig = 0
    for c in ASSETS:
        rr = fit_egarch(rets.loc[a:b, c], seed=abs(hash(c)) % 9999)
        if rr.pvalues['gamma[1]'] < 0.05 and rr.params['gamma[1]'] < 0:
            n_sig += 1
    rows.append({'Segmentation': name,
                 'Crisis window': "%s to %s" % (a, b),
                 'N': len(rets.loc[a:b]),
                 'alpha': round(res.params['alpha[1]'], 3),
                 'beta': round(res.params['beta[1]'], 3),
                 'gamma': round(res.params['gamma[1]'], 3),
                 'gamma_p': round(res.pvalues['gamma[1]'], 4),
                 'sectors_sig_neg_gamma': "%d/14" % n_sig})
    print("  %-20s %s to %s  n=%4d  gamma=%+.3f (p=%.3f)  significant: %d/14"
          % (name, a, b, rows[-1]['N'], rows[-1]['gamma'], rows[-1]['gamma_p'], n_sig))
robust_df = pd.DataFrame(rows)

# --- outputs ---
bic_df.to_csv('break_selection_bic.csv', index=False)
robust_df.to_csv('egarch_segmentation_robustness.csv', index=False)
pd.DataFrame({'break_date': [str(d.date()) for d in breaks]}).to_csv('detected_breaks.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(r.index, r.values, lw=0.6, color='#3b4a6b')
ymax = ax.get_ylim()[1]
for d in breaks:
    ax.axvline(d, color='#c1272d', ls='--', lw=1.6)
    ax.text(d, ymax * 0.95, " " + str(d.date()), color='#c1272d',
            fontsize=8, rotation=90, va='top')
for lab in ['Crisis', 'Recovery']:
    ax.axvline(pd.Timestamp(POLICY[lab][0]), color='#777777', ls=':', lw=1.3)
ax.set_ylabel('ASPI log return (%)')
ax.set_xlabel('Date')
ax.set_title('Estimated variance breaks (dashed) and policy-milestone boundaries (dotted)')
ax.margins(x=0.01)
fig.tight_layout()
fig.savefig('Structural_Breaks_ASPI.png', dpi=300)
print("\nSaved: break_selection_bic.csv, detected_breaks.csv, "
      "egarch_segmentation_robustness.csv, Structural_Breaks_ASPI.png")
