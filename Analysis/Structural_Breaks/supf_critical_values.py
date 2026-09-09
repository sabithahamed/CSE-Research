"""Simulated null distribution for the sup-F variance-break statistic.

Rather than quote published critical values, we simulate the null directly:
under no break, standardised returns are i.i.d., so the log-squared-return
series is i.i.d. draws of log(z^2) with z ~ N(0,1). We compute sup-F over the
same trimmed range used in the application and take empirical quantiles.
"""
import numpy as np

N = 1558          # matches the ASPI return sample
TRIM = 0.15
REPS = 5000
rng = np.random.default_rng(20260908)


def supF(x):
    n = x.size
    lo, hi = int(TRIM * n), int((1 - TRIM) * n)
    S1 = np.concatenate(([0.0], np.cumsum(x)))
    S2 = np.concatenate(([0.0], np.cumsum(x * x)))
    tot = S2[n] - S1[n] ** 2 / n
    i = np.arange(lo, hi)
    ss1 = S2[i] - S1[i] ** 2 / i
    ss2 = (S2[n] - S2[i]) - (S1[n] - S1[i]) ** 2 / (n - i)
    ssr = ss1 + ss2
    F = ((tot - ssr) / 1.0) / (ssr / (n - 2))
    return F.max()


stats = np.empty(REPS)
for b in range(REPS):
    z = rng.standard_normal(N)
    stats[b] = supF(np.log(z * z + 1e-8))

for q in [0.90, 0.95, 0.99]:
    print("  %.0f%% critical value: %.2f" % (q * 100, np.quantile(stats, q)))
print("  observed sup-F in the ASPI data: 124.5")
print("  simulated p-value: %.4f" % float((stats >= 124.5).mean()))
