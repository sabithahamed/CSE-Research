"""Visual summary of the asymmetry result: gamma for every series in every regime.

The point the figure has to make is that significant negative asymmetry is
confined to the sovereign-default regime, and is absent during COVID despite
COVID being a high-volatility period.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

d = pd.read_csv('egarch_1_1_results.csv')
ORDER = ['ASPI', 'Banks', 'Capital_Goods', 'Consumer_Durables_and_Apparel',
         'Consumer_Services', 'Diversified_Financials', 'Energy',
         'Food,_Beverage_and_Tobacco', 'Healthcare_Equipment_and_Services',
         'Insurance', 'Materials', 'Real_Estate_Management_and_Development',
         'Retailing', 'Telecommunication_Services']
NICE = ['ASPI', 'Banks', 'Capital Goods', 'Cons. Durables', 'Cons. Services',
        'Div. Financials', 'Energy', 'Food & Bev.', 'Healthcare', 'Insurance',
        'Materials', 'Real Estate', 'Retailing', 'Telecom']
REG = ['Pre-COVID', 'COVID', 'Crisis', 'Recovery']       # CSV labels
LBL = ['Pre-COVID', 'COVID', 'Default', 'Recovery']      # figure labels

G = np.full((len(ORDER), len(REG)), np.nan)
S = np.zeros_like(G, dtype=bool)
for i, a in enumerate(ORDER):
    for j, r in enumerate(REG):
        row = d[(d.Asset == a) & (d.Period == r)]
        if row.empty:
            continue
        row = row.iloc[0]
        G[i, j] = row['Gamma (Leverage)']
        S[i, j] = (row['Gamma P-Value'] < 0.05) and (row['Gamma (Leverage)'] < 0)

fig, ax = plt.subplots(figsize=(3.4, 4.3))
norm = TwoSlopeNorm(vmin=-0.45, vcenter=0.0, vmax=0.30)
im = ax.imshow(G, cmap='RdBu_r', norm=norm, aspect='auto')

for i in range(len(ORDER)):
    for j in range(len(REG)):
        if S[i, j]:
            ax.text(j, i, '*', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white')

ax.set_xticks(range(len(REG)))
ax.set_xticklabels(LBL, fontsize=7.5, rotation=30, ha='right')
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels(NICE, fontsize=7.5)
ax.set_xticks(np.arange(-.5, len(REG), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(ORDER), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=0.8)
ax.tick_params(which='minor', length=0)

counts = S.sum(axis=0)
for j, c in enumerate(counts):
    ax.text(j, len(ORDER) - 0.35, '%d/14' % c, ha='center', va='top',
            fontsize=7.5, fontweight='bold', transform=ax.get_xaxis_transform())

cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label(r'asymmetry coefficient $\gamma$', fontsize=7.5)
cb.ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig('Asymmetry_Grid.png', dpi=400, bbox_inches='tight')
print("saved Asymmetry_Grid.png")
print("significant negative gamma per regime:", dict(zip(LBL, counts)))
