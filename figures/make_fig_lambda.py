#!/usr/bin/env python3
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT   = '/gpfs/home6/yelkacemi/output'
OUTDIR = os.path.expanduser('~/figs_appendix')
os.makedirs(OUTDIR, exist_ok=True)

s = json.load(open(os.path.join(ROOT, 'legal-bert_scm_seed42',
                                'lambda_sweep_faithfulness.json')))
order = ['baseline', '0.01', '0.05', '0.1', '0.5']
labels = ['Baseline', r'$\lambda$=0.01', r'$\lambda$=0.05',
          r'$\lambda$=0.1', r'$\lambda$=0.5']
suff = [s['aggregated'][k]['k=0.01']['sufficiency'] for k in order]
comp = [s['aggregated'][k]['k=0.01']['comprehensiveness'] for k in order]

print('plotted values (k=1%):')
for l, sf, cp in zip(labels, suff, comp):
    print(f'  {l:14s} suff={sf:.4f}  comp={cp:.4f}')

x = np.arange(len(order))
fig, ax1 = plt.subplots(figsize=(7.2, 4.2))

c_suff = '#2C7A3F'
c_comp = '#2C5A8C'

ax1.plot(x, suff, 'o-', color=c_suff, lw=2, ms=7, label='Sufficiency (left)')
ax1.set_ylabel('Sufficiency', color=c_suff, fontsize=11)
ax1.tick_params(axis='y', labelcolor=c_suff)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

ax2 = ax1.twinx()
ax2.plot(x, comp, 's--', color=c_comp, lw=2, ms=6, label='Comprehensiveness (right)')
ax2.set_ylabel('Comprehensiveness', color=c_comp, fontsize=11)
ax2.tick_params(axis='y', labelcolor=c_comp)

# mark classification optimum lambda=0.1
opt = order.index('0.1')
ax1.annotate('classification\noptimum ($\\lambda$=0.1)',
             xy=(opt, suff[opt]), xytext=(opt-1.4, suff[opt]+0.012),
             fontsize=9, ha='left',
             arrowprops=dict(arrowstyle='->', color='0.4', lw=1))

ax1.set_title('SHAP faithfulness across regularisation strengths '
              '($k=1\\%$, seed 42)', fontsize=11)

lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, lab1+lab2, loc='upper left', frameon=False, fontsize=9)

fig.tight_layout()
png = os.path.join(OUTDIR, 'fig_faithfulness_lambda.png')
pdf = os.path.join(OUTDIR, 'fig_faithfulness_lambda.pdf')
fig.savefig(png, dpi=200, bbox_inches='tight')
fig.savefig(pdf, bbox_inches='tight')
print(f'\nsaved {png}\nsaved {pdf}')
