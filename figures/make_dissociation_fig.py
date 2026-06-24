#!/usr/bin/env python3
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT   = '/gpfs/home6/yelkacemi/output'
OUTDIR = os.path.expanduser('~/figs_appendix')
os.makedirs(OUTDIR, exist_ok=True)

RELIABLE7 = ['Art.2', 'Art.3', 'Art.5', 'Art.6', 'Art.8', 'Art.10', 'P1-1']
SEEDS = [42, 13, 7, 21, 100]
C = {'Baseline': '#4C72B0', 'SCM': '#C44E52', 'Shuffled': '#55A868', 'Word-pair': '#8172B3'}

def mean_dpd7(block, axis):
    arts = block[axis]
    return float(np.mean([arts[a]['DPD'] for a in RELIABLE7 if a in arts]))

def load_perf(cond, s):
    d = json.load(open(os.path.join(ROOT, f'legal-bert_{cond}_seed{s}', 'contrastive_performance.json')))
    return d['baseline']['macro_f1'], d['scm']['macro_f1']

def load_fair(cond, s):
    d = json.load(open(os.path.join(ROOT, f'legal-bert_{cond}_seed{s}', 'contrastive_fairness.json')))
    return {ax: (mean_dpd7(d['baseline'], ax), mean_dpd7(d['scm'], ax)) for ax in ('gender','ethnicity')}

def load_faith(cond, s):
    d = json.load(open(os.path.join(ROOT, f'legal-bert_{cond}_seed{s}', 'contrastive_faithfulness.json')))
    return d['baseline']['k=0.01']['sufficiency'], d['scm']['k=0.01']['sufficiency']

def load_wordpair_fair(s):
    res = {}
    for axis, tag in (('gender','gender'), ('ethnicity','eth')):
        d = json.load(open(os.path.join(ROOT, f'legal-bert_wordpair_{tag}_seed{s}', 'contrastive_fairness.json')))
        res[axis] = (mean_dpd7(d['baseline'], axis), mean_dpd7(d['scm'], axis))
    return res

def ms(vals):
    return float(np.mean(vals)), float(np.std(vals))

print('=== PERFORMANCE (macro F1, all-10) ===')
perf = {}
perf['Baseline'] = ms([load_perf('scm', s)[0] for s in SEEDS])
perf['SCM']      = ms([load_perf('scm', s)[1] for s in SEEDS])
perf['Shuffled'] = ms([load_perf('shuffled', s)[1] for s in SEEDS])
for k,(m,sd) in perf.items(): print(f'  {k:10s} F1 = {m:.4f} +/- {sd:.4f}')

print('\n=== FAIRNESS (mean |DPD| over 7 articles, both axes averaged) ===')
def fair_both(loader, s, which):
    d = loader(s)
    return np.mean([d['gender'][which], d['ethnicity'][which]])
fair = {}
fair['Baseline'] = ms([fair_both(lambda ss: load_fair('scm', ss), s, 0) for s in SEEDS])
fair['SCM']      = ms([fair_both(lambda ss: load_fair('scm', ss), s, 1) for s in SEEDS])
fair['Shuffled'] = ms([fair_both(lambda ss: load_fair('shuffled', ss), s, 1) for s in SEEDS])
fair['Word-pair']= ms([fair_both(load_wordpair_fair, s, 1) for s in SEEDS])
for k,(m,sd) in fair.items(): print(f'  {k:10s} DPD = {m:.4f} +/- {sd:.4f}')

print('\n=== FAITHFULNESS (SHAP sufficiency @ k=1%) ===')
faith = {}
faith['Baseline'] = ms([load_faith('scm', s)[0] for s in SEEDS])
faith['SCM']      = ms([load_faith('scm', s)[1] for s in SEEDS])
faith['Shuffled'] = ms([load_faith('shuffled', s)[1] for s in SEEDS])
for k,(m,sd) in faith.items(): print(f'  {k:10s} suff = {m:.4f} +/- {sd:.4f}')

def panel(ax, data, ylabel, title):
    names = list(data.keys())
    means = [data[n][0] for n in names]; stds = [data[n][1] for n in names]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4, color=[C[n] for n in names], width=0.62)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11)

fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
panel(axs[0], perf,  'Macro F1',                'Performance')
panel(axs[1], fair,  'Mean |DPD| (7 articles)', 'Fairness')
panel(axs[2], faith, 'SHAP sufficiency (k=1%)', 'Faithfulness')
fig.suptitle('Dissociation across interventions: performance and fairness flat, faithfulness degrades under contrastive regularisation', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
png = os.path.join(OUTDIR, 'fig_dissociation_summary.png')
pdf = os.path.join(OUTDIR, 'fig_dissociation_summary.pdf')
fig.savefig(png, dpi=200, bbox_inches='tight')
fig.savefig(pdf, bbox_inches='tight')
print(f'\nsaved {png}\nsaved {pdf}')
