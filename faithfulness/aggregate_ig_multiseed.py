"""
Aggregate the five-seed IG faithfulness runs into a mean +/- std table.

Reads, per seed dir:  contrastive_ig_multiseed.json
Writes:               ~/output/aggregate_ig_multiseed.json
                      ~/output/aggregate_ig_multiseed_latex.txt
"""
import os, json
import numpy as np

# Main folder where all seed output folders are stored
ROOT  = os.path.expanduser('~/output')

# The five seeds used for the multi-seed experiment
SEEDS = [42, 13, 7, 21, 100]

# Model/run settings used to build the seed folder names
ENC, PAIRS = 'legal-bert', 'scm'

# The k values used for IG faithfulness
KS = ['k=0.01', 'k=0.05', 'k=0.1']

def seed_dir(s):
    # Return the folder path for one seed
    return os.path.join(ROOT, f'{ENC}_{PAIRS}_seed{s}')

def mean_std(vals):
    # Calculate mean and population standard deviation over seeds
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))

# Store sufficiency and comprehensiveness scores for baseline and SCM
suff_b = {k: [] for k in KS}; suff_s = {k: [] for k in KS}
comp_b = {k: [] for k in KS}; comp_s = {k: [] for k in KS}

# Keep per-seed values for checking and for the JSON output
per_seed = {}
lams = {}

# Read the IG result file for each seed
for s in SEEDS:
    p = os.path.join(seed_dir(s), 'contrastive_ig_multiseed.json')

    # Skip a seed if the file is missing
    if not os.path.exists(p):
        print(f'  (missing) {p}')
        continue

    d = json.load(open(p))

    # Store the lambda used for this seed
    lams[s] = d['lambda']
    per_seed[s] = {}

    # Collect baseline and SCM scores for each k
    for k in KS:
        suff_b[k].append(d['baseline'][k]['sufficiency'])
        suff_s[k].append(d['scm'][k]['sufficiency'])
        comp_b[k].append(d['baseline'][k]['comprehensiveness'])
        comp_s[k].append(d['scm'][k]['comprehensiveness'])
        per_seed[s][k] = {
            'suff_base': d['baseline'][k]['sufficiency'],
            'suff_scm':  d['scm'][k]['sufficiency'],
        }

# Print a readable console summary
print('=' * 70)
print(f'IG faithfulness, mean +/- std over seeds {list(lams)}')
print('Lower sufficiency = more faithful. SCM raises sufficiency => degradation.')
print('=' * 70)
print(f'{"k":>6} | {"Suff base":>17} {"Suff SCM":>17} {"dSuff":>8}')
print('-' * 60)

# Print mean +/- std for every k
for k in KS:
    mb, sb = mean_std(suff_b[k]); ms, ss = mean_std(suff_s[k])
    print(f'{k:>6} | {mb:.4f} +/- {sb:.4f}  {ms:.4f} +/- {ss:.4f}  {ms-mb:+.4f}')

# Extra check to compare the per-seed SCM values with the earlier single-seed table
print('\nPer-seed SCM sufficiency (sanity vs Table 6, seed 42 ~0.1469/0.1475/0.1454):')
for s in SEEDS:
    if s in per_seed:
        row = '  '.join(f'{per_seed[s][k]["suff_scm"]:.4f}' for k in KS)
        print(f'  seed {s:>3} (lam={lams[s]}): {row}')

# -- LaTeX (drop-in replacement for Table 6) -----------------------------------

# Build the LaTeX table line by line
lines = []
lines.append('% Multi-seed IG faithfulness (replaces single-seed Table 6).')
lines.append('% Lower sufficiency = more faithful; SCM raises sufficiency at every k.')
lines.append(r'\begin{tabular}{lcc}')
lines.append(r'\toprule')
lines.append(r'$k$ & Base Suff. & SCM Suff. \\')
lines.append(r'\midrule')

# Add one LaTeX row per k value
for k in KS:
    mb, sb = mean_std(suff_b[k]); ms, ss = mean_std(suff_s[k])
    kk = k.replace('k=0.01', '1\\%').replace('k=0.05', '5\\%').replace('k=0.1', '10\\%')
    lines.append(rf'{kk} & ${mb:.4f}\pm{sb:.4f}$ & ${ms:.4f}\pm{ss:.4f}$ \\')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

# Save the LaTeX table
with open(os.path.join(ROOT, 'aggregate_ig_multiseed_latex.txt'), 'w') as f:
    f.write('\n'.join(lines))

# Save the same results in a machine-readable JSON format
dump = {
    'seeds': list(lams), 'lambdas': lams,
    'sufficiency': {k: {'base': mean_std(suff_b[k]), 'scm': mean_std(suff_s[k])} for k in KS},
    'comprehensiveness': {k: {'base': mean_std(comp_b[k]), 'scm': mean_std(comp_s[k])} for k in KS},
    'per_seed': per_seed,
}
with open(os.path.join(ROOT, 'aggregate_ig_multiseed.json'), 'w') as f:
    json.dump(dump, f, indent=2)

print('\nWrote aggregate_ig_multiseed.json and aggregate_ig_multiseed_latex.txt')
```
