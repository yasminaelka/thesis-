"""
Aggregate fairness + faithfulness across 5 seeds.
Yasmina El Kacemi - University of Amsterdam

Reads, per seed dir legal-bert_scm_seed{S}:
  - fairness_ci_eod.json        (DPD, DI, EOD, delta-DPD CI per article/axis)
  - contrastive_faithfulness.json (sufficiency, comprehensiveness per k)

Headline aggregation (given group + article imbalance):
  - Fairness metrics aggregated over RELIABLE articles only.
  - Per seed: mean|DPD|, mean EOD (reliable). DI reliable-only, reported as
    supporting (unstable under group imbalance).
  - delta-DPD: per seed, count of reliable articles whose 95% bootstrap CI
    crosses zero (the fairness-null evidence). Reported as mean across seeds.
  - Across seeds: mean +/- std (population std, ddof=0).

  - Faithfulness: mean +/- std across seeds of sufficiency / comprehensiveness
    at each k, baseline vs SCM. (Lower sufficiency = more faithful, ERASER.)

Stats note: with n=5 seeds a Wilcoxon signed-rank test cannot reach p<0.05
(min two-sided p ~ 0.0625), so no significance test is run; mean +/- std and
direction/consistency are reported instead.

Output: console tables + LaTeX blocks written to aggregate_multiseed_latex.txt
        and a machine-readable aggregate_multiseed.json
"""

import os, json
import numpy as np

# Main output folder where all seed result folders are saved
ROOT   = os.path.expanduser('~/output')

# These are the five random seeds used in the experiments
SEEDS  = [42, 13, 7, 21, 100]

# Settings used to find the correct result folders
PAIRS  = 'scm'
ENC    = 'legal-bert'

# The two protected attribute axes that are evaluated
AXES   = ['gender', 'ethnicity']

# The different rationale sizes used for the faithfulness results
KS     = ['k=0.01', 'k=0.05', 'k=0.1']

def seed_dir(s):
    # Build the path to the result folder for one seed
    return os.path.join(ROOT, f'{ENC}_{PAIRS}_seed{s}')

def mean_std(vals):
    # Remove missing values before calculating the mean and standard deviation
    a = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)

    # If there are no valid values, return NaN so the script does not crash
    if a.size == 0:
        return float('nan'), float('nan')

    # Use population standard deviation because the thesis reports variation over these seeds
    return float(a.mean()), float(a.std(ddof=0))

# ---------------------------------------------------------------- FAIRNESS ----
# Per seed, per axis: collect reliable-article values.

# This dictionary stores all fairness values before summarising them over seeds
fair = {ax: {'dpd_base': [], 'dpd_scm': [], 'eod_base': [], 'eod_scm': [],
             'di_base': [], 'di_scm': [],
             'n_reliable_cross_zero': [], 'n_reliable_total': []}
        for ax in AXES}

per_seed_fair = {}   # for the JSON dump

# Go through all seeds one by one
for s in SEEDS:
    # Load the fairness results for this seed
    d = json.load(open(os.path.join(seed_dir(s), 'fairness_ci_eod.json')))

    # Store the per-seed fairness numbers separately for the final JSON file
    per_seed_fair[s] = {}

    # Repeat the same aggregation for gender and ethnicity
    for ax in AXES:
        # Get all article-level fairness results for this axis
        arts = d['results'][ax]['articles']

        # Only keep reliable articles, because unreliable articles should not affect the main result
        rel  = {name: a for name, a in arts.items() if a['reliable']}

        # Collect absolute DPD values for baseline and SCM
        dpd_b = [abs(a['dpd_base']) for a in rel.values() if a['dpd_base'] is not None]
        dpd_s = [abs(a['dpd_scm'])  for a in rel.values() if a['dpd_scm']  is not None]

        # Collect EOD values for baseline and SCM
        eod_b = [a['eod_base'] for a in rel.values() if a['eod_base'] is not None]
        eod_s = [a['eod_scm']  for a in rel.values() if a['eod_scm']  is not None]

        # Collect DI values for baseline and SCM
        di_b  = [a['di_base']  for a in rel.values() if a['di_base']  is not None]
        di_s  = [a['di_scm']   for a in rel.values() if a['di_scm']   is not None]

        # Check whether the bootstrap confidence interval for delta-DPD crosses zero
        cross = [a['delta_dpd_ci_crosses_zero'] for a in rel.values()
                 if a['delta_dpd_ci_crosses_zero'] is not None]

        # Count how many reliable articles have a CI that crosses zero
        n_cross = int(sum(1 for c in cross if c))

        # Count the total number of reliable articles with this CI information
        n_total = len(cross)

        # per-seed reduction over reliable articles

        # Average the article-level fairness metrics for this seed
        m_dpd_b = float(np.mean(dpd_b)) if dpd_b else float('nan')
        m_dpd_s = float(np.mean(dpd_s)) if dpd_s else float('nan')
        m_eod_b = float(np.mean(eod_b)) if eod_b else float('nan')
        m_eod_s = float(np.mean(eod_s)) if eod_s else float('nan')
        m_di_b  = float(np.mean(di_b))  if di_b  else float('nan')
        m_di_s  = float(np.mean(di_s))  if di_s  else float('nan')

        # Save the per-seed averages so they can later be averaged across seeds
        fair[ax]['dpd_base'].append(m_dpd_b)
        fair[ax]['dpd_scm'].append(m_dpd_s)
        fair[ax]['eod_base'].append(m_eod_b)
        fair[ax]['eod_scm'].append(m_eod_s)
        fair[ax]['di_base'].append(m_di_b)
        fair[ax]['di_scm'].append(m_di_s)
        fair[ax]['n_reliable_cross_zero'].append(n_cross)
        fair[ax]['n_reliable_total'].append(n_total)

        # Store readable per-seed results for the JSON output
        per_seed_fair[s][ax] = {
            'mean_abs_dpd_base': m_dpd_b, 'mean_abs_dpd_scm': m_dpd_s,
            'mean_eod_base': m_eod_b, 'mean_eod_scm': m_eod_s,
            'mean_di_base': m_di_b, 'mean_di_scm': m_di_s,
            'reliable_cross_zero': f'{n_cross}/{n_total}',
        }

# ------------------------------------------------------------ FAITHFULNESS ----

# This dictionary stores faithfulness scores for every k value
faith = {k: {'suff_base': [], 'suff_scm': [], 'comp_base': [], 'comp_scm': []} for k in KS}

# This stores the faithfulness information per seed for the JSON output
per_seed_faith = {}

# Go through all seeds and read their faithfulness result files
for s in SEEDS:
    # Load the faithfulness results for this seed
    d = json.load(open(os.path.join(seed_dir(s), 'contrastive_faithfulness.json')))

    # Save lambda because it is useful to report which setting was used
    per_seed_faith[s] = {'lambda': d['lambda']}

    # Store sufficiency and comprehensiveness for every k
    for k in KS:
        faith[k]['suff_base'].append(d['baseline'][k]['sufficiency'])
        faith[k]['suff_scm'].append(d['scm'][k]['sufficiency'])
        faith[k]['comp_base'].append(d['baseline'][k]['comprehensiveness'])
        faith[k]['comp_scm'].append(d['scm'][k]['comprehensiveness'])

# ---------------------------------------------------------------- PRINTING ----
def fmt(m, sd, dp=4):
    # Format mean and standard deviation in the same style for all tables
    return f'{m:.{dp}f} +/- {sd:.{dp}f}'

# Print a clear title for the console output
print('=' * 78)
print(f'MULTI-SEED AGGREGATION  (n={len(SEEDS)} seeds: {SEEDS})')
print('Fairness: reliable articles only. Faithfulness: lower sufficiency = more faithful.')
print('=' * 78)

# Print fairness summary
print('\n--- FAIRNESS (mean over reliable articles, then mean +/- std over seeds) ---')

# Print fairness results separately for gender and ethnicity
for ax in AXES:
    print(f'\n[{ax}]')

    # Print DPD and EOD for baseline and SCM
    for label, kb, ks_ in [('mean|DPD|', 'dpd_base', 'dpd_scm'),
                           ('mean EOD ', 'eod_base', 'eod_scm')]:
        mb, sb = mean_std(fair[ax][kb])
        ms, ss = mean_std(fair[ax][ks_])

        # Delta shows whether SCM is higher or lower than baseline
        print(f'  {label}   baseline {fmt(mb, sb)}   SCM {fmt(ms, ss)}   '
              f'delta {ms - mb:+.4f}')

    # Print DI separately because it is only supporting evidence here
    mb, sb = mean_std(fair[ax]['di_base'])
    ms, ss = mean_std(fair[ax]['di_scm'])
    print(f'  mean DI* baseline {fmt(mb, sb, 3)}   SCM {fmt(ms, ss, 3)}   '
          f'(*supporting; unstable under group imbalance)')

    # Print how often the delta-DPD confidence interval crosses zero
    cz = fair[ax]['n_reliable_cross_zero']
    nt = fair[ax]['n_reliable_total']
    print(f'  dDPD CI crosses zero: per-seed {[f"{a}/{b}" for a,b in zip(cz,nt)]}  '
          f'mean {np.mean(cz):.1f}/{nt[0]} reliable articles')

# Print faithfulness summary
print('\n--- FAITHFULNESS (mean +/- std over seeds) ---')
print(f'{"k":>6} | {"Suff base":>16} {"Suff SCM":>16} | {"Comp base":>16} {"Comp SCM":>16}')
print('-' * 78)

# Print one row per k value
for k in KS:
    sb_m, sb_s = mean_std(faith[k]['suff_base'])
    ss_m, ss_s = mean_std(faith[k]['suff_scm'])
    cb_m, cb_s = mean_std(faith[k]['comp_base'])
    cs_m, cs_s = mean_std(faith[k]['comp_scm'])

    # This row compares baseline and SCM for sufficiency and comprehensiveness
    print(f'{k:>6} | {fmt(sb_m,sb_s):>16} {fmt(ss_m,ss_s):>16} | '
          f'{fmt(cb_m,cb_s):>16} {fmt(cs_m,cs_s):>16}')

print('  (sufficiency RISES under SCM => explanations less faithful)')

# Print lambda values for all seeds so the experiment settings are visible
print(f'\nPer-seed lambda: ' +
      ', '.join(f'seed{s}={per_seed_faith[s]["lambda"]}' for s in SEEDS))

# ---------------------------------------------------------------- LATEX -------

# This list will contain the LaTeX table lines
lines = []

# Start the fairness LaTeX table
lines.append('% --- Fairness (reliable articles, mean +/- std over 5 seeds) ---')
lines.append(r'\begin{tabular}{llcc}')
lines.append(r'\toprule')
lines.append(r'Axis & Metric & Baseline & SCM \\')
lines.append(r'\midrule')

# Add fairness rows for each axis
for ax in AXES:
    mb, sb = mean_std(fair[ax]['dpd_base']); ms, ss = mean_std(fair[ax]['dpd_scm'])
    lines.append(rf'{ax} & mean$|$DPD$|$ & ${mb:.4f}\pm{sb:.4f}$ & ${ms:.4f}\pm{ss:.4f}$ \\')

    mb, sb = mean_std(fair[ax]['eod_base']); ms, ss = mean_std(fair[ax]['eod_scm'])
    lines.append(rf' & mean EOD & ${mb:.4f}\pm{sb:.4f}$ & ${ms:.4f}\pm{ss:.4f}$ \\')

# Finish the fairness LaTeX table
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append('')

# Start the faithfulness LaTeX table
lines.append('% --- Faithfulness (mean +/- std over 5 seeds) ---')
lines.append(r'\begin{tabular}{lcccc}')
lines.append(r'\toprule')
lines.append(r'$k$ & Suff. base & Suff. SCM & Compr. base & Compr. SCM \\')
lines.append(r'\midrule')

# Add faithfulness rows for every k value
for k in KS:
    sb_m, sb_s = mean_std(faith[k]['suff_base']); ss_m, ss_s = mean_std(faith[k]['suff_scm'])
    cb_m, cb_s = mean_std(faith[k]['comp_base']); cs_m, cs_s = mean_std(faith[k]['comp_scm'])

    # Remove "k=" so the LaTeX table only shows the number
    kk = k.replace('k=', '')

    lines.append(rf'{kk} & ${sb_m:.4f}\pm{sb_s:.4f}$ & ${ss_m:.4f}\pm{ss_s:.4f}$ & '
                 rf'${cb_m:.4f}\pm{cb_s:.4f}$ & ${cs_m:.4f}\pm{cs_s:.4f}$ \\')

# Finish the faithfulness LaTeX table
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

# Save the LaTeX tables to a text file
with open('aggregate_multiseed_latex.txt', 'w') as f:
    f.write('\n'.join(lines))

# machine-readable

# Store all important results in a JSON file so they can be reused later
dump = {
    'seeds': SEEDS,
    'per_seed_fairness': per_seed_fair,
    'per_seed_faithfulness': per_seed_faith,
    'fairness_summary': {ax: {
        'mean_abs_dpd_base': mean_std(fair[ax]['dpd_base']),
        'mean_abs_dpd_scm':  mean_std(fair[ax]['dpd_scm']),
        'mean_eod_base':     mean_std(fair[ax]['eod_base']),
        'mean_eod_scm':      mean_std(fair[ax]['eod_scm']),
        'mean_di_base':      mean_std(fair[ax]['di_base']),
        'mean_di_scm':       mean_std(fair[ax]['di_scm']),
        'dDPD_cross_zero_per_seed': list(zip(fair[ax]['n_reliable_cross_zero'],
                                             fair[ax]['n_reliable_total'])),
    } for ax in AXES},
    'faithfulness_summary': {k: {
        'suff_base': mean_std(faith[k]['suff_base']),
        'suff_scm':  mean_std(faith[k]['suff_scm']),
        'comp_base': mean_std(faith[k]['comp_base']),
        'comp_scm':  mean_std(faith[k]['comp_scm']),
    } for k in KS},
}

# Save the machine-readable summary
with open('aggregate_multiseed.json', 'w') as f:
    json.dump(dump, f, indent=2)

# Final message so it is clear which files were created
print('\nWrote: aggregate_multiseed_latex.txt  and  aggregate_multiseed.json')
```
