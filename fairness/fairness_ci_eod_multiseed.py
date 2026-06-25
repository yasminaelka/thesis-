"""
Fairness evaluation with bootstrap CIs and Equalized Odds.

This script loads the baseline and SCM models for one seed, computes fairness
metrics, and saves the results in the same seed folder.
"""

import os, re, json, argparse, warnings

# Keep output cleaner
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# -- CLI -----------------------------------------------------------------------

# Read run settings
ap = argparse.ArgumentParser()
ap.add_argument('--encoder', default='legal-bert')
ap.add_argument('--pairs',   default='scm')
ap.add_argument('--seed',    type=int, default=42)
ap.add_argument('--root',    default='/gpfs/home6/yelkacemi/output')
args = ap.parse_args()

# Set seed
SEED = args.seed

# Make results reproducible
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Use GPU if available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# Print GPU name
if DEVICE == 'cuda':
    print(f'GPU   : {torch.cuda.get_device_name(0)}')

# -- Per-seed paths ------------------------------------------------------------

# Output folder for this seed
RUN_TAG    = f'{args.encoder}_{args.pairs}_seed{SEED}'
OUTPUT_DIR = os.path.join(args.root, RUN_TAG)

# Check that the folder exists
assert os.path.isdir(OUTPUT_DIR), f'missing dir {OUTPUT_DIR}'

print(f'Seed: {SEED}  Dir: {OUTPUT_DIR}')

# Read best lambda for this seed
BEST_LAM = json.load(
    open(os.path.join(OUTPUT_DIR, 'contrastive_faithfulness.json'))
)['lambda']

print(f'BEST_LAM (per-seed, from contrastive_faithfulness.json) = {BEST_LAM}')

# Main settings
MODEL_NAME    = 'nlpaueb/legal-bert-base-uncased'
N_LABELS      = 10
MAX_LEN       = 512
HEAD, TAIL    = 256, 256
N_BOOT        = 10000
TAU_GRID      = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Article labels
ARTICLE_NAMES = ['Art.2', 'Art.3', 'Art.5', 'Art.6', 'Art.8', 'Art.9',
                 'Art.10', 'Art.11', 'Art.14', 'P1-1']

# Reliable articles
RELIABLE      = {'Art.2', 'Art.3', 'Art.5', 'Art.6', 'Art.8', 'Art.10', 'P1-1'}

# Gender keywords
GENDER_KEYWORDS = [
    # female-referential
    'woman', 'women', 'female', 'girl', 'mother', 'wife', 'daughter',
    'sister', 'she', 'her', 'hers', 'lady', 'bride', 'girlfriend',
    'stepmother', 'grandmother', 'schoolgirl', 'mommy', 'aunt', 'niece',
    # male-referential
    'man', 'men', 'male', 'boy', 'father', 'husband', 'son', 'brother',
    'he', 'him', 'his', 'gentleman', 'groom', 'boyfriend', 'stepfather',
    'grandfather', 'schoolboy', 'daddy', 'uncle', 'nephew',
]

# Ethnicity keywords
ETHNICITY_KEYWORDS = [
    'roma', 'romani', 'gypsy', 'kurdish', 'kurd', 'chechen',
    'jewish', 'muslim', 'christian', 'orthodox',
    'asylum', 'refugee', 'immigrant', 'migrant', 'foreigner',
    'minority', 'ethnic', 'ethnicity', 'race', 'racial',
    'indigenous', 'aboriginal', 'caste',
]

# Model and output paths
BASELINE_CKPT = os.path.join(OUTPUT_DIR, 'contrastive_baseline.pt')
SCM_CKPT      = os.path.join(OUTPUT_DIR, f'contrastive_lam{BEST_LAM}.pt')
OUT_PATH      = os.path.join(OUTPUT_DIR, 'fairness_ci_eod.json')

# Check model files
assert os.path.exists(BASELINE_CKPT), BASELINE_CKPT
assert os.path.exists(SCM_CKPT), SCM_CKPT

# -- Tokeniser + head/tail truncation ------------------------------------------

# Load tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_head_tail(text):
    # Join text parts if needed
    if isinstance(text, list):
        text = ' '.join(text)

    # Tokenize first, then truncate manually
    t = TOKENIZER(text, truncation=False, add_special_tokens=True,
                  return_tensors='pt')
    ids, mask = t['input_ids'][0], t['attention_mask'][0]

    # Keep start and end for long documents
    if len(ids) > MAX_LEN:
        ids  = torch.cat([ids[:HEAD],  ids[-TAIL:]])
        mask = torch.cat([mask[:HEAD], mask[-TAIL:]])

    # Pad shorter documents
    pad = MAX_LEN - len(ids)
    if pad > 0:
        ids  = torch.cat([ids,  torch.zeros(pad, dtype=torch.long)])
        mask = torch.cat([mask, torch.zeros(pad, dtype=torch.long)])

    return ids, mask

# -- Model ---------------------------------------------------------------------

class BERTClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()

        # Encoder and classifier head
        self.bert       = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Forward pass
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS representation
        cls = out.last_hidden_state[:, 0, :]

        return self.classifier(cls)

def load_model(path):
    # Build model
    m = BERTClassifier(N_LABELS).to(DEVICE)

    # Load checkpoint
    ckpt = torch.load(path, map_location=DEVICE)

    # Get model weights
    sd = ckpt.get('model_state_dict', ckpt)

    # Load weights
    m.load_state_dict(sd, strict=False)

    # Evaluation mode
    m.eval()

    return m

# -- Data ----------------------------------------------------------------------

# Load dataset
print('Loading dataset ...')
raw = load_dataset('coastalcph/lex_glue', 'ecthr_a', trust_remote_code=True)

def split_to_arrays(split):
    # Store inputs, labels, and text
    ids, masks, labels, texts = [], [], [], []

    for ex in split:
        # Tokenize document
        i, m = tokenize_head_tail(ex['text'])
        ids.append(i)
        masks.append(m)

        # Make multi-label vector
        y = np.zeros(N_LABELS, dtype=np.int64)

        for l in ex['labels']:
            if l < N_LABELS:
                y[l] = 1

        labels.append(y)

        # Store text for keyword matching
        texts.append((' '.join(ex['text']) if isinstance(ex['text'], list)
                     else ex['text']).lower())

    return (torch.stack(ids), torch.stack(masks), np.array(labels), texts)

# Prepare validation and test data
val_ids,  val_mask,  val_y,  _          = split_to_arrays(raw['validation'])
test_ids, test_mask, test_y, test_texts = split_to_arrays(raw['test'])

print(f'Val: {len(val_y)} | Test: {len(test_y)}')

# -- Probabilities -------------------------------------------------------------

@torch.no_grad()
def predict_probs(model, ids, mask, batch=16):
    # Store probabilities
    probs = []

    # Predict in batches
    for i in range(0, len(ids), batch):
        b_ids  = ids[i:i + batch].to(DEVICE)
        b_mask = mask[i:i + batch].to(DEVICE)

        # Convert logits to probabilities
        p = torch.sigmoid(model(b_ids, b_mask)).cpu().numpy()
        probs.append(p)

    return np.vstack(probs)

# -- Per-article threshold tuning on validation --------------------------------

def tune_thresholds(val_probs, val_labels):
    # Start with default thresholds
    thr = np.full(N_LABELS, 0.5)

    # Tune one threshold per article
    for a in range(N_LABELS):
        best_f1, best_t = -1.0, 0.5
        y = val_labels[:, a]

        for t in TAU_GRID:
            pred = (val_probs[:, a] >= t).astype(int)

            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())

            prec = tp / (tp + fp) if tp + fp else 0.0
            rec  = tp / (tp + fn) if tp + fn else 0.0
            f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

            if f1 > best_f1:
                best_f1, best_t = f1, t

        thr[a] = best_t

    return thr

def macro_f1_reliable(pred, y):
    # Calculate average F1
    f1s = []

    for a, name in enumerate(ARTICLE_NAMES):
        p, yy = pred[:, a], y[:, a]

        tp = int(((p == 1) & (yy == 1)).sum())
        fp = int(((p == 1) & (yy == 0)).sum())
        fn = int(((p == 0) & (yy == 1)).sum())

        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    return float(np.mean(f1s))

# -- Group membership ----------------------------------------------------------

def build_keyword_pattern(keywords):
    # Build one whole-word regex
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', flags=re.IGNORECASE)

def build_group_mask(texts, keywords):
    # Mark protected documents
    pattern = build_keyword_pattern(keywords)
    protected = np.zeros(len(texts), dtype=bool)

    for i, txt in enumerate(texts):
        protected[i] = bool(pattern.search(txt))

    return protected

# -- Vectorised fairness metrics -----------------------------------------------

def dpd_di_vec(pred, prot):
    # Split groups
    P, U = pred[prot], pred[~prot]

    # Return NaN if a group is empty
    if P.shape[0] == 0 or U.shape[0] == 0:
        nan = np.full(N_LABELS, np.nan)
        return nan, nan

    # Positive rates
    p_prot, p_unp = P.mean(0), U.mean(0)

    # DPD
    dpd = np.abs(p_prot - p_unp)

    # DI
    with np.errstate(divide='ignore', invalid='ignore'):
        di = np.where(p_unp > 0, p_prot / p_unp, np.nan)

    return dpd, di

def eod_vec(pred, y, prot):
    # Use float predictions for calculations
    pred = pred.astype(float)

    def grp(mask_rows):
        # Select group rows
        pos = (y == 1) & mask_rows[:, None]
        neg = (y == 0) & mask_rows[:, None]

        # TPR and FPR
        with np.errstate(divide='ignore', invalid='ignore'):
            tpr = (pred * pos).sum(0) / pos.sum(0)
            fpr = (pred * neg).sum(0) / neg.sum(0)

        return tpr, fpr

    # Compare groups
    tpr_p, fpr_p = grp(prot)
    tpr_u, fpr_u = grp(~prot)

    # EOD
    d_tpr = np.abs(tpr_p - tpr_u)
    d_fpr = np.abs(fpr_p - fpr_u)

    return np.fmax(d_tpr, d_fpr)

# -- Load models and compute test predictions ---------------------------------

# Load models
print('\nLoading models ...')
base_model = load_model(BASELINE_CKPT)
scm_model  = load_model(SCM_CKPT)

# Validation predictions for thresholds
print('Scoring validation (for thresholds) ...')
val_probs_base = predict_probs(base_model, val_ids, val_mask)
val_probs_scm  = predict_probs(scm_model,  val_ids, val_mask)

# Tune thresholds
thr_base = tune_thresholds(val_probs_base, val_y)
thr_scm  = tune_thresholds(val_probs_scm,  val_y)

# Test predictions
print('Scoring test ...')
test_probs_base = predict_probs(base_model, test_ids, test_mask)
test_probs_scm  = predict_probs(scm_model,  test_ids, test_mask)

# Binary predictions
pred_base = (test_probs_base >= thr_base).astype(int)
pred_scm  = (test_probs_scm  >= thr_scm ).astype(int)

# Quick F1 check
f1_base = macro_f1_reliable(pred_base, test_y)
f1_scm  = macro_f1_reliable(pred_scm,  test_y)
print(f'\nSanity: macro F1  baseline={f1_base:.4f}  SCM={f1_scm:.4f}')

# Protected-group axes
AXES = {'gender': GENDER_KEYWORDS, 'ethnicity': ETHNICITY_KEYWORDS}

# -- Point estimates + bootstrap CIs per axis ----------------------------------

def percentile_ci(arr):
    # 95% bootstrap CI
    lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
    return lo, hi

def _num(x):
    # Convert NaN to None for JSON
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

# Store final results
results = {}

# Random generator for bootstrap
rng = np.random.default_rng(SEED)

# Number of test examples
n = len(test_y)

# Run fairness evaluation for each axis
for axis, kw in AXES.items():
    # Protected mask
    prot = build_group_mask(test_texts, kw)

    # Group sizes
    n_p, n_u = int(prot.sum()), int((~prot).sum())
    print(f'\n=== {axis} ===   protected={n_p}  unprotected={n_u}')

    # Point estimates
    dpd_b0, di_b0 = dpd_di_vec(pred_base, prot)
    dpd_s0, di_s0 = dpd_di_vec(pred_scm,  prot)
    eod_b0 = eod_vec(pred_base, test_y, prot)
    eod_s0 = eod_vec(pred_scm,  test_y, prot)

    # DPD change
    delta0 = dpd_s0 - dpd_b0

    # Bootstrap storage
    boot = {k: np.full((N_BOOT, N_LABELS), np.nan)
            for k in ['dpd_b', 'dpd_s', 'delta', 'di_b', 'di_s', 'eod_b', 'eod_s']}

    # Bootstrap loop
    for b in range(N_BOOT):
        # Sample test examples
        idx = rng.integers(0, n, size=n)

        # Apply sample
        prot_b = prot[idx]
        pb, ps, yb = pred_base[idx], pred_scm[idx], test_y[idx]

        # Recompute metrics
        d_b, i_b = dpd_di_vec(pb, prot_b)
        d_s, i_s = dpd_di_vec(ps, prot_b)

        # Store values
        boot['dpd_b'][b], boot['dpd_s'][b] = d_b, d_s
        boot['delta'][b] = d_s - d_b
        boot['di_b'][b],  boot['di_s'][b]  = i_b, i_s
        boot['eod_b'][b] = eod_vec(pb, yb, prot_b)
        boot['eod_s'][b] = eod_vec(ps, yb, prot_b)

    # Confidence intervals
    ci = {k: percentile_ci(v) for k, v in boot.items()}

    # Article-level results
    axis_res = {}

    print(f'{"Article":8s} {"DPD_base":>9s} {"DPD_scm":>9s} '
          f'{"dDPD":>8s} {"dDPD 95% CI":>20s} {"Reliable"}')
    print('-' * 70)

    # Save and print article results
    for a, name in enumerate(ARTICLE_NAMES):
        lo, hi = ci['delta'][0][a], ci['delta'][1][a]
        rel = name in RELIABLE
        crosses_zero = bool(lo <= 0 <= hi) if not (np.isnan(lo) or np.isnan(hi)) else None

        axis_res[name] = {
            'reliable': bool(rel),
            'n_pos_test': int(test_y[:, a].sum()),
            'dpd_base': _num(dpd_b0[a]),
            'dpd_base_ci': [_num(ci['dpd_b'][0][a]), _num(ci['dpd_b'][1][a])],
            'dpd_scm': _num(dpd_s0[a]),
            'dpd_scm_ci': [_num(ci['dpd_s'][0][a]), _num(ci['dpd_s'][1][a])],
            'delta_dpd': _num(delta0[a]),
            'delta_dpd_ci': [_num(lo), _num(hi)],
            'delta_dpd_ci_crosses_zero': crosses_zero,
            'di_base': _num(di_b0[a]),
            'di_base_ci': [_num(ci['di_b'][0][a]), _num(ci['di_b'][1][a])],
            'di_scm': _num(di_s0[a]),
            'di_scm_ci': [_num(ci['di_s'][0][a]), _num(ci['di_s'][1][a])],
            'eod_base': _num(eod_b0[a]),
            'eod_base_ci': [_num(ci['eod_b'][0][a]), _num(ci['eod_b'][1][a])],
            'eod_scm': _num(eod_s0[a]),
            'eod_scm_ci': [_num(ci['eod_s'][0][a]), _num(ci['eod_s'][1][a])],
        }

        ci_str = f'[{lo:+.4f}, {hi:+.4f}]' if not np.isnan(lo) else 'n/a'

        print(f'{name:8s} {dpd_b0[a]:9.4f} {dpd_s0[a]:9.4f} '
              f'{delta0[a]:+8.4f} {ci_str:>20s} {"yes" if rel else "no"}')

    # Store axis results
    results[axis] = {
        'n_protected': n_p,
        'n_unprotected': n_u,
        'articles': axis_res,
    }

# Metadata
meta = {
    'seed': SEED,
    'pairs': args.pairs,
    'keyword_set': 'targeted',
    'matching': 'whole_word_regex',
    'n_bootstrap': N_BOOT,
    'lambda': float(BEST_LAM),
    'macro_f1_reliable': {'baseline': f1_base, 'scm': f1_scm},
    'thresholds': {'baseline': thr_base.tolist(), 'scm': thr_scm.tolist()},
    'gender_keywords': GENDER_KEYWORDS,
    'ethnicity_keywords': ETHNICITY_KEYWORDS,
}

# Final output
out = {'meta': meta, 'results': results}

# Save results
with open(OUT_PATH, 'w') as f:
    json.dump(out, f, indent=2)

print(f'\nSaved: {OUT_PATH}')
