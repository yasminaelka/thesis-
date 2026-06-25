"""
Fairness evaluation with bootstrap CIs + Equalized Odds, MULTI-SEED version.
Yasmina El Kacemi - University of Amsterdam

Parametrised copy of fairness_ci_eod.py. All metric / bootstrap / threshold
logic is unchanged. Only the I/O is parametrised so it can run per seed:

  python fairness_ci_eod_multiseed.py --pairs scm --seed 42

  - OUTPUT_DIR  = {root}/legal-bert_{pairs}_seed{seed}
  - BEST_LAM    is read per-seed from that dir's contrastive_faithfulness.json
                (same operating point as DPD/DI and faithfulness)
  - models      loaded from that seed dir
  - output      written into that seed dir as fairness_ci_eod.json
    (so a seed array never overwrites another seed)

Group membership uses the SAME whole-word regex matcher and the SAME
'targeted' keyword sets as run_contrastive_v2.py (gender = 40 terms incl.
pronouns, ethnicity = 23 curated minority terms), so EOD is computed on the
same protected groups as the DPD/DI in contrastive_fairness.json.
"""

import os, re, json, argparse, warnings

# Hide warnings so the output is easier to read
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# -- CLI -----------------------------------------------------------------------

# Read command line arguments for the run
ap = argparse.ArgumentParser()
ap.add_argument('--encoder', default='legal-bert')
ap.add_argument('--pairs',   default='scm')
ap.add_argument('--seed',    type=int, default=42)
ap.add_argument('--root',    default='/gpfs/home6/yelkacemi/output')
args = ap.parse_args()

# Use the selected seed and make the run reproducible
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Use GPU if available, otherwise use CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU   : {torch.cuda.get_device_name(0)}')

# -- Per-seed paths ------------------------------------------------------------

# Build the output folder path for this seed
RUN_TAG    = f'{args.encoder}_{args.pairs}_seed{SEED}'
OUTPUT_DIR = os.path.join(args.root, RUN_TAG)

# Stop if the expected seed folder does not exist
assert os.path.isdir(OUTPUT_DIR), f'missing dir {OUTPUT_DIR}'
print(f'Seed: {SEED}  Dir: {OUTPUT_DIR}')

# per-seed best lambda, read from the same JSON the faithfulness run used
BEST_LAM = json.load(
    open(os.path.join(OUTPUT_DIR, 'contrastive_faithfulness.json'))
)['lambda']
print(f'BEST_LAM (per-seed, from contrastive_faithfulness.json) = {BEST_LAM}')

# Main model and evaluation settings
MODEL_NAME    = 'nlpaueb/legal-bert-base-uncased'
N_LABELS      = 10
MAX_LEN       = 512
HEAD, TAIL    = 256, 256
N_BOOT        = 10000
TAU_GRID      = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Article names in the same order as the model output labels
ARTICLE_NAMES = ['Art.2', 'Art.3', 'Art.5', 'Art.6', 'Art.8', 'Art.9',
                 'Art.10', 'Art.11', 'Art.14', 'P1-1']

# Articles that are treated as reliable for the main summaries
RELIABLE      = {'Art.2', 'Art.3', 'Art.5', 'Art.6', 'Art.8', 'Art.10', 'P1-1'}

# EXACT copy of run_contrastive_v2.py 'targeted' active sets.
# Gender targeted = extended (40 terms, female + male referential), pronouns kept.
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

# Ethnicity targeted = 23 curated minority terms (no nationality adjectives).
ETHNICITY_KEYWORDS = [
    'roma', 'romani', 'gypsy', 'kurdish', 'kurd', 'chechen',
    'jewish', 'muslim', 'christian', 'orthodox',
    'asylum', 'refugee', 'immigrant', 'migrant', 'foreigner',
    'minority', 'ethnic', 'ethnicity', 'race', 'racial',
    'indigenous', 'aboriginal', 'caste',
]

# Baseline model path and output path for the lambda sweep results
BASELINE_CKPT = os.path.join(OUTPUT_DIR, 'contrastive_baseline.pt')
OUT_PATH      = os.path.join(OUTPUT_DIR, 'lambda_sweep_fairness.json')

# Check that the baseline checkpoint exists
assert os.path.exists(BASELINE_CKPT), BASELINE_CKPT

# Lambda sweep: evaluate every trained checkpoint against the same baseline.
LAMBDAS = [0.01, 0.05, 0.1, 0.5]

# Build the checkpoint path for every lambda value
LAM_CKPTS = {lam: os.path.join(OUTPUT_DIR, f'contrastive_lam{lam}.pt')
             for lam in LAMBDAS}

# Check that all lambda checkpoints exist
for lam, p in LAM_CKPTS.items():
    assert os.path.exists(p), p

# -- Tokeniser + head/tail truncation ------------------------------------------

# Load the Legal-BERT tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_head_tail(text):
    # Join text parts if the document is stored as a list
    if isinstance(text, list):
        text = ' '.join(text)

    # Tokenize first without truncation, because truncation is done manually
    t = TOKENIZER(text, truncation=False, add_special_tokens=True,
                  return_tensors='pt')
    ids, mask = t['input_ids'][0], t['attention_mask'][0]

    # For long documents, keep the beginning and end of the text
    if len(ids) > MAX_LEN:
        ids  = torch.cat([ids[:HEAD],  ids[-TAIL:]])
        mask = torch.cat([mask[:HEAD], mask[-TAIL:]])

    # Pad shorter documents up to MAX_LEN
    pad = MAX_LEN - len(ids)
    if pad > 0:
        ids  = torch.cat([ids,  torch.zeros(pad, dtype=torch.long)])
        mask = torch.cat([mask, torch.zeros(pad, dtype=torch.long)])

    return ids, mask

# -- Model ---------------------------------------------------------------------

class BERTClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()

        # Load Legal-BERT as the encoder
        self.bert       = AutoModel.from_pretrained(MODEL_NAME)

        # Final layer predicts the article labels
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Run the document through BERT
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the CLS token as document representation
        cls = out.last_hidden_state[:, 0, :]

        return self.classifier(cls)

def load_model(path):
    # Create the model architecture and move it to the device
    m = BERTClassifier(N_LABELS).to(DEVICE)

    # Load the checkpoint
    ckpt = torch.load(path, map_location=DEVICE)

    # Some checkpoints store the weights under model_state_dict
    sd = ckpt.get('model_state_dict', ckpt)

    # Load weights and check for unexpected mismatches
    res = m.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if 'position_ids' not in k]
    unexpected = list(res.unexpected_keys)

    # Stop if there is a real checkpoint mismatch
    if missing or unexpected:
        raise RuntimeError(
            f'state_dict mismatch loading {path}\n'
            f'  missing={missing}\n  unexpected={unexpected}')

    # Set model to evaluation mode
    m.eval()

    return m

# -- Data ----------------------------------------------------------------------

# Load the ECtHR dataset
print('Loading dataset ...')
raw = load_dataset('coastalcph/lex_glue', 'ecthr_a', trust_remote_code=True)

def split_to_arrays(split):
    # Store tokenized inputs, labels, and text
    ids, masks, labels, texts = [], [], [], []

    # Process every document in the split
    for ex in split:
        # Tokenize with head/tail truncation
        i, m = tokenize_head_tail(ex['text'])
        ids.append(i)
        masks.append(m)

        # Create a multi-label vector for the articles
        y = np.zeros(N_LABELS, dtype=np.int64)
        for l in ex['labels']:
            if l < N_LABELS:
                y[l] = 1
        labels.append(y)

        # Store lowercased text for keyword group matching
        texts.append((' '.join(ex['text']) if isinstance(ex['text'], list)
                     else ex['text']).lower())

    return (torch.stack(ids), torch.stack(masks), np.array(labels), texts)

# Prepare validation and test splits
val_ids,  val_mask,  val_y,  _          = split_to_arrays(raw['validation'])
test_ids, test_mask, test_y, test_texts = split_to_arrays(raw['test'])
print(f'Val: {len(val_y)} | Test: {len(test_y)}')

# -- Probabilities -------------------------------------------------------------

@torch.no_grad()
def predict_probs(model, ids, mask, batch=16):
    # Store probabilities for all batches
    probs = []

    # Predict in batches to avoid memory issues
    for i in range(0, len(ids), batch):
        b_ids  = ids[i:i + batch].to(DEVICE)
        b_mask = mask[i:i + batch].to(DEVICE)

        # Convert logits to probabilities
        p = torch.sigmoid(model(b_ids, b_mask)).cpu().numpy()
        probs.append(p)

    return np.vstack(probs)

# -- Per-article threshold tuning on validation (maximise F1) ------------------

def tune_thresholds(val_probs, val_labels):
    # Start with default threshold 0.5 for every label
    thr = np.full(N_LABELS, 0.5)

    # Tune one threshold per article
    for a in range(N_LABELS):
        best_f1, best_t = -1.0, 0.5
        y = val_labels[:, a]

        # Try all thresholds and keep the one with best F1
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
    # Calculate F1 for each article and then average
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

# -- Group membership: EXACT whole-word regex matcher from run_contrastive_v2.py

def build_keyword_pattern(keywords):
    # Escape keywords and make one whole-word regex pattern
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', flags=re.IGNORECASE)

def build_group_mask(texts, keywords):
    # Mark documents as protected when they match one of the keywords
    pattern = build_keyword_pattern(keywords)
    protected = np.zeros(len(texts), dtype=bool)
    for i, txt in enumerate(texts):
        protected[i] = bool(pattern.search(txt))
    return protected

# -- Vectorised fairness metrics -----------------------------------------------

def dpd_di_vec(pred, prot):
    # Split predictions into protected and unprotected groups
    P, U = pred[prot], pred[~prot]

    # Return NaN if one group is empty
    if P.shape[0] == 0 or U.shape[0] == 0:
        nan = np.full(N_LABELS, np.nan)
        return nan, nan

    # Positive prediction rate for both groups
    p_prot, p_unp = P.mean(0), U.mean(0)

    # DPD is the absolute gap between the two groups
    dpd = np.abs(p_prot - p_unp)

    # DI is the ratio between protected and unprotected positive rates
    with np.errstate(divide='ignore', invalid='ignore'):
        di = np.where(p_unp > 0, p_prot / p_unp, np.nan)

    return dpd, di

def eod_vec(pred, y, prot):
    # Convert predictions to float for metric calculation
    pred = pred.astype(float)

    def grp(mask_rows):
        # Select true positives and true negatives for the group
        pos = (y == 1) & mask_rows[:, None]
        neg = (y == 0) & mask_rows[:, None]

        # Calculate true positive rate and false positive rate
        with np.errstate(divide='ignore', invalid='ignore'):
            tpr = (pred * pos).sum(0) / pos.sum(0)
            fpr = (pred * neg).sum(0) / neg.sum(0)

        return tpr, fpr

    # Calculate rates for protected and unprotected groups
    tpr_p, fpr_p = grp(prot)
    tpr_u, fpr_u = grp(~prot)

    # EOD takes the larger gap between TPR and FPR gaps
    d_tpr = np.abs(tpr_p - tpr_u)
    d_fpr = np.abs(fpr_p - fpr_u)
    return np.fmax(d_tpr, d_fpr)

# -- Load models and compute test predictions ---------------------------------

# Load the baseline only once because it is compared to every lambda
print('\nLoading baseline once ...')
base_model = load_model(BASELINE_CKPT)

# Tune baseline thresholds on the validation set
print('Scoring validation for baseline thresholds ...')
val_probs_base = predict_probs(base_model, val_ids, val_mask)
thr_base = tune_thresholds(val_probs_base, val_y)

# Get baseline predictions on the test set
print('Scoring test for baseline ...')
test_probs_base = predict_probs(base_model, test_ids, test_mask)
pred_base = (test_probs_base >= thr_base).astype(int)

# Baseline performance check
f1_base = macro_f1_reliable(pred_base, test_y)
print(f'Baseline macro F1 (all-10 mean): {f1_base:.4f}')

# Protected group axes to evaluate
AXES = {'gender': GENDER_KEYWORDS, 'ethnicity': ETHNICITY_KEYWORDS}

def percentile_ci(arr):
    # Calculate 95% bootstrap percentile confidence interval
    lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
    return lo, hi

def _num(x):
    # Convert NaN to None for cleaner JSON output
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

# Number of test examples
n = len(test_y)

# Store the full lambda sweep results
sweep = {}

# Evaluate every lambda checkpoint
for lam in LAMBDAS:
    print(f'\n========== lambda = {lam} ==========')

    # Load the SCM model for this lambda
    scm_model = load_model(LAM_CKPTS[lam])

    # Tune thresholds for this SCM model
    val_probs_scm = predict_probs(scm_model, val_ids, val_mask)
    thr_scm = tune_thresholds(val_probs_scm, val_y)

    # Predict on the test set
    test_probs_scm = predict_probs(scm_model, test_ids, test_mask)
    pred_scm = (test_probs_scm >= thr_scm).astype(int)

    # Print performance for this lambda
    f1_scm = macro_f1_reliable(pred_scm, test_y)
    print(f'lambda={lam}  macro F1={f1_scm:.4f}  (baseline {f1_base:.4f})')

    results = {}

    # Reset rng to SEED before each lambda so all lambdas share the SAME
    # bootstrap resamples against the SAME baseline -> comparable CIs.
    rng = np.random.default_rng(SEED)

    # Run fairness metrics for gender and ethnicity
    for axis, kw in AXES.items():
        # Build protected/unprotected mask
        prot = build_group_mask(test_texts, kw)
        n_p, n_u = int(prot.sum()), int((~prot).sum())
        print(f'  {axis}: protected={n_p}  unprotected={n_u}')

        # Point estimates for baseline and SCM
        dpd_b0, di_b0 = dpd_di_vec(pred_base, prot)
        dpd_s0, di_s0 = dpd_di_vec(pred_scm,  prot)
        eod_b0 = eod_vec(pred_base, test_y, prot)
        eod_s0 = eod_vec(pred_scm,  test_y, prot)

        # Change in DPD from baseline to SCM
        delta0 = dpd_s0 - dpd_b0

        # Bootstrap arrays for confidence intervals
        boot = {k: np.full((N_BOOT, N_LABELS), np.nan)
                for k in ['dpd_b', 'dpd_s', 'delta', 'di_b', 'di_s', 'eod_b', 'eod_s']}

        # Bootstrap loop
        for b in range(N_BOOT):
            # Sample test examples with replacement
            idx = rng.integers(0, n, size=n)

            # Apply the same resample to groups, predictions, and labels
            prot_b = prot[idx]
            pb, ps, yb = pred_base[idx], pred_scm[idx], test_y[idx]

            # Recalculate fairness metrics on the resampled data
            d_b, i_b = dpd_di_vec(pb, prot_b)
            d_s, i_s = dpd_di_vec(ps, prot_b)

            # Store bootstrap values
            boot['dpd_b'][b], boot['dpd_s'][b] = d_b, d_s
            boot['delta'][b] = d_s - d_b
            boot['di_b'][b],  boot['di_s'][b]  = i_b, i_s
            boot['eod_b'][b] = eod_vec(pb, yb, prot_b)
            boot['eod_s'][b] = eod_vec(ps, yb, prot_b)

        # Convert bootstrap samples into confidence intervals
        ci = {k: percentile_ci(v) for k, v in boot.items()}

        # Store article-level results
        axis_res = {}
        for a, name in enumerate(ARTICLE_NAMES):
            lo, hi = ci['delta'][0][a], ci['delta'][1][a]
            rel = name in RELIABLE

            # Check whether the delta-DPD CI includes zero
            crosses_zero = bool(lo <= 0 <= hi) if not (np.isnan(lo) or np.isnan(hi)) else None

            axis_res[name] = {
                'reliable': bool(rel),
                'n_pos_test': int(test_y[:, a].sum()),
                'dpd_base': _num(dpd_b0[a]),
                'dpd_scm': _num(dpd_s0[a]),
                'delta_dpd': _num(delta0[a]),
                'delta_dpd_ci': [_num(lo), _num(hi)],
                'delta_dpd_ci_crosses_zero': crosses_zero,
                'di_base': _num(di_b0[a]),
                'di_scm': _num(di_s0[a]),
                'eod_base': _num(eod_b0[a]),
                'eod_scm': _num(eod_s0[a]),
            }

        # reliable-only macro means for this axis/lambda
        rel_idx = [i for i, nm in enumerate(ARTICLE_NAMES) if nm in RELIABLE]
        mean_dpd_base = float(np.nanmean([dpd_b0[i] for i in rel_idx]))
        mean_dpd_scm  = float(np.nanmean([dpd_s0[i] for i in rel_idx]))
        mean_eod_base = float(np.nanmean([eod_b0[i] for i in rel_idx]))
        mean_eod_scm  = float(np.nanmean([eod_s0[i] for i in rel_idx]))

        # Save axis-level summary and article-level details
        results[axis] = {
            'n_protected': n_p,
            'n_unprotected': n_u,
            'mean_dpd_reliable_base': mean_dpd_base,
            'mean_dpd_reliable_scm':  mean_dpd_scm,
            'mean_eod_reliable_base': mean_eod_base,
            'mean_eod_reliable_scm':  mean_eod_scm,
            'articles': axis_res,
        }

    # Save everything for this lambda
    sweep[str(lam)] = {
        'lambda': float(lam),
        'macro_f1_all10': {'baseline': f1_base, 'scm': f1_scm},
        'thresholds': {'baseline': thr_base.tolist(), 'scm': thr_scm.tolist()},
        'results': results,
    }

# Metadata so the JSON output is clear later
meta = {
    'seed': SEED,
    'pairs': args.pairs,
    'keyword_set': 'targeted',
    'matching': 'whole_word_regex',
    'n_bootstrap': N_BOOT,
    'reliable_articles': sorted(RELIABLE),
    'lambdas': LAMBDAS,
    'gender_keywords': GENDER_KEYWORDS,
    'ethnicity_keywords': ETHNICITY_KEYWORDS,
}

# Final output object
out = {'meta': meta, 'sweep': sweep}

# Save the lambda sweep fairness results
with open(OUT_PATH, 'w') as f:
    json.dump(out, f, indent=2)

print(f'\nSaved: {OUT_PATH}')
```
