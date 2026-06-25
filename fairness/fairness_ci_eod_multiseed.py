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

# Ignore warning messages so the output stays cleaner
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# -- CLI -----------------------------------------------------------------------

# Read command line arguments so the same script can run for different seeds
ap = argparse.ArgumentParser()
ap.add_argument('--encoder', default='legal-bert')
ap.add_argument('--pairs',   default='scm')
ap.add_argument('--seed',    type=int, default=42)
ap.add_argument('--root',    default='/gpfs/home6/yelkacemi/output')
args = ap.parse_args()

# Use the seed from the command line
SEED = args.seed

# Set random seeds to make the bootstrap and model evaluation reproducible
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Use GPU if available, otherwise use CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# Print GPU name when CUDA is used
if DEVICE == 'cuda':
    print(f'GPU   : {torch.cuda.get_device_name(0)}')

# -- Per-seed paths ------------------------------------------------------------

# Create the folder name for this specific run
RUN_TAG    = f'{args.encoder}_{args.pairs}_seed{SEED}'

# Full output directory for this seed
OUTPUT_DIR = os.path.join(args.root, RUN_TAG)

# Stop the script if the expected seed folder does not exist
assert os.path.isdir(OUTPUT_DIR), f'missing dir {OUTPUT_DIR}'

# Print which seed and folder are being used
print(f'Seed: {SEED}  Dir: {OUTPUT_DIR}')

# per-seed best lambda, read from the same JSON the faithfulness run used

# Read the best lambda value for this seed
BEST_LAM = json.load(
    open(os.path.join(OUTPUT_DIR, 'contrastive_faithfulness.json'))
)['lambda']

# Print lambda to check that the correct SCM model will be loaded
print(f'BEST_LAM (per-seed, from contrastive_faithfulness.json) = {BEST_LAM}')

# Model and evaluation settings
MODEL_NAME    = 'nlpaueb/legal-bert-base-uncased'
N_LABELS      = 10
MAX_LEN       = 512

# Head/tail truncation keeps the start and end of long documents
HEAD, TAIL    = 256, 256

# Number of bootstrap samples used for confidence intervals
N_BOOT        = 10000

# Threshold grid used for tuning the prediction threshold per article
TAU_GRID      = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Article labels in the same order as the model outputs
ARTICLE_NAMES = ['Art.2', 'Art.3', 'Art.5', 'Art.6', 'Art.8', 'Art.9',
                 'Art.10', 'Art.11', 'Art.14', 'P1-1']

# Articles considered reliable enough for the main fairness summary
RELIABLE      = {'Art.2', 'Art.3', 'Art.5', 'Art.6', 'Art.8', 'Art.10', 'P1-1'}

# EXACT copy of run_contrastive_v2.py 'targeted' active sets.
# Gender targeted = extended (40 terms, female + male referential), pronouns kept.

# Gender keywords used to identify protected-group documents
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

# Ethnicity keywords used to identify protected-group documents
ETHNICITY_KEYWORDS = [
    'roma', 'romani', 'gypsy', 'kurdish', 'kurd', 'chechen',
    'jewish', 'muslim', 'christian', 'orthodox',
    'asylum', 'refugee', 'immigrant', 'migrant', 'foreigner',
    'minority', 'ethnic', 'ethnicity', 'race', 'racial',
    'indigenous', 'aboriginal', 'caste',
]

# Paths to the baseline model, SCM model, and output JSON file
BASELINE_CKPT = os.path.join(OUTPUT_DIR, 'contrastive_baseline.pt')
SCM_CKPT      = os.path.join(OUTPUT_DIR, f'contrastive_lam{BEST_LAM}.pt')
OUT_PATH      = os.path.join(OUTPUT_DIR, 'fairness_ci_eod.json')

# Check that both model checkpoints exist before continuing
assert os.path.exists(BASELINE_CKPT), BASELINE_CKPT
assert os.path.exists(SCM_CKPT), SCM_CKPT

# -- Tokeniser + head/tail truncation ------------------------------------------

# Load the tokenizer that belongs to Legal-BERT
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_head_tail(text):
    # If the text is split into parts, join it into one string
    if isinstance(text, list):
        text = ' '.join(text)

    # Tokenize without truncation first, because custom head/tail truncation is used
    t = TOKENIZER(text, truncation=False, add_special_tokens=True,
                  return_tensors='pt')
    ids, mask = t['input_ids'][0], t['attention_mask'][0]

    # If the document is too long, keep the first HEAD and last TAIL tokens
    if len(ids) > MAX_LEN:
        ids  = torch.cat([ids[:HEAD],  ids[-TAIL:]])
        mask = torch.cat([mask[:HEAD], mask[-TAIL:]])

    # Pad shorter documents up to MAX_LEN
    pad = MAX_LEN - len(ids)
    if pad > 0:
        ids  = torch.cat([ids,  torch.zeros(pad, dtype=torch.long)])
        mask = torch.cat([mask, torch.zeros(pad, dtype=torch.long)])

    # Return input IDs and attention mask
    return ids, mask

# -- Model ---------------------------------------------------------------------

class BERTClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()

        # Load the Legal-BERT encoder
        self.bert       = AutoModel.from_pretrained(MODEL_NAME)

        # Final classification layer for the article labels
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Run the input through BERT
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the CLS token representation for classification
        cls = out.last_hidden_state[:, 0, :]

        # Return raw logits for all article labels
        return self.classifier(cls)

def load_model(path):
    # Create the model architecture
    m = BERTClassifier(N_LABELS).to(DEVICE)

    # Load the saved checkpoint
    ckpt = torch.load(path, map_location=DEVICE)

    # Get the actual model weights from the checkpoint
    sd = ckpt.get('model_state_dict', ckpt)

    # Load the weights into the model
    m.load_state_dict(sd, strict=False)

    # Put the model in evaluation mode
    m.eval()

    return m

# -- Data ----------------------------------------------------------------------

# Load the ECtHR dataset
print('Loading dataset ...')
raw = load_dataset('coastalcph/lex_glue', 'ecthr_a', trust_remote_code=True)

def split_to_arrays(split):
    # Lists for tokenized inputs, labels, and original texts
    ids, masks, labels, texts = [], [], [], []

    # Process every example in the split
    for ex in split:
        # Tokenize the text with head/tail truncation
        i, m = tokenize_head_tail(ex['text'])
        ids.append(i)
        masks.append(m)

        # Create a multi-label vector for the ten articles
        y = np.zeros(N_LABELS, dtype=np.int64)

        # Mark labels that are present in this example
        for l in ex['labels']:
            if l < N_LABELS:
                y[l] = 1

        # Store the label vector
        labels.append(y)

        # Store lowercased text for keyword matching later
        texts.append((' '.join(ex['text']) if isinstance(ex['text'], list)
                     else ex['text']).lower())

    # Return arrays/tensors needed for scoring and fairness metrics
    return (torch.stack(ids), torch.stack(masks), np.array(labels), texts)

# Convert validation and test splits into tensors and arrays
val_ids,  val_mask,  val_y,  _          = split_to_arrays(raw['validation'])
test_ids, test_mask, test_y, test_texts = split_to_arrays(raw['test'])

# Print the number of validation and test examples
print(f'Val: {len(val_y)} | Test: {len(test_y)}')

# -- Probabilities -------------------------------------------------------------

@torch.no_grad()
def predict_probs(model, ids, mask, batch=16):
    # Store predicted probabilities for all batches
    probs = []

    # Score the data in small batches to avoid GPU memory issues
    for i in range(0, len(ids), batch):
        b_ids  = ids[i:i + batch].to(DEVICE)
        b_mask = mask[i:i + batch].to(DEVICE)

        # Convert logits to probabilities using sigmoid
        p = torch.sigmoid(model(b_ids, b_mask)).cpu().numpy()
        probs.append(p)

    # Combine all batches into one array
    return np.vstack(probs)

# -- Per-article threshold tuning on validation (maximise F1) ------------------

def tune_thresholds(val_probs, val_labels):
    # Start with default threshold 0.5 for all labels
    thr = np.full(N_LABELS, 0.5)

    # Tune a separate threshold for each article
    for a in range(N_LABELS):
        best_f1, best_t = -1.0, 0.5
        y = val_labels[:, a]

        # Try all thresholds in the grid
        for t in TAU_GRID:
            pred = (val_probs[:, a] >= t).astype(int)

            # Calculate true positives, false positives, and false negatives
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())

            # Calculate precision, recall, and F1 safely
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec  = tp / (tp + fn) if tp + fn else 0.0
            f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

            # Keep the threshold with the best F1
            if f1 > best_f1:
                best_f1, best_t = f1, t

        # Save the best threshold for this article
        thr[a] = best_t

    return thr

def macro_f1_reliable(pred, y):
    # Store F1 scores for all articles
    f1s = []

    # Calculate F1 for each article separately
    for a, name in enumerate(ARTICLE_NAMES):
        p, yy = pred[:, a], y[:, a]

        # Count true positives, false positives, and false negatives
        tp = int(((p == 1) & (yy == 1)).sum())
        fp = int(((p == 1) & (yy == 0)).sum())
        fn = int(((p == 0) & (yy == 1)).sum())

        # Calculate precision, recall, and F1 safely
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    # Return the average F1 over articles
    return float(np.mean(f1s))

# -- Group membership: EXACT whole-word regex matcher from run_contrastive_v2.py

def build_keyword_pattern(keywords):
    # Escape keywords so they are treated as normal text in the regex
    escaped = [re.escape(kw) for kw in keywords]

    # Build a whole-word regex pattern
    return re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', flags=re.IGNORECASE)

def build_group_mask(texts, keywords):
    # Build the regex pattern for this protected-group keyword list
    pattern = build_keyword_pattern(keywords)

    # Start with all documents marked as not protected
    protected = np.zeros(len(texts), dtype=bool)

    # Mark a document as protected if it contains at least one keyword
    for i, txt in enumerate(texts):
        protected[i] = bool(pattern.search(txt))

    return protected

# -- Vectorised fairness metrics -----------------------------------------------

def dpd_di_vec(pred, prot):
    # Split predictions into protected and unprotected groups
    P, U = pred[prot], pred[~prot]

    # If one group is empty, the metric cannot be calculated
    if P.shape[0] == 0 or U.shape[0] == 0:
        nan = np.full(N_LABELS, np.nan)
        return nan, nan

    # Calculate positive prediction rates per group
    p_prot, p_unp = P.mean(0), U.mean(0)

    # DPD is the absolute difference in positive prediction rates
    dpd = np.abs(p_prot - p_unp)

    # DI is the protected rate divided by the unprotected rate
    with np.errstate(divide='ignore', invalid='ignore'):
        di = np.where(p_unp > 0, p_prot / p_unp, np.nan)

    return dpd, di

def eod_vec(pred, y, prot):
    # Convert predictions to float for calculations
    pred = pred.astype(float)

    def grp(mask_rows):
        # Select positive and negative true-label cases for one group
        pos = (y == 1) & mask_rows[:, None]
        neg = (y == 0) & mask_rows[:, None]

        # Calculate TPR and FPR for each article
        with np.errstate(divide='ignore', invalid='ignore'):
            tpr = (pred * pos).sum(0) / pos.sum(0)
            fpr = (pred * neg).sum(0) / neg.sum(0)

        return tpr, fpr

    # Calculate TPR and FPR for protected and unprotected groups
    tpr_p, fpr_p = grp(prot)
    tpr_u, fpr_u = grp(~prot)

    # Equalized odds uses the larger difference between TPR gap and FPR gap
    d_tpr = np.abs(tpr_p - tpr_u)
    d_fpr = np.abs(fpr_p - fpr_u)

    return np.fmax(d_tpr, d_fpr)

# -- Load models and compute test predictions ---------------------------------

# Load both trained models
print('\nLoading models ...')
base_model = load_model(BASELINE_CKPT)
scm_model  = load_model(SCM_CKPT)

# Score validation data to tune thresholds
print('Scoring validation (for thresholds) ...')
val_probs_base = predict_probs(base_model, val_ids, val_mask)
val_probs_scm  = predict_probs(scm_model,  val_ids, val_mask)

# Tune thresholds separately for baseline and SCM
thr_base = tune_thresholds(val_probs_base, val_y)
thr_scm  = tune_thresholds(val_probs_scm,  val_y)

# Score the test set
print('Scoring test ...')
test_probs_base = predict_probs(base_model, test_ids, test_mask)
test_probs_scm  = predict_probs(scm_model,  test_ids, test_mask)

# Convert probabilities into binary predictions using tuned thresholds
pred_base = (test_probs_base >= thr_base).astype(int)
pred_scm  = (test_probs_scm  >= thr_scm ).astype(int)

# Calculate a sanity-check macro F1 score
f1_base = macro_f1_reliable(pred_base, test_y)
f1_scm  = macro_f1_reliable(pred_scm,  test_y)
print(f'\nSanity: macro F1  baseline={f1_base:.4f}  SCM={f1_scm:.4f}')

# Protected-group axes and their keyword lists
AXES = {'gender': GENDER_KEYWORDS, 'ethnicity': ETHNICITY_KEYWORDS}

# -- Point estimates + bootstrap CIs per axis ----------------------------------

def percentile_ci(arr):
    # Calculate the 95% percentile bootstrap confidence interval
    lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
    return lo, hi

def _num(x):
    # Convert NaN values to None so the JSON output is valid and readable
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

# Dictionary for all final fairness results
results = {}

# Random generator for bootstrap sampling
rng = np.random.default_rng(SEED)

# Number of test examples
n = len(test_y)

# Run the fairness evaluation for each protected axis
for axis, kw in AXES.items():
    # Build protected/unprotected group mask for this axis
    prot = build_group_mask(test_texts, kw)

    # Count protected and unprotected documents
    n_p, n_u = int(prot.sum()), int((~prot).sum())
    print(f'\n=== {axis} ===   protected={n_p}  unprotected={n_u}')

    # Calculate point estimates for DPD and DI
    dpd_b0, di_b0 = dpd_di_vec(pred_base, prot)
    dpd_s0, di_s0 = dpd_di_vec(pred_scm,  prot)

    # Calculate point estimates for Equalized Odds
    eod_b0 = eod_vec(pred_base, test_y, prot)
    eod_s0 = eod_vec(pred_scm,  test_y, prot)

    # Delta shows how SCM changes DPD compared to baseline
    delta0 = dpd_s0 - dpd_b0

    # Create arrays for bootstrap results
    boot = {k: np.full((N_BOOT, N_LABELS), np.nan)
            for k in ['dpd_b', 'dpd_s', 'delta', 'di_b', 'di_s', 'eod_b', 'eod_s']}

    # Bootstrap loop for confidence intervals
    for b in range(N_BOOT):
        # Sample test examples with replacement
        idx = rng.integers(0, n, size=n)

        # Apply the same sampled indices to groups, predictions, and labels
        prot_b = prot[idx]
        pb, ps, yb = pred_base[idx], pred_scm[idx], test_y[idx]

        # Recalculate DPD and DI on the bootstrap sample
        d_b, i_b = dpd_di_vec(pb, prot_b)
        d_s, i_s = dpd_di_vec(ps, prot_b)

        # Store bootstrap fairness values
        boot['dpd_b'][b], boot['dpd_s'][b] = d_b, d_s
        boot['delta'][b] = d_s - d_b
        boot['di_b'][b],  boot['di_s'][b]  = i_b, i_s
        boot['eod_b'][b] = eod_vec(pb, yb, prot_b)
        boot['eod_s'][b] = eod_vec(ps, yb, prot_b)

    # Convert bootstrap samples into 95% confidence intervals
    ci = {k: percentile_ci(v) for k, v in boot.items()}

    # Store article-level results for this axis
    axis_res = {}

    # Print table header for this axis
    print(f'{"Article":8s} {"DPD_base":>9s} {"DPD_scm":>9s} '
          f'{"dDPD":>8s} {"dDPD 95% CI":>20s} {"Reliable"}')
    print('-' * 70)

    # Save and print metrics for each article
    for a, name in enumerate(ARTICLE_NAMES):
        # Get confidence interval for delta-DPD
        lo, hi = ci['delta'][0][a], ci['delta'][1][a]

        # Check whether this article is in the reliable set
        rel = name in RELIABLE

        # Check whether the confidence interval crosses zero
        crosses_zero = bool(lo <= 0 <= hi) if not (np.isnan(lo) or np.isnan(hi)) else None

        # Store all metrics and CIs for this article
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

        # Format CI for printing
        ci_str = f'[{lo:+.4f}, {hi:+.4f}]' if not np.isnan(lo) else 'n/a'

        # Print one article row
        print(f'{name:8s} {dpd_b0[a]:9.4f} {dpd_s0[a]:9.4f} '
              f'{delta0[a]:+8.4f} {ci_str:>20s} {"yes" if rel else "no"}')

    # Store the results for this protected axis
    results[axis] = {
        'n_protected': n_p,
        'n_unprotected': n_u,
        'articles': axis_res,
    }

# Store metadata about the run so the JSON file is self-explanatory
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

# Final output dictionary with metadata and all results
out = {'meta': meta, 'results': results}

# Save the fairness results for this seed
with open(OUT_PATH, 'w') as f:
    json.dump(out, f, indent=2)

# Print where the output file was saved
print(f'\nSaved: {OUT_PATH}')
```
