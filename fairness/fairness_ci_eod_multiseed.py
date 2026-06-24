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
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# -- CLI -----------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('--encoder', default='legal-bert')
ap.add_argument('--pairs',   default='scm')
ap.add_argument('--seed',    type=int, default=42)
ap.add_argument('--root',    default='/gpfs/home6/yelkacemi/output')
args = ap.parse_args()

SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU   : {torch.cuda.get_device_name(0)}')

# -- Per-seed paths ------------------------------------------------------------
RUN_TAG    = f'{args.encoder}_{args.pairs}_seed{SEED}'
OUTPUT_DIR = os.path.join(args.root, RUN_TAG)
assert os.path.isdir(OUTPUT_DIR), f'missing dir {OUTPUT_DIR}'
print(f'Seed: {SEED}  Dir: {OUTPUT_DIR}')

# per-seed best lambda, read from the same JSON the faithfulness run used
BEST_LAM = json.load(
    open(os.path.join(OUTPUT_DIR, 'contrastive_faithfulness.json'))
)['lambda']
print(f'BEST_LAM (per-seed, from contrastive_faithfulness.json) = {BEST_LAM}')

MODEL_NAME    = 'nlpaueb/legal-bert-base-uncased'
N_LABELS      = 10
MAX_LEN       = 512
HEAD, TAIL    = 256, 256
N_BOOT        = 10000
TAU_GRID      = np.round(np.arange(0.05, 0.96, 0.05), 2)

ARTICLE_NAMES = ['Art.2', 'Art.3', 'Art.5', 'Art.6', 'Art.8', 'Art.9',
                 'Art.10', 'Art.11', 'Art.14', 'P1-1']
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

BASELINE_CKPT = os.path.join(OUTPUT_DIR, 'contrastive_baseline.pt')
SCM_CKPT      = os.path.join(OUTPUT_DIR, f'contrastive_lam{BEST_LAM}.pt')
OUT_PATH      = os.path.join(OUTPUT_DIR, 'fairness_ci_eod.json')
assert os.path.exists(BASELINE_CKPT), BASELINE_CKPT
assert os.path.exists(SCM_CKPT), SCM_CKPT

# -- Tokeniser + head/tail truncation ------------------------------------------
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_head_tail(text):
    if isinstance(text, list):
        text = ' '.join(text)
    t = TOKENIZER(text, truncation=False, add_special_tokens=True,
                  return_tensors='pt')
    ids, mask = t['input_ids'][0], t['attention_mask'][0]
    if len(ids) > MAX_LEN:
        ids  = torch.cat([ids[:HEAD],  ids[-TAIL:]])
        mask = torch.cat([mask[:HEAD], mask[-TAIL:]])
    pad = MAX_LEN - len(ids)
    if pad > 0:
        ids  = torch.cat([ids,  torch.zeros(pad, dtype=torch.long)])
        mask = torch.cat([mask, torch.zeros(pad, dtype=torch.long)])
    return ids, mask

# -- Model ---------------------------------------------------------------------
class BERTClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

def load_model(path):
    m = BERTClassifier(N_LABELS).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    sd = ckpt.get('model_state_dict', ckpt)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

# -- Data ----------------------------------------------------------------------
print('Loading dataset ...')
raw = load_dataset('coastalcph/lex_glue', 'ecthr_a', trust_remote_code=True)

def split_to_arrays(split):
    ids, masks, labels, texts = [], [], [], []
    for ex in split:
        i, m = tokenize_head_tail(ex['text'])
        ids.append(i)
        masks.append(m)
        y = np.zeros(N_LABELS, dtype=np.int64)
        for l in ex['labels']:
            if l < N_LABELS:
                y[l] = 1
        labels.append(y)
        texts.append((' '.join(ex['text']) if isinstance(ex['text'], list)
                     else ex['text']).lower())
    return (torch.stack(ids), torch.stack(masks), np.array(labels), texts)

val_ids,  val_mask,  val_y,  _          = split_to_arrays(raw['validation'])
test_ids, test_mask, test_y, test_texts = split_to_arrays(raw['test'])
print(f'Val: {len(val_y)} | Test: {len(test_y)}')

# -- Probabilities -------------------------------------------------------------
@torch.no_grad()
def predict_probs(model, ids, mask, batch=16):
    probs = []
    for i in range(0, len(ids), batch):
        b_ids  = ids[i:i + batch].to(DEVICE)
        b_mask = mask[i:i + batch].to(DEVICE)
        p = torch.sigmoid(model(b_ids, b_mask)).cpu().numpy()
        probs.append(p)
    return np.vstack(probs)

# -- Per-article threshold tuning on validation (maximise F1) ------------------
def tune_thresholds(val_probs, val_labels):
    thr = np.full(N_LABELS, 0.5)
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
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', flags=re.IGNORECASE)

def build_group_mask(texts, keywords):
    pattern = build_keyword_pattern(keywords)
    protected = np.zeros(len(texts), dtype=bool)
    for i, txt in enumerate(texts):
        protected[i] = bool(pattern.search(txt))
    return protected

# -- Vectorised fairness metrics -----------------------------------------------
def dpd_di_vec(pred, prot):
    P, U = pred[prot], pred[~prot]
    if P.shape[0] == 0 or U.shape[0] == 0:
        nan = np.full(N_LABELS, np.nan)
        return nan, nan
    p_prot, p_unp = P.mean(0), U.mean(0)
    dpd = np.abs(p_prot - p_unp)
    with np.errstate(divide='ignore', invalid='ignore'):
        di = np.where(p_unp > 0, p_prot / p_unp, np.nan)
    return dpd, di

def eod_vec(pred, y, prot):
    pred = pred.astype(float)

    def grp(mask_rows):
        pos = (y == 1) & mask_rows[:, None]
        neg = (y == 0) & mask_rows[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            tpr = (pred * pos).sum(0) / pos.sum(0)
            fpr = (pred * neg).sum(0) / neg.sum(0)
        return tpr, fpr

    tpr_p, fpr_p = grp(prot)
    tpr_u, fpr_u = grp(~prot)
    d_tpr = np.abs(tpr_p - tpr_u)
    d_fpr = np.abs(fpr_p - fpr_u)
    return np.fmax(d_tpr, d_fpr)

# -- Load models and compute test predictions ---------------------------------
print('\nLoading models ...')
base_model = load_model(BASELINE_CKPT)
scm_model  = load_model(SCM_CKPT)

print('Scoring validation (for thresholds) ...')
val_probs_base = predict_probs(base_model, val_ids, val_mask)
val_probs_scm  = predict_probs(scm_model,  val_ids, val_mask)
thr_base = tune_thresholds(val_probs_base, val_y)
thr_scm  = tune_thresholds(val_probs_scm,  val_y)

print('Scoring test ...')
test_probs_base = predict_probs(base_model, test_ids, test_mask)
test_probs_scm  = predict_probs(scm_model,  test_ids, test_mask)
pred_base = (test_probs_base >= thr_base).astype(int)
pred_scm  = (test_probs_scm  >= thr_scm ).astype(int)

f1_base = macro_f1_reliable(pred_base, test_y)
f1_scm  = macro_f1_reliable(pred_scm,  test_y)
print(f'\nSanity: macro F1  baseline={f1_base:.4f}  SCM={f1_scm:.4f}')

AXES = {'gender': GENDER_KEYWORDS, 'ethnicity': ETHNICITY_KEYWORDS}

# -- Point estimates + bootstrap CIs per axis ----------------------------------
def percentile_ci(arr):
    lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
    return lo, hi

def _num(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

results = {}
rng = np.random.default_rng(SEED)
n = len(test_y)

for axis, kw in AXES.items():
    prot = build_group_mask(test_texts, kw)
    n_p, n_u = int(prot.sum()), int((~prot).sum())
    print(f'\n=== {axis} ===   protected={n_p}  unprotected={n_u}')

    dpd_b0, di_b0 = dpd_di_vec(pred_base, prot)
    dpd_s0, di_s0 = dpd_di_vec(pred_scm,  prot)
    eod_b0 = eod_vec(pred_base, test_y, prot)
    eod_s0 = eod_vec(pred_scm,  test_y, prot)
    delta0 = dpd_s0 - dpd_b0

    boot = {k: np.full((N_BOOT, N_LABELS), np.nan)
            for k in ['dpd_b', 'dpd_s', 'delta', 'di_b', 'di_s', 'eod_b', 'eod_s']}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        prot_b = prot[idx]
        pb, ps, yb = pred_base[idx], pred_scm[idx], test_y[idx]
        d_b, i_b = dpd_di_vec(pb, prot_b)
        d_s, i_s = dpd_di_vec(ps, prot_b)
        boot['dpd_b'][b], boot['dpd_s'][b] = d_b, d_s
        boot['delta'][b] = d_s - d_b
        boot['di_b'][b],  boot['di_s'][b]  = i_b, i_s
        boot['eod_b'][b] = eod_vec(pb, yb, prot_b)
        boot['eod_s'][b] = eod_vec(ps, yb, prot_b)

    ci = {k: percentile_ci(v) for k, v in boot.items()}

    axis_res = {}
    print(f'{"Article":8s} {"DPD_base":>9s} {"DPD_scm":>9s} '
          f'{"dDPD":>8s} {"dDPD 95% CI":>20s} {"Reliable"}')
    print('-' * 70)
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

    results[axis] = {
        'n_protected': n_p,
        'n_unprotected': n_u,
        'articles': axis_res,
    }

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
out = {'meta': meta, 'results': results}
with open(OUT_PATH, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved: {OUT_PATH}')
