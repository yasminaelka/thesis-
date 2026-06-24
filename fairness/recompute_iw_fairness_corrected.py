"""
recompute_iw_fairness_corrected.py
----------------------------------
Recomputes the Appendix E (instance-weighted) per-article DPD under the
CORRECTED whole-word keyword grouping (gender 931/69, ethnicity 136/864),
replacing the stale 978/22 grouping in instance_weighted_fairness.json.

No retraining: loads the saved IW lambda=0.5 checkpoint and the seed-42
baseline checkpoint, tunes per-article thresholds per model on validation,
applies the corrected grouping, and prints per-article DPD for the seven
reliable articles on both axes.

All metric / grouping / threshold / tokenisation logic is copied verbatim
from fairness_ci_eod_multiseed.py so the numbers are directly comparable to
the rest of the thesis.

Run on a gpu_a100 node:
  python recompute_iw_fairness_corrected.py
"""
import os, re, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.metrics import f1_score

ROOT          = '/gpfs/home6/yelkacemi/output'
IW_CKPT       = os.path.join(ROOT, 'instance_weighted_lam0.5.pt')
BASELINE_CKPT = os.path.join(ROOT, 'legal-bert_scm_seed42', 'contrastive_baseline.pt')
SEED          = 42
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

MODEL_NAME    = 'nlpaueb/legal-bert-base-uncased'
N_LABELS      = 10
MAX_LEN       = 512
HEAD, TAIL    = 256, 256
TAU_GRID      = np.round(np.arange(0.05, 0.96, 0.05), 2)
ARTICLE_NAMES = ['Art.2', 'Art.3', 'Art.5', 'Art.6', 'Art.8', 'Art.9',
                 'Art.10', 'Art.11', 'Art.14', 'P1-1']
RELIABLE      = {'Art.2', 'Art.3', 'Art.5', 'Art.6', 'Art.8', 'Art.10', 'P1-1'}

# -- keyword lists: verbatim from fairness_ci_eod_multiseed.py (lines 75-92) ---
GENDER_KEYWORDS = [
    'woman', 'women', 'female', 'girl', 'mother', 'wife', 'daughter',
    'sister', 'she', 'her', 'hers', 'lady', 'bride', 'girlfriend',
    'stepmother', 'grandmother', 'schoolgirl', 'mommy', 'aunt', 'niece',
    'man', 'men', 'male', 'boy', 'father', 'husband', 'son', 'brother',
    'he', 'him', 'his', 'gentleman', 'groom', 'boyfriend', 'stepfather',
    'grandfather', 'schoolboy', 'daddy', 'uncle', 'nephew',
]
ETHNICITY_KEYWORDS = [
    'roma', 'romani', 'gypsy', 'kurdish', 'kurd', 'chechen',
    'jewish', 'muslim', 'christian', 'orthodox',
    'asylum', 'refugee', 'immigrant', 'migrant', 'foreigner',
    'minority', 'ethnic', 'ethnicity', 'race', 'racial',
    'indigenous', 'aboriginal', 'caste',
]

# -- tokeniser + head/tail truncation (verbatim) -------------------------------
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_head_tail(text):
    t = TOKENIZER(text, truncation=False, add_special_tokens=True,
                  return_tensors='pt')
    ids  = t['input_ids'][0]
    mask = t['attention_mask'][0]
    if len(ids) > MAX_LEN:
        ids  = torch.cat([ids[:HEAD],  ids[-TAIL:]])
        mask = torch.cat([mask[:HEAD], mask[-TAIL:]])
    pad = MAX_LEN - len(ids)
    if pad > 0:
        ids  = torch.cat([ids,  torch.zeros(pad, dtype=torch.long)])
        mask = torch.cat([mask, torch.zeros(pad, dtype=torch.long)])
    return ids, mask

# -- model (verbatim from run_contrastive_v2.py) -------------------------------
class BERTClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls     = outputs.last_hidden_state[:, 0, :]
        return {'logits': self.classifier(cls)}

def load_model(path):
    m = BERTClassifier(N_LABELS).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    sd   = ckpt.get('model_state_dict', ckpt)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

@torch.no_grad()
def predict_probs(model, ids, mask, batch=16):
    out = []
    for i in range(0, len(ids), batch):
        b_ids  = ids[i:i+batch].to(DEVICE)
        b_mask = mask[i:i+batch].to(DEVICE)
        logits = model(b_ids, b_mask)['logits']
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, 0)

def tune_thresholds(val_probs, val_labels):
    thr = np.full(N_LABELS, 0.5)
    for a in range(N_LABELS):
        best_f1, best_t = -1.0, 0.5
        for t in TAU_GRID:
            pred = (val_probs[:, a] >= t).astype(int)
            f1 = f1_score(val_labels[:, a], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[a] = best_t
    return thr

# -- grouping + DPD (verbatim) -------------------------------------------------
def build_keyword_pattern(keywords):
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', flags=re.IGNORECASE)

def build_group_mask(texts, keywords):
    pattern = build_keyword_pattern(keywords)
    protected = np.zeros(len(texts), dtype=bool)
    for i, txt in enumerate(texts):
        protected[i] = bool(pattern.search(txt))
    return protected

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

# -- data (text built EXACTLY as in fairness_ci_eod_multiseed.py) --------------
print('Loading ECtHR ecthr_a ...')
raw = load_dataset('coastalcph/lex_glue', 'ecthr_a', trust_remote_code=True)

def split_to_arrays(split):
    ids, masks, labels, texts = [], [], [], []
    for ex in split:
        _txt = ' '.join(ex['text']) if isinstance(ex['text'], list) else ex['text']
        i, m = tokenize_head_tail(_txt)
        ids.append(i); masks.append(m)
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

# -- score both models ---------------------------------------------------------
print('Loading models ...')
base_model = load_model(BASELINE_CKPT)
iw_model   = load_model(IW_CKPT)

print('Scoring validation (thresholds) ...')
thr_base = tune_thresholds(predict_probs(base_model, val_ids, val_mask), val_y)
thr_iw   = tune_thresholds(predict_probs(iw_model,   val_ids, val_mask), val_y)

print('Scoring test ...')
pred_base = (predict_probs(base_model, test_ids, test_mask) >= thr_base).astype(int)
pred_iw   = (predict_probs(iw_model,   test_ids, test_mask) >= thr_iw  ).astype(int)

# -- corrected grouping + per-article DPD --------------------------------------
AXES = {'gender': GENDER_KEYWORDS, 'ethnicity': ETHNICITY_KEYWORDS}
print('\n=== Corrected-grouping instance-weighted DPD (seed 42, lambda=0.5) ===')
summary = {}
for axis, kw in AXES.items():
    prot = build_group_mask(test_texts, kw)
    print(f'\n[{axis}] protected={int(prot.sum())}  unprotected={int((~prot).sum())}')
    dpd_base, _ = dpd_di_vec(pred_base, prot)
    dpd_iw,   _ = dpd_di_vec(pred_iw,   prot)
    print(f'{"Article":8} {"base":>8} {"IW":>8} {"dDPD":>9}')
    for a, name in enumerate(ARTICLE_NAMES):
        if name not in RELIABLE:
            continue
        print(f'{name:8} {dpd_base[a]:8.4f} {dpd_iw[a]:8.4f} {dpd_iw[a]-dpd_base[a]:+9.4f}')
    rel_idx = [i for i, n in enumerate(ARTICLE_NAMES) if n in RELIABLE]
    mb = float(np.nanmean([dpd_base[i] for i in rel_idx]))
    mi = float(np.nanmean([dpd_iw[i]   for i in rel_idx]))
    print(f'{"MEAN":8} {mb:8.4f} {mi:8.4f} {mi-mb:+9.4f}')
    summary[axis] = {'protected': int(prot.sum()), 'mean_base': mb,
                     'mean_iw': mi, 'mean_delta': mi-mb}

with open('iw_fairness_corrected_seed42.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('\nSaved -> iw_fairness_corrected_seed42.json')
