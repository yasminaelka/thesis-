"""
Job 1 — Wilcoxon signed-rank tests on per-instance SHAP sufficiency
Yasmina El Kacemi — University of Amsterdam

Token selection: first-k% of real tokens (position-based, deterministic).
Same selection applied to both models → valid paired Wilcoxon test.
No gradient computation, no attention extraction — two forward passes per doc.
"""
import os, json, random, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from scipy.stats import wilcoxon

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

OUTPUT_DIR  = '/gpfs/home6/yelkacemi/thesis_outputs'
MODEL_NAME  = 'nlpaueb/legal-bert-base-uncased'
N_LABELS    = 10
MAX_LEN     = 512
SHAP_N_DOCS = 200
K_VALUES    = [0.01, 0.05, 0.10]

ARTICLE_NAMES     = ['Art.2','Art.3','Art.5','Art.6','Art.8','Art.9',
                     'Art.10','Art.11','Art.14','P1-1']
RELIABLE_ARTICLES = ['Art.2','Art.3','Art.5','Art.6','Art.8','P1-1']
ALPHA_CORRECTED   = 0.005

class LegalBertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, N_LABELS)
    def forward(self, input_ids, attention_mask):
        out  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pool = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pool))

class ECtHRDataset(Dataset):
    def __init__(self, tokenizer):
        ds = load_dataset('coastalcph/lex_glue', 'ecthr_a', split='test')
        self.items = []
        for ex in ds:
            txt = ' '.join(ex['text']) if isinstance(ex['text'], list) else ex['text']
            enc = tokenizer(txt, max_length=MAX_LEN, truncation=True,
                            padding='max_length', return_tensors='pt')
            label = torch.zeros(N_LABELS)
            for l in ex['labels']:
                if l < N_LABELS:
                    label[l] = 1.0
            self.items.append({
                'input_ids':      enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'labels':         label,
            })
    def __len__(self):        return len(self.items)
    def __getitem__(self, i): return self.items[i]

def load_model(path):
    model = LegalBertClassifier().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def compute_sufficiency(model, docs, k_values):
    """
    For each doc: keep first k% of real tokens, mask the rest.
    Sufficiency = mean drop in sigmoid prob over active labels.
    Returns {k: np.array(n_docs,)}
    """
    mask_id = tokenizer.mask_token_id
    results = {k: [] for k in k_values}

    with torch.no_grad():
        for idx, doc in enumerate(docs):
            if idx % 20 == 0:
                print(f'  doc {idx}/{len(docs)}')

            ids  = doc['input_ids'].unsqueeze(0).to(DEVICE)    # (1,512)
            amsk = doc['attention_mask'].unsqueeze(0).to(DEVICE)

            probs_orig = torch.sigmoid(model(ids, amsk)).squeeze(0).cpu().numpy()

            active = np.where(doc['labels'].numpy() > 0.5)[0]
            if len(active) == 0:
                active = np.arange(N_LABELS)

            # real content tokens: exclude [CLS]=0, [SEP], padding
            sep_id   = tokenizer.sep_token_id
            all_ids  = ids.squeeze(0).cpu().numpy()
            real_pos = [
                p for p in amsk.squeeze(0).cpu().numpy().nonzero()[0]
                if p != 0 and all_ids[p] != sep_id
            ]
            real_pos = np.array(real_pos)
            seq_len  = len(real_pos)

            for k in k_values:
                top_n      = max(1, int(np.ceil(k * seq_len)))
                # keep first top_n content tokens (positional proxy for importance)
                keep_pos   = set(real_pos[:top_n].tolist())

                ids_masked = ids.clone()
                for pos in real_pos:
                    if int(pos) not in keep_pos:
                        ids_masked[0, int(pos)] = mask_id

                probs_masked = torch.sigmoid(
                    model(ids_masked, amsk)).squeeze(0).cpu().numpy()

                suff = float(np.mean(
                    probs_orig[active] - probs_masked[active]))
                results[k].append(suff)

    return {k: np.array(v) for k, v in results.items()}

# ── Load data
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print('Loading dataset...')
test_ds = ECtHRDataset(tokenizer)

rng     = np.random.default_rng(SEED)
indices = rng.choice(len(test_ds), size=SHAP_N_DOCS, replace=False)
docs    = [test_ds[int(i)] for i in indices]
labels  = np.array([d['labels'].numpy() for d in docs])
print(f'Sampled {len(docs)} documents.')

# ── Load models and compute
print('\n=== BASELINE ===')
baseline_model = load_model(
    os.path.join(OUTPUT_DIR, 'baseline_model.pt'))
base_suff = compute_sufficiency(baseline_model, docs, K_VALUES)
del baseline_model; torch.cuda.empty_cache()

print('\n=== SCM (λ=0.1) ===')
scm_model = load_model(
    os.path.join(OUTPUT_DIR, 'contrastive_lam0.1.pt'))
scm_suff = compute_sufficiency(scm_model, docs, K_VALUES)
del scm_model; torch.cuda.empty_cache()

# ── Save raw arrays
raw = {'doc_indices': indices.tolist(), 'doc_labels': labels.tolist(),
       'article_names': ARTICLE_NAMES}
for k in K_VALUES:
    raw[f'k={k}'] = {'baseline': base_suff[k].tolist(),
                     'scm':      scm_suff[k].tolist()}
with open(os.path.join(OUTPUT_DIR, 'per_instance_sufficiency.json'), 'w') as f:
    json.dump(raw, f, indent=2)
print('\nPer-instance arrays saved.')

# ── Wilcoxon tests
print('\n=== WILCOXON SIGNED-RANK TESTS ===')
print(f'Bonferroni α\' = {ALPHA_CORRECTED}')

wilcoxon_results = {}
for k in K_VALUES:
    ba, sa = base_suff[k], scm_suff[k]
    wilcoxon_results[f'k={k}'] = {}
    print(f'\n─── k={int(k*100)}% ───')

    for level in ['macro'] + RELIABLE_ARTICLES:
        if level == 'macro':
            bv, sv = ba, sa
        else:
            art_idx = ARTICLE_NAMES.index(level)
            msk     = labels[:, art_idx] > 0.5
            bv, sv  = ba[msk], sa[msk]

        n = len(bv)
        if n < 10:
            print(f'  {level:<8} n={n:3d}  SKIPPED (n<10)')
            wilcoxon_results[f'k={k}'][level] = {'n': n, 'skipped': True}
            continue

        diffs = sv - bv
        if np.all(diffs == 0):
            print(f'  {level:<8} n={n:3d}  SKIPPED (all diffs zero)')
            wilcoxon_results[f'k={k}'][level] = {'n': n, 'skipped': True}
            continue

        stat, p = wilcoxon(bv, sv, alternative='two-sided')
        diff    = float(diffs.mean())
        sig     = p < ALPHA_CORRECTED
        wilcoxon_results[f'k={k}'][level] = {
            'n': n, 'baseline_mean': float(bv.mean()),
            'scm_mean': float(sv.mean()), 'mean_diff': diff,
            'statistic': float(stat), 'p_value': float(p),
            'significant': bool(sig)
        }
        print(f'  {level:<8} n={n:3d}  '
              f'base={bv.mean():.4f}  scm={sv.mean():.4f}  '
              f'Δ={diff:+.4f}  W={stat:.1f}  p={p:.4f}  '
              f'{"✓ sig" if sig else "ns"}')

with open(os.path.join(OUTPUT_DIR, 'wilcoxon_results.json'), 'w') as f:
    json.dump(wilcoxon_results, f, indent=2)
print('\nResults saved to wilcoxon_results.json')

print('\n=== THESIS TABLE (k=1%) ===')
print(f'{"Level":<10}{"n":>4}{"Baseline":>10}{"SCM":>10}{"Δ":>8}{"p":>10}{"Sig":>5}')
print('─'*52)
for level in ['macro'] + RELIABLE_ARTICLES:
    r = wilcoxon_results['k=0.01'].get(level, {})
    if r.get('skipped'):
        print(f'{level:<10}{r.get("n",0):>4}  {"SKIPPED":>35}')
    else:
        print(f'{level:<10}{r["n"]:>4}{r["baseline_mean"]:>10.4f}'
              f'{r["scm_mean"]:>10.4f}{r["mean_diff"]:>+8.4f}'
              f'{r["p_value"]:>10.4f}{"✓":>5 if r["significant"] else "":>5}')
