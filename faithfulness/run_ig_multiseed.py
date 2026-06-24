"""
Integrated Gradients faithfulness, MULTI-SEED version.
Yasmina El Kacemi - University of Amsterdam

Extends the single-seed Table 6 IG check to all five seeds, so the IG
corroboration of the SHAP sufficiency degradation is no longer n=1.

Design choices (deliberately matched to the existing pipeline):
  - Loads per-seed checkpoints contrastive_baseline.pt and
    contrastive_lam{BEST_LAM}.pt, exactly as fairness_ci_eod_multiseed.py.
  - BEST_LAM read from contrastive_performance.json['scm']['lambda'], the SAME
    source run_faithfulness_only.py uses (faithfulness lineage), so this sits
    next to Table 5 at the same operating point.
  - Same MODEL_NAME, MAX_LEN=512, head+tail 256/256 truncation.
  - Sufficiency / comprehensiveness defined identically to the SHAP path:
      sufficiency      = |orig_prob - prob(keep top-k attributed tokens)|
      comprehensiveness = |orig_prob - prob(remove top-k attributed tokens)|
    averaged over the SAME 200 random test docs (rng seeded for reproducibility)
    and over the SAME seven reliable articles used elsewhere.
  - IG: 50 integration steps, embedding-space interpolation from a [PAD]/zero
    baseline to the real input embeddings (no captum dependency; pure autograd).
  - k in {1%,5%,10%}; "kept"/"removed" tokens chosen by per-token IG magnitude.
  - tokens that are masked out are replaced with [MASK], matching the SHAP path.

Output: per seed dir -> contrastive_ig_multiseed.json
        (NEW filename; never overwrites contrastive_faithfulness.json or the
         quarantined DEGENERATE files)

NOTE: seed 42 with these settings should reproduce the existing Table 6
      SCM sufficiency (~0.1469 / 0.1475 / 0.1454). If it does not, stop and
      compare against run_faithfulness_only.py before trusting other seeds.
"""

import os, json, argparse, warnings
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
ap.add_argument('--n_docs',  type=int, default=200)
ap.add_argument('--ig_steps', type=int, default=50)
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

# per-seed best lambda, from the SAME json run_faithfulness_only.py used
BEST_LAM = json.load(
    open(os.path.join(OUTPUT_DIR, 'contrastive_performance.json'))
)['scm']['lambda']
print(f'BEST_LAM (from contrastive_performance.json) = {BEST_LAM}')

MODEL_NAME = 'nlpaueb/legal-bert-base-uncased'
N_LABELS   = 10
MAX_LEN    = 512
HEAD, TAIL = 256, 256
K_VALUES   = [0.01, 0.05, 0.10]

ARTICLE_NAMES = ['Art.2', 'Art.3', 'Art.5', 'Art.6', 'Art.8', 'Art.9',
                 'Art.10', 'Art.11', 'Art.14', 'P1-1']
RELIABLE_IDX  = [i for i, n in enumerate(ARTICLE_NAMES)
                 if n in {'Art.2', 'Art.3', 'Art.5', 'Art.6',
                          'Art.8', 'Art.10', 'P1-1'}]

BASELINE_CKPT = os.path.join(OUTPUT_DIR, 'contrastive_baseline.pt')
SCM_CKPT      = os.path.join(OUTPUT_DIR, f'contrastive_lam{BEST_LAM}.pt')
OUT_PATH      = os.path.join(OUTPUT_DIR, 'contrastive_ig_multiseed.json')
assert os.path.exists(BASELINE_CKPT), BASELINE_CKPT
assert os.path.exists(SCM_CKPT), SCM_CKPT

TOKENIZER  = AutoTokenizer.from_pretrained(MODEL_NAME)
MASK_ID    = TOKENIZER.mask_token_id
PAD_ID     = TOKENIZER.pad_token_id

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
        ids  = torch.cat([ids,  torch.full((pad,), PAD_ID, dtype=torch.long)])
        mask = torch.cat([mask, torch.zeros(pad, dtype=torch.long)])
    return ids, mask

# -- Model (identical architecture to fairness_ci_eod_multiseed.py) ------------
class BERTClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is not None:
            out = self.bert(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask)
        else:
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

def embed_layer(model):
    return model.bert.embeddings.word_embeddings

# -- Data ----------------------------------------------------------------------
print('Loading dataset ...')
raw = load_dataset('coastalcph/lex_glue', 'ecthr_a', trust_remote_code=True)

def to_arrays(split):
    ids, masks, labels = [], [], []
    for ex in split:
        i, m = tokenize_head_tail(ex['text'])
        ids.append(i); masks.append(m)
        y = np.zeros(N_LABELS, dtype=np.int64)
        for l in ex['labels']:
            if l < N_LABELS:
                y[l] = 1
        labels.append(y)
    return torch.stack(ids), torch.stack(masks), np.array(labels)

test_ids, test_mask, test_y = to_arrays(raw['test'])
print(f'Test: {len(test_y)} docs')

# Same 200-doc sample for every model, reproducible per seed.
rng = np.random.default_rng(SEED)
doc_idx = rng.choice(len(test_y), size=min(args.n_docs, len(test_y)),
                     replace=False)
print(f'Using {len(doc_idx)} test docs for IG (seed-reproducible sample).')

def ig_attributions(model, ids, mask, target_articles, steps):
    """Integrated Gradients, matched EXACTLY to run_ig_faithfulness.py:
      - baseline = ZERO embeddings (not PAD)
      - target   = sigmoid(logits).sum() over ALL labels
      - aggregate over hidden dim by L2 NORM (Atanasova et al. 2020)
      - Riemann sum over linspace(0,1,steps+1), divide by (steps+1)
    target_articles is accepted for signature compatibility but NOT used,
    because the original explains the sum over all labels."""
    emb = embed_layer(model)
    ids_d  = ids.unsqueeze(0).to(DEVICE)
    mask_d = mask.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        input_embeds = emb(ids_d)                       # (1,L,H)
    baseline_embeds = torch.zeros_like(input_embeds)    # ZERO baseline
    alphas   = torch.linspace(0, 1, steps + 1).to(DEVICE)
    grad_sum = torch.zeros_like(input_embeds)
    for alpha in alphas:
        interp = (baseline_embeds + alpha * (input_embeds - baseline_embeds)) \
                 .clone().detach().requires_grad_(True)
        logits = model(inputs_embeds=interp, attention_mask=mask_d)
        target = torch.sigmoid(logits).sum()            # ALL labels
        grad, = torch.autograd.grad(target, interp)
        grad_sum += grad.detach()
    ig_embeds = (input_embeds - baseline_embeds) * grad_sum / (steps + 1)
    # L2 norm over hidden dim -> (L,) token importance
    return ig_embeds.squeeze(0).norm(dim=-1).cpu().numpy()

def prob_from_ids_arr(model, ids, mask):
    """Sigmoid over ALL 10 logits, returned as a (1,10) numpy array, matching
    run_faithfulness_only.py exactly."""
    with torch.no_grad():
        logits = model(input_ids=ids.unsqueeze(0).to(DEVICE),
                       attention_mask=mask.unsqueeze(0).to(DEVICE))
        return torch.sigmoid(logits).cpu().numpy()      # (1,10)

def compute_ig_faithfulness(model, tag):
    """IG attribution + sufficiency/comprehensiveness defined IDENTICALLY to the
    original SHAP path in run_faithfulness_only.py:
      - rank real tokens by attribution magnitude
      - sufficiency: keep top-k, [MASK] the rest, |orig - suf| averaged over ALL
        10 classes, then over docs
      - comprehensiveness: [MASK] the top-k, |orig - com| over ALL 10 classes
    The ONLY change vs the original is the attribution source (IG, not SHAP)."""
    print(f'\n[{tag}] computing IG faithfulness over {len(doc_idx)} docs ...')
    suff = {k: [] for k in K_VALUES}
    comp = {k: [] for k in K_VALUES}
    for n, di in enumerate(doc_idx):
        ids  = test_ids[di]
        mask = test_mask[di]
        # IG target = reliable articles the doc is positive for; fall back to all
        # reliable if none. (Attribution target only; faithfulness still scored
        # over all 10 classes below, matching the original.)
        pos_articles = [a for a in RELIABLE_IDX if test_y[di, a] == 1]
        targets = pos_articles if pos_articles else RELIABLE_IDX
        ig = ig_attributions(model, ids, mask, targets, args.ig_steps)

        # real token positions: attention==1, excluding [CLS] at index 0
        attn = mask.cpu().numpy()
        real_pos = np.where(attn == 1)[0]
        real_pos = real_pos[real_pos != 0]
        seq_len  = len(real_pos)
        if seq_len < 2:
            continue
        imp_real = ig[real_pos]
        orig_prob = prob_from_ids_arr(model, ids, mask)        # (1,10)

        for k in K_VALUES:
            top_k_n   = max(1, int(seq_len * k))
            order     = np.argsort(imp_real)
            top_k_pos = set(real_pos[order[-top_k_n:]].tolist())
            non_top_k = [int(p) for p in real_pos if int(p) not in top_k_pos]

            suf_ids = ids.clone()
            if non_top_k:
                suf_ids[non_top_k] = MASK_ID
            suf_prob = prob_from_ids_arr(model, suf_ids, mask)
            suff[k].append(float(np.abs(orig_prob - suf_prob).mean()))

            com_ids = ids.clone()
            com_ids[list(top_k_pos)] = MASK_ID
            com_prob = prob_from_ids_arr(model, com_ids, mask)
            comp[k].append(float(np.abs(orig_prob - com_prob).mean()))

        if (n + 1) % 50 == 0:
            print(f'   {n+1}/{len(doc_idx)} docs')
    out = {}
    for k in K_VALUES:
        out[f'k={k}'] = {
            'sufficiency': float(np.mean(suff[k])),
            'comprehensiveness': float(np.mean(comp[k])),
        }
    return out

# -- Run -----------------------------------------------------------------------
base_model = load_model(BASELINE_CKPT)
scm_model  = load_model(SCM_CKPT)

faith_base = compute_ig_faithfulness(base_model, 'baseline')
faith_scm  = compute_ig_faithfulness(scm_model,  f'SCM (lam={BEST_LAM})')

result = {
    'seed': SEED,
    'pairs': args.pairs,
    'lambda': float(BEST_LAM),
    'method': 'integrated_gradients',
    'ig_steps': args.ig_steps,
    'n_docs': len(doc_idx),
    'baseline': faith_base,
    'scm': faith_scm,
}
with open(OUT_PATH, 'w') as f:
    json.dump(result, f, indent=2)

print('\n=== IG faithfulness (this seed) ===')
for k in K_VALUES:
    kk = f'k={k}'
    b = faith_base[kk]['sufficiency']
    s = faith_scm[kk]['sufficiency']
    print(f'  {kk:7s} suff  base={b:.4f}  SCM={s:.4f}  dSuff={s-b:+.4f}')
print(f'\nSaved: {OUT_PATH}')
if SEED == 42:
    print('\nSANITY (seed 42): expected SCM suff ~ 0.1469 / 0.1475 / 0.1454 '
          'at k=1/5/10%. Compare against Table 6.')
