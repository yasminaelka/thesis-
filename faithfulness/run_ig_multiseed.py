"""
Integrated Gradients faithfulness for multiple seeds.

This script loads the saved baseline and SCM models for one seed, then computes
IG-based sufficiency and comprehensiveness scores.
"""

import os, json, argparse, warnings

# Keep terminal output cleaner
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
ap.add_argument('--n_docs',  type=int, default=200)
ap.add_argument('--ig_steps', type=int, default=50)
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

# Build output folder for this seed
RUN_TAG    = f'{args.encoder}_{args.pairs}_seed{SEED}'
OUTPUT_DIR = os.path.join(args.root, RUN_TAG)

# Check that the folder exists
assert os.path.isdir(OUTPUT_DIR), f'missing dir {OUTPUT_DIR}'

print(f'Seed: {SEED}  Dir: {OUTPUT_DIR}')

# Read best lambda for this seed
BEST_LAM = json.load(
    open(os.path.join(OUTPUT_DIR, 'contrastive_performance.json'))
)['scm']['lambda']

print(f'BEST_LAM (from contrastive_performance.json) = {BEST_LAM}')

# Main settings
MODEL_NAME = 'nlpaueb/legal-bert-base-uncased'
N_LABELS   = 10
MAX_LEN    = 512
HEAD, TAIL = 256, 256
K_VALUES   = [0.01, 0.05, 0.10]

# Article labels
ARTICLE_NAMES = ['Art.2', 'Art.3', 'Art.5', 'Art.6', 'Art.8', 'Art.9',
                 'Art.10', 'Art.11', 'Art.14', 'P1-1']

# Reliable article indices
RELIABLE_IDX  = [i for i, n in enumerate(ARTICLE_NAMES)
                 if n in {'Art.2', 'Art.3', 'Art.5', 'Art.6',
                          'Art.8', 'Art.10', 'P1-1'}]

# Model and output paths
BASELINE_CKPT = os.path.join(OUTPUT_DIR, 'contrastive_baseline.pt')
SCM_CKPT      = os.path.join(OUTPUT_DIR, f'contrastive_lam{BEST_LAM}.pt')
OUT_PATH      = os.path.join(OUTPUT_DIR, 'contrastive_ig_multiseed.json')

# Check model files
assert os.path.exists(BASELINE_CKPT), BASELINE_CKPT
assert os.path.exists(SCM_CKPT), SCM_CKPT

# Load tokenizer
TOKENIZER  = AutoTokenizer.from_pretrained(MODEL_NAME)
MASK_ID    = TOKENIZER.mask_token_id
PAD_ID     = TOKENIZER.pad_token_id

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
        ids  = torch.cat([ids,  torch.full((pad,), PAD_ID, dtype=torch.long)])
        mask = torch.cat([mask, torch.zeros(pad, dtype=torch.long)])

    return ids, mask

# -- Model ---------------------------------------------------------------------

class BERTClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()

        # Encoder and classifier head
        self.bert       = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        # Forward pass with either IDs or embeddings
        if inputs_embeds is not None:
            out = self.bert(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask)
        else:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS representation
        cls = out.last_hidden_state[:, 0, :]

        return self.classifier(cls)

def load_model(path):
    # Build model
    m = BERTClassifier(N_LABELS).to(DEVICE)

    # Load checkpoint
    ckpt = torch.load(path, map_location=DEVICE)
    sd = ckpt.get('model_state_dict', ckpt)

    # Load weights
    m.load_state_dict(sd, strict=False)
    m.eval()

    return m

def embed_layer(model):
    # Get embedding layer for IG
    return model.bert.embeddings.word_embeddings

# -- Data ----------------------------------------------------------------------

# Load dataset
print('Loading dataset ...')
raw = load_dataset('coastalcph/lex_glue', 'ecthr_a', trust_remote_code=True)

def to_arrays(split):
    # Store inputs and labels
    ids, masks, labels = [], [], []

    for ex in split:
        # Tokenize document
        i, m = tokenize_head_tail(ex['text'])
        ids.append(i); masks.append(m)

        # Create label vector
        y = np.zeros(N_LABELS, dtype=np.int64)

        for l in ex['labels']:
            if l < N_LABELS:
                y[l] = 1

        labels.append(y)

    return torch.stack(ids), torch.stack(masks), np.array(labels)

# Prepare test split
test_ids, test_mask, test_y = to_arrays(raw['test'])
print(f'Test: {len(test_y)} docs')

# Sample documents for this seed
rng = np.random.default_rng(SEED)
doc_idx = rng.choice(len(test_y), size=min(args.n_docs, len(test_y)),
                     replace=False)
print(f'Using {len(doc_idx)} test docs for IG (seed-reproducible sample).')

def ig_attributions(model, ids, mask, target_articles, steps):
    """Compute Integrated Gradients token importance."""

    # Get embedding layer
    emb = embed_layer(model)

    # Add batch dimension
    ids_d  = ids.unsqueeze(0).to(DEVICE)
    mask_d = mask.unsqueeze(0).to(DEVICE)

    # Real input embeddings
    with torch.no_grad():
        input_embeds = emb(ids_d)                       # (1,L,H)

    # Zero baseline
    baseline_embeds = torch.zeros_like(input_embeds)    # ZERO baseline

    # Interpolation points
    alphas   = torch.linspace(0, 1, steps + 1).to(DEVICE)

    # Store summed gradients
    grad_sum = torch.zeros_like(input_embeds)

    # Integrated Gradients loop
    for alpha in alphas:
        interp = (baseline_embeds + alpha * (input_embeds - baseline_embeds)) \
                 .clone().detach().requires_grad_(True)

        # Forward pass
        logits = model(inputs_embeds=interp, attention_mask=mask_d)

        # Target score
        target = torch.sigmoid(logits).sum()            # ALL labels

        # Gradient with respect to embeddings
        grad, = torch.autograd.grad(target, interp)
        grad_sum += grad.detach()

    # Combine gradients with input difference
    ig_embeds = (input_embeds - baseline_embeds) * grad_sum / (steps + 1)

    # Token-level importance
    return ig_embeds.squeeze(0).norm(dim=-1).cpu().numpy()

def prob_from_ids_arr(model, ids, mask):
    """Return probabilities for all labels."""

    # Predict probabilities
    with torch.no_grad():
        logits = model(input_ids=ids.unsqueeze(0).to(DEVICE),
                       attention_mask=mask.unsqueeze(0).to(DEVICE))
        return torch.sigmoid(logits).cpu().numpy()      # (1,10)

def compute_ig_faithfulness(model, tag):
    """Compute IG sufficiency and comprehensiveness."""

    print(f'\n[{tag}] computing IG faithfulness over {len(doc_idx)} docs ...')

    # Store scores
    suff = {k: [] for k in K_VALUES}
    comp = {k: [] for k in K_VALUES}

    # Loop over sampled documents
    for n, di in enumerate(doc_idx):
        ids  = test_ids[di]
        mask = test_mask[di]

        # Pick target articles for attribution
        pos_articles = [a for a in RELIABLE_IDX if test_y[di, a] == 1]
        targets = pos_articles if pos_articles else RELIABLE_IDX

        # Get IG token importance
        ig = ig_attributions(model, ids, mask, targets, args.ig_steps)

        # Real token positions, excluding CLS
        attn = mask.cpu().numpy()
        real_pos = np.where(attn == 1)[0]
        real_pos = real_pos[real_pos != 0]
        seq_len  = len(real_pos)

        # Skip very short examples
        if seq_len < 2:
            continue

        # Importance for real tokens
        imp_real = ig[real_pos]

        # Original prediction
        orig_prob = prob_from_ids_arr(model, ids, mask)        # (1,10)

        # Evaluate each k
        for k in K_VALUES:
            # Number of top tokens
            top_k_n   = max(1, int(seq_len * k))

            # Rank tokens
            order     = np.argsort(imp_real)

            # Top-k token positions
            top_k_pos = set(real_pos[order[-top_k_n:]].tolist())

            # All other real tokens
            non_top_k = [int(p) for p in real_pos if int(p) not in top_k_pos]

            # Sufficiency
            suf_ids = ids.clone()
            if non_top_k:
                suf_ids[non_top_k] = MASK_ID
            suf_prob = prob_from_ids_arr(model, suf_ids, mask)
            suff[k].append(float(np.abs(orig_prob - suf_prob).mean()))

            # Comprehensiveness
            com_ids = ids.clone()
            com_ids[list(top_k_pos)] = MASK_ID
            com_prob = prob_from_ids_arr(model, com_ids, mask)
            comp[k].append(float(np.abs(orig_prob - com_prob).mean()))

        # Print progress
        if (n + 1) % 50 == 0:
            print(f'   {n+1}/{len(doc_idx)} docs')

    # Average scores
    out = {}
    for k in K_VALUES:
        out[f'k={k}'] = {
            'sufficiency': float(np.mean(suff[k])),
            'comprehensiveness': float(np.mean(comp[k])),
        }

    return out

# -- Run -----------------------------------------------------------------------

# Load models
base_model = load_model(BASELINE_CKPT)
scm_model  = load_model(SCM_CKPT)

# Compute IG faithfulness
faith_base = compute_ig_faithfulness(base_model, 'baseline')
faith_scm  = compute_ig_faithfulness(scm_model,  f'SCM (lam={BEST_LAM})')

# Store results
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

# Save results
with open(OUT_PATH, 'w') as f:
    json.dump(result, f, indent=2)

# Print summary
print('\n=== IG faithfulness (this seed) ===')
for k in K_VALUES:
    kk = f'k={k}'
    b = faith_base[kk]['sufficiency']
    s = faith_scm[kk]['sufficiency']
    print(f'  {kk:7s} suff  base={b:.4f}  SCM={s:.4f}  dSuff={s-b:+.4f}')

print(f'\nSaved: {OUT_PATH}')

# Extra check for seed 42
if SEED == 42:
    print('\nSANITY (seed 42): expected SCM suff ~ 0.1469 / 0.1475 / 0.1454 '
          'at k=1/5/10%. Compare against Table 6.')
