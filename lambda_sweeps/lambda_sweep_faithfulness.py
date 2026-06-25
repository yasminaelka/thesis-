#!/usr/bin/env python3
"""
Recompute SHAP faithfulness for already trained models.

This script loads the saved baseline and SCM models for one seed and compares
their SHAP sufficiency and comprehensiveness scores.
"""

import os, json, random, time, warnings, argparse

# Keep terminal output cleaner
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import shap

# Read run settings
ap = argparse.ArgumentParser()
ap.add_argument('--encoder', default='legal-bert')
ap.add_argument('--pairs',   default='scm')
ap.add_argument('--seed',    type=int, default=42)
ap.add_argument('--root',    default='/gpfs/home6/yelkacemi/output')
args = ap.parse_args()

# Set seed for reproducibility
SEED = args.seed
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# Use GPU if available
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# Pick encoder model
ENCODER_MODELS = {'legal-bert':'nlpaueb/legal-bert-base-uncased','bert':'bert-base-uncased','roberta':'roberta-base'}
MODEL_NAME  = ENCODER_MODELS[args.encoder]

# Main settings
N_LABELS    = 10
MAX_LEN     = 512
HEAD        = 256
TAIL        = 256
SHAP_N_DOCS = 200
SHAP_N_BG   = 10
K_VALUES    = [0.01, 0.05, 0.10]

# Output folder for this run
RUN_TAG    = f'{args.encoder}_{args.pairs}_seed{SEED}'
OUTPUT_DIR = os.path.join(args.root, RUN_TAG)
print(f'Seed: {SEED}  Device: {DEVICE}  Dir: {OUTPUT_DIR}')

# Check that the folder exists
assert os.path.isdir(OUTPUT_DIR), f'missing dir {OUTPUT_DIR}'

# Load tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_head_tail(text):
    # Join text parts if needed
    if isinstance(text, list):
        text = ' '.join(text)

    # Tokenize first, then truncate manually
    tokens = TOKENIZER(text, truncation=False, add_special_tokens=True,
                       return_tensors='pt')
    ids  = tokens['input_ids'][0]
    mask = tokens['attention_mask'][0]

    # Keep beginning and end for long documents
    if len(ids) > MAX_LEN:
        ids  = torch.cat([ids[:HEAD],  ids[-TAIL:]])
        mask = torch.cat([mask[:HEAD], mask[-TAIL:]])

    # Pad shorter documents
    pad_len = MAX_LEN - len(ids)
    if pad_len > 0:
        ids  = torch.cat([ids,  torch.zeros(pad_len, dtype=torch.long)])
        mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])

    return ids, mask

class ECTHRDataset(Dataset):
    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Tokenize one example
        example = self.data[idx]
        input_ids, attention_mask = tokenize_head_tail(example['text'])

        # Make multi-label vector
        label_vector = torch.zeros(N_LABELS)
        for l in example['labels']:
            label_vector[l] = 1.0

        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': label_vector}

# Load data
print('Loading dataset ...')
raw           = load_dataset('coastalcph/lex_glue', 'ecthr_a')
train_dataset = ECTHRDataset(raw['train'])
test_dataset  = ECTHRDataset(raw['test'])
print(f'Train: {len(train_dataset)} | Test: {len(test_dataset)}')

class BERTClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()

        # Encoder and classifier head
        self.bert       = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        # Forward pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=output_hidden_states)

        # Use CLS representation
        cls    = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)

        result = {'logits': logits}
        if output_hidden_states:
            result['hidden_states'] = outputs.hidden_states
        return result

def load_model(path):
    # Build model
    m = BERTClassifier(N_LABELS).to(DEVICE)

    # Load checkpoint
    ckpt = torch.load(path, map_location=DEVICE)
    sd   = ckpt.get('model_state_dict', ckpt)

    # Load weights
    res = m.load_state_dict(sd, strict=False)

    # Check for missing keys
    missing = [k for k in res.missing_keys if 'position_ids' not in k]
    if missing:
        raise RuntimeError(f'state_dict missing keys loading {path}: {missing}')

    m.eval()
    return m

def make_predict_fn(model):
    # Wrapper for SHAP predictions
    def predict_fn(token_array):
        all_probs = []

        # Predict in batches
        for i in range(0, len(token_array), 32):
            chunk = torch.tensor(token_array[i:i+32], dtype=torch.long).to(DEVICE)
            attn  = (chunk != TOKENIZER.pad_token_id).long()

            with torch.no_grad():
                out   = model(chunk, attn)
                probs = torch.sigmoid(out['logits']).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)

    return predict_fn

def compute_faithfulness(model, label=''):
    # Compute SHAP faithfulness scores
    print(f'\nFaithfulness: {label}')
    model.eval()

    predict_fn = make_predict_fn(model)
    mask_id    = TOKENIZER.mask_token_id

    # Sample test documents and background documents
    rng        = np.random.default_rng(SEED)
    indices    = rng.choice(len(test_dataset), size=SHAP_N_DOCS, replace=False)
    bg_idx     = rng.choice(len(train_dataset), size=SHAP_N_BG, replace=False)
    background = np.stack([train_dataset[int(i)]['input_ids'].numpy() for i in bg_idx])

    # Create SHAP explainer
    explainer  = shap.KernelExplainer(predict_fn, background)
    print('  KernelExplainer ready.')

    # Store scores per k
    results_by_k = {k: {'sufficiency': [], 'comprehensiveness': []} for k in K_VALUES}
    t_start = time.time()

    # Loop over sampled documents
    for doc_num, idx in enumerate(indices):
        # Print progress
        if doc_num % 20 == 0 and doc_num > 0:
            el = (time.time()-t_start)/60
            print(f'  [{doc_num}/{SHAP_N_DOCS}] {el:.1f} min, ~{el/doc_num*(SHAP_N_DOCS-doc_num):.1f} min left')

        # Get one test item
        item      = test_dataset[int(idx)]
        input_ids = item['input_ids'].unsqueeze(0).to(DEVICE)
        attn_mask = item['attention_mask'].unsqueeze(0).to(DEVICE)
        ids_np    = input_ids.cpu().numpy()

        # Compute SHAP values
        shap_vals        = explainer.shap_values(ids_np, nsamples=128, silent=True)

        # Convert SHAP values to token importance
        sv_arr           = np.array(shap_vals)
        if sv_arr.ndim == 3:
            sv_arr = sv_arr[0]                       # (seq_len, n_classes)
        token_importance = np.abs(sv_arr).mean(axis=-1)  # (seq_len,)

        # Only use real tokens, excluding CLS
        attn_np  = attn_mask.squeeze(0).cpu().numpy()
        real_pos = np.where(attn_np == 1)[0]
        real_pos = real_pos[real_pos != 0]
        seq_len  = len(real_pos)

        # Skip very short documents
        if seq_len < 2:
            continue

        imp_real = token_importance[real_pos]

        # Original probabilities
        with torch.no_grad():
            orig_prob = torch.sigmoid(model(input_ids, attn_mask)['logits']).cpu().numpy()

        # Evaluate each k
        for k in K_VALUES:
            # Number of tokens to keep or mask
            top_k_n   = max(1, int(seq_len * k))

            # Rank tokens by importance
            order     = np.argsort(imp_real)
            top_k_pos = set(real_pos[order[-top_k_n:]].tolist())
            non_top_k = [int(p) for p in real_pos if int(p) not in top_k_pos]

            # Sufficiency
            suf_ids = ids_np.copy()
            if non_top_k:
                suf_ids[0, non_top_k] = mask_id
            suf_t = torch.tensor(suf_ids, dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                suf_prob = torch.sigmoid(model(suf_t, attn_mask)['logits']).cpu().numpy()
            sufficiency = float(np.abs(orig_prob - suf_prob).mean())

            # Comprehensiveness
            com_ids = ids_np.copy()
            com_ids[0, list(top_k_pos)] = mask_id
            com_t = torch.tensor(com_ids, dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                com_prob = torch.sigmoid(model(com_t, attn_mask)['logits']).cpu().numpy()
            comprehensiveness = float(np.abs(orig_prob - com_prob).mean())

            # Save scores
            results_by_k[k]['sufficiency'].append(sufficiency)
            results_by_k[k]['comprehensiveness'].append(comprehensiveness)

    # Aggregate results
    aggregated, per_doc = {}, {}

    # Print result table
    print(f'\n  {"k":6s}  {"Sufficiency":>12s}  {"Comprehensiveness":>18s}')
    print('  ' + '-'*40)

    for k in K_VALUES:
        sa, ca = results_by_k[k]['sufficiency'], results_by_k[k]['comprehensiveness']
        suf, comp = float(np.mean(sa)), float(np.mean(ca))
        aggregated[f'k={k}'] = {'sufficiency': suf, 'comprehensiveness': comp}
        per_doc[f'k={k}']    = {'sufficiency': sa, 'comprehensiveness': ca}
        print(f'  {k:6.2f}  {suf:>12.4f}  {comp:>18.4f}')

    print(f'  Total: {(time.time()-t_start)/60:.1f} min')
    return aggregated, per_doc

# Lambdas to compare
LAMBDAS = [0.01, 0.05, 0.1, 0.5]

# Model paths
baseline_path = os.path.join(OUTPUT_DIR, 'contrastive_baseline.pt')
assert os.path.exists(baseline_path), baseline_path
lam_paths = {lam: os.path.join(OUTPUT_DIR, f'contrastive_lam{lam}.pt')
             for lam in LAMBDAS}

# Check lambda checkpoints
for lam, p in lam_paths.items():
    assert os.path.exists(p), p

# Store results
sweep_agg = {}
sweep_perdoc = {}

# Baseline faithfulness
print('\n===== baseline =====')
bm = load_model(baseline_path)
agg, perdoc = compute_faithfulness(bm, 'Baseline')
sweep_agg['baseline'] = agg
sweep_perdoc['baseline'] = perdoc
del bm
if DEVICE == 'cuda': torch.cuda.empty_cache()

# SCM faithfulness for each lambda
for lam in LAMBDAS:
    print(f'\n===== lambda = {lam} =====')
    sm = load_model(lam_paths[lam])
    agg, perdoc = compute_faithfulness(sm, f'SCM (lam={lam})')
    sweep_agg[str(lam)] = agg
    sweep_perdoc[str(lam)] = perdoc
    del sm
    if DEVICE == 'cuda': torch.cuda.empty_cache()

# Save aggregated results
out = {
    'seed': SEED,
    'encoder': args.encoder,
    'pairs': args.pairs,
    'shap_n_docs': SHAP_N_DOCS,
    'shap_n_bg': SHAP_N_BG,
    'k_values': K_VALUES,
    'lambdas': LAMBDAS,
    'aggregated': sweep_agg,
}
OUT_PATH = os.path.join(OUTPUT_DIR, 'lambda_sweep_faithfulness.json')
with open(OUT_PATH, 'w') as f:
    json.dump(out, f, indent=2)

# Save per-document results
with open(os.path.join(OUTPUT_DIR, 'lambda_sweep_faithfulness_perdoc.json'), 'w') as f:
    json.dump({'seed': SEED, 'per_doc': sweep_perdoc}, f, indent=2)
print(f'\nSaved: {OUT_PATH}')

# Print small summary
print('\n--- sufficiency at k=1% (Figure 9 column) ---')
print(f'  baseline   {sweep_agg["baseline"]["k=0.01"]["sufficiency"]:.4f}')
for lam in LAMBDAS:
    print(f'  lambda={lam:<4} {sweep_agg[str(lam)]["k=0.01"]["sufficiency"]:.4f}')
