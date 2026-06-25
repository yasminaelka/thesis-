#!/usr/bin/env python3
"""Recompute ONLY the SHAP faithfulness phase using already-trained models in
each seed dir. Rebuilds tokenizer/dataset/model identically to
run_contrastive_v2.py. Same SEED -> same sampled docs (comparable to old run).
Usage: python run_faithfulness_only.py --encoder legal-bert --pairs scm --seed 42"""
import os, json, random, time, warnings, argparse

# Hide warnings so the terminal output is less messy
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import shap

# Read command line arguments, so the script can be used for different seeds/runs
ap = argparse.ArgumentParser()
ap.add_argument('--encoder', default='legal-bert')
ap.add_argument('--pairs',   default='scm')
ap.add_argument('--seed',    type=int, default=42)
ap.add_argument('--root',    default='/gpfs/home6/yelkacemi/output')
args = ap.parse_args()

# Set the seed to make the sampling reproducible
SEED = args.seed
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# Use GPU if available, otherwise use CPU
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# Map encoder names to Hugging Face model names
ENCODER_MODELS = {'legal-bert':'nlpaueb/legal-bert-base-uncased','bert':'bert-base-uncased','roberta':'roberta-base'}
MODEL_NAME  = ENCODER_MODELS[args.encoder]

# Main experiment settings
N_LABELS    = 10
MAX_LEN     = 512
HEAD        = 256
TAIL        = 256
SHAP_N_DOCS = 200
SHAP_N_BG   = 10
K_VALUES    = [0.01, 0.05, 0.10]

# Build the output directory for this seed
RUN_TAG    = f'{args.encoder}_{args.pairs}_seed{SEED}'
OUTPUT_DIR = os.path.join(args.root, RUN_TAG)
print(f'Seed: {SEED}  Device: {DEVICE}  Dir: {OUTPUT_DIR}')

# Stop if the expected seed folder is missing
assert os.path.isdir(OUTPUT_DIR), f'missing dir {OUTPUT_DIR}'

# Load tokenizer for the selected encoder
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_head_tail(text):
    # Join paragraph lists into one string
    if isinstance(text, list):
        text = ' '.join(text)

    # Tokenize without truncation first, because head/tail truncation is manual
    tokens = TOKENIZER(text, truncation=False, add_special_tokens=True,
                       return_tensors='pt')
    ids  = tokens['input_ids'][0]
    mask = tokens['attention_mask'][0]

    # If the document is too long, keep the beginning and the end
    if len(ids) > MAX_LEN:
        ids  = torch.cat([ids[:HEAD],  ids[-TAIL:]])
        mask = torch.cat([mask[:HEAD], mask[-TAIL:]])

    # Pad shorter documents up to the maximum length
    pad_len = MAX_LEN - len(ids)
    if pad_len > 0:
        ids  = torch.cat([ids,  torch.zeros(pad_len, dtype=torch.long)])
        mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])

    return ids, mask

class ECTHRDataset(Dataset):
    def __init__(self, hf_split):
        # Store the Hugging Face split
        self.data = hf_split

    def __len__(self):
        # Return number of documents
        return len(self.data)

    def __getitem__(self, idx):
        # Get one example and tokenize it
        example = self.data[idx]
        input_ids, attention_mask = tokenize_head_tail(example['text'])

        # Create multi-label vector for the 10 article labels
        label_vector = torch.zeros(N_LABELS)
        for l in example['labels']:
            label_vector[l] = 1.0

        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': label_vector}

# Load train and test data
print('Loading dataset ...')
raw           = load_dataset('coastalcph/lex_glue', 'ecthr_a')
train_dataset = ECTHRDataset(raw['train'])
test_dataset  = ECTHRDataset(raw['test'])
print(f'Train: {len(train_dataset)} | Test: {len(test_dataset)}')

class BERTClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()

        # Load the transformer encoder
        self.bert       = AutoModel.from_pretrained(MODEL_NAME)

        # Final classification layer for the article labels
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        # Run input through the encoder
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=output_hidden_states)

        # Use the CLS token as the document representation
        cls    = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)

        # Return logits, and hidden states only if requested
        result = {'logits': logits}
        if output_hidden_states:
            result['hidden_states'] = outputs.hidden_states
        return result

def load_model(path):
    # Build the model architecture
    m = BERTClassifier(N_LABELS).to(DEVICE)

    # Load checkpoint from disk
    ckpt = torch.load(path, map_location=DEVICE)
    sd   = ckpt.get('model_state_dict', ckpt)

    # Load weights into the model
    res = m.load_state_dict(sd, strict=False)

    # Ignore position_ids, but raise an error for other missing keys
    missing = [k for k in res.missing_keys if 'position_ids' not in k]
    if missing:
        raise RuntimeError(f'state_dict missing keys loading {path}: {missing}')

    # Set to evaluation mode
    m.eval()
    return m

def make_predict_fn(model):
    # This wrapper is used by SHAP to get model probabilities
    def predict_fn(token_array):
        all_probs = []

        # Predict in batches to avoid memory issues
        for i in range(0, len(token_array), 32):
            chunk = torch.tensor(token_array[i:i+32], dtype=torch.long).to(DEVICE)
            attn  = (chunk != TOKENIZER.pad_token_id).long()

            # Convert logits to probabilities
            with torch.no_grad():
                out   = model(chunk, attn)
                probs = torch.sigmoid(out['logits']).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)

    return predict_fn

def compute_faithfulness(model, label=''):
    # Compute SHAP-based sufficiency and comprehensiveness
    print(f'\nFaithfulness: {label}')
    model.eval()

    # Create SHAP prediction function
    predict_fn = make_predict_fn(model)
    mask_id    = TOKENIZER.mask_token_id

    # Sample the same test documents and background documents for this seed
    rng        = np.random.default_rng(SEED)
    indices    = rng.choice(len(test_dataset), size=SHAP_N_DOCS, replace=False)
    bg_idx     = rng.choice(len(train_dataset), size=SHAP_N_BG, replace=False)
    background = np.stack([train_dataset[int(i)]['input_ids'].numpy() for i in bg_idx])

    # Create the SHAP KernelExplainer
    explainer  = shap.KernelExplainer(predict_fn, background)
    print('  KernelExplainer ready.')

    # Store scores separately for each k value
    results_by_k = {k: {'sufficiency': [], 'comprehensiveness': []} for k in K_VALUES}
    t_start = time.time()

    # Loop over sampled test documents
    for doc_num, idx in enumerate(indices):
        # Print progress every 20 documents
        if doc_num % 20 == 0 and doc_num > 0:
            el = (time.time()-t_start)/60
            print(f'  [{doc_num}/{SHAP_N_DOCS}] {el:.1f} min, ~{el/doc_num*(SHAP_N_DOCS-doc_num):.1f} min left')

        # Get one test document
        item      = test_dataset[int(idx)]
        input_ids = item['input_ids'].unsqueeze(0).to(DEVICE)
        attn_mask = item['attention_mask'].unsqueeze(0).to(DEVICE)
        ids_np    = input_ids.cpu().numpy()

        # Compute SHAP values for this document
        shap_vals        = explainer.shap_values(ids_np, nsamples=128, silent=True)

        # shap_vals is a single array (1, seq_len, n_classes).
        # Drop batch dim -> (seq_len, n_classes); mean |importance| over
        # classes -> (seq_len,), one score per TOKEN.
        sv_arr           = np.array(shap_vals)
        if sv_arr.ndim == 3:
            sv_arr = sv_arr[0]                       # (seq_len, n_classes)
        token_importance = np.abs(sv_arr).mean(axis=-1)  # (seq_len,)

        # Get real token positions and remove CLS at position 0
        attn_np  = attn_mask.squeeze(0).cpu().numpy()
        real_pos = np.where(attn_np == 1)[0]
        real_pos = real_pos[real_pos != 0]
        seq_len  = len(real_pos)

        # Skip documents that are too short
        if seq_len < 2:
            continue

        # Importance scores only for real tokens
        imp_real = token_importance[real_pos]

        # Original prediction probabilities
        with torch.no_grad():
            orig_prob = torch.sigmoid(model(input_ids, attn_mask)['logits']).cpu().numpy()

        # Test each k value
        for k in K_VALUES:
            # Number of tokens to keep/remove
            top_k_n   = max(1, int(seq_len * k))

            # Rank tokens by SHAP importance
            order     = np.argsort(imp_real)
            top_k_pos = set(real_pos[order[-top_k_n:]].tolist())
            non_top_k = [int(p) for p in real_pos if int(p) not in top_k_pos]

            # Sufficiency: keep top-k tokens and mask the rest
            suf_ids = ids_np.copy()
            if non_top_k:
                suf_ids[0, non_top_k] = mask_id
            suf_t = torch.tensor(suf_ids, dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                suf_prob = torch.sigmoid(model(suf_t, attn_mask)['logits']).cpu().numpy()
            sufficiency = float(np.abs(orig_prob - suf_prob).mean())

            # Comprehensiveness: mask the top-k tokens
            com_ids = ids_np.copy()
            com_ids[0, list(top_k_pos)] = mask_id
            com_t = torch.tensor(com_ids, dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                com_prob = torch.sigmoid(model(com_t, attn_mask)['logits']).cpu().numpy()
            comprehensiveness = float(np.abs(orig_prob - com_prob).mean())

            # Store document-level scores
            results_by_k[k]['sufficiency'].append(sufficiency)
            results_by_k[k]['comprehensiveness'].append(comprehensiveness)

    # Aggregate scores over documents
    aggregated, per_doc = {}, {}

    # Print small result table
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

# Lambda values to compare
LAMBDAS = [0.01, 0.05, 0.1, 0.5]

# Load path for baseline and lambda models
baseline_path = os.path.join(OUTPUT_DIR, 'contrastive_baseline.pt')
assert os.path.exists(baseline_path), baseline_path
lam_paths = {lam: os.path.join(OUTPUT_DIR, f'contrastive_lam{lam}.pt')
             for lam in LAMBDAS}

# Check that all lambda checkpoints exist
for lam, p in lam_paths.items():
    assert os.path.exists(p), p

# Store aggregated and per-document results
sweep_agg = {}
sweep_perdoc = {}

# First compute faithfulness for the baseline
print('\n===== baseline =====')
bm = load_model(baseline_path)
agg, perdoc = compute_faithfulness(bm, 'Baseline')
sweep_agg['baseline'] = agg
sweep_perdoc['baseline'] = perdoc
del bm
if DEVICE == 'cuda': torch.cuda.empty_cache()

# Then compute faithfulness for every SCM lambda
for lam in LAMBDAS:
    print(f'\n===== lambda = {lam} =====')
    sm = load_model(lam_paths[lam])
    agg, perdoc = compute_faithfulness(sm, f'SCM (lam={lam})')
    sweep_agg[str(lam)] = agg
    sweep_perdoc[str(lam)] = perdoc
    del sm
    if DEVICE == 'cuda': torch.cuda.empty_cache()

# Save the main aggregated results
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

# Save per-document results separately
with open(os.path.join(OUTPUT_DIR, 'lambda_sweep_faithfulness_perdoc.json'), 'w') as f:
    json.dump({'seed': SEED, 'per_doc': sweep_perdoc}, f, indent=2)
print(f'\nSaved: {OUT_PATH}')

# Print a compact k=0.01 summary (the Figure 9 column)
print('\n--- sufficiency at k=1% (Figure 9 column) ---')
print(f'  baseline   {sweep_agg["baseline"]["k=0.01"]["sufficiency"]:.4f}')
for lam in LAMBDAS:
    print(f'  lambda={lam:<4} {sweep_agg[str(lam)]["k=0.01"]["sufficiency"]:.4f}')
```
