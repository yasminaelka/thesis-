"""
Wilcoxon tests for per-document sufficiency.

This script compares baseline and SCM sufficiency scores on sampled ECtHR test
documents and saves the paired Wilcoxon results.
"""
import os, json, random, warnings

# Keep output cleaner
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from scipy.stats import wilcoxon

# Set seed
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# Use GPU if available
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# Print GPU name
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Output folder
OUTPUT_DIR  = '/gpfs/home6/yelkacemi/thesis_outputs'

# Base model
MODEL_NAME  = 'nlpaueb/legal-bert-base-uncased'

# Number of labels
N_LABELS    = 10

# Maximum sequence length
MAX_LEN     = 512

# Number of sampled test documents
SHAP_N_DOCS = 200

# k values
K_VALUES    = [0.01, 0.05, 0.10]

# Article labels
ARTICLE_NAMES     = ['Art.2','Art.3','Art.5','Art.6','Art.8','Art.9',
                     'Art.10','Art.11','Art.14','P1-1']

# Reliable articles for article-level tests
RELIABLE_ARTICLES = ['Art.2','Art.3','Art.5','Art.6','Art.8','P1-1']

# Corrected significance threshold
ALPHA_CORRECTED   = 0.005

class LegalBertClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder    = AutoModel.from_pretrained(MODEL_NAME)

        # Dropout before classifier
        self.dropout    = nn.Dropout(0.1)

        # Classification layer
        self.classifier = nn.Linear(self.encoder.config.hidden_size, N_LABELS)

    def forward(self, input_ids, attention_mask):
        # Forward pass
        out  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS representation
        pool = out.last_hidden_state[:, 0, :]

        # Return logits
        return self.classifier(self.dropout(pool))

class ECtHRDataset(Dataset):
    def __init__(self, tokenizer):
        # Load test split
        ds = load_dataset('coastalcph/lex_glue', 'ecthr_a', split='test')

        # Store tokenized examples
        self.items = []

        # Tokenize all examples
        for ex in ds:
            # Join text parts if needed
            txt = ' '.join(ex['text']) if isinstance(ex['text'], list) else ex['text']

            # Tokenize and pad/truncate
            enc = tokenizer(txt, max_length=MAX_LEN, truncation=True,
                            padding='max_length', return_tensors='pt')

            # Create label vector
            label = torch.zeros(N_LABELS)

            # Mark active labels
            for l in ex['labels']:
                if l < N_LABELS:
                    label[l] = 1.0

            # Store tensors
            self.items.append({
                'input_ids':      enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'labels':         label,
            })

    # Number of examples
    def __len__(self):        return len(self.items)

    # One example
    def __getitem__(self, i): return self.items[i]

def load_model(path):
    # Build model
    model = LegalBertClassifier().to(DEVICE)

    # Load checkpoint
    state = torch.load(path, map_location=DEVICE)

    # Get weights if saved inside model_state_dict
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    # Load weights
    model.load_state_dict(state, strict=False)

    # Evaluation mode
    model.eval()

    return model

def compute_sufficiency(model, docs, k_values):
    """
    Compute sufficiency scores for sampled documents.
    """
    # Mask token ID
    mask_id = tokenizer.mask_token_id

    # Store scores per k
    results = {k: [] for k in k_values}

    # Evaluation only
    with torch.no_grad():
        # Loop over documents
        for idx, doc in enumerate(docs):
            # Print progress
            if idx % 20 == 0:
                print(f'  doc {idx}/{len(docs)}')

            # Add batch dimension
            ids  = doc['input_ids'].unsqueeze(0).to(DEVICE)    # (1,512)
            amsk = doc['attention_mask'].unsqueeze(0).to(DEVICE)

            # Original probabilities
            probs_orig = torch.sigmoid(model(ids, amsk)).squeeze(0).cpu().numpy()

            # Active labels for this document
            active = np.where(doc['labels'].numpy() > 0.5)[0]

            # Use all labels if none are active
            if len(active) == 0:
                active = np.arange(N_LABELS)

            # Get real token positions
            sep_id   = tokenizer.sep_token_id
            all_ids  = ids.squeeze(0).cpu().numpy()

            real_pos = [
                p for p in amsk.squeeze(0).cpu().numpy().nonzero()[0]
                if p != 0 and all_ids[p] != sep_id
            ]
            real_pos = np.array(real_pos)

            # Number of real tokens
            seq_len  = len(real_pos)

            # Evaluate each k
            for k in k_values:
                # Number of tokens to keep
                top_n      = max(1, int(np.ceil(k * seq_len)))

                # Keep first tokens
                keep_pos   = set(real_pos[:top_n].tolist())

                # Copy input IDs
                ids_masked = ids.clone()

                # Mask tokens not kept
                for pos in real_pos:
                    if int(pos) not in keep_pos:
                        ids_masked[0, int(pos)] = mask_id

                # Masked probabilities
                probs_masked = torch.sigmoid(
                    model(ids_masked, amsk)).squeeze(0).cpu().numpy()

                # Sufficiency score
                suff = float(np.mean(
                    probs_orig[active] - probs_masked[active]))

                # Store score
                results[k].append(suff)

    # Convert to arrays
    return {k: np.array(v) for k, v in results.items()}

# ── Load data
print('Loading tokenizer...')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print('Loading dataset...')

# Load test dataset
test_ds = ECtHRDataset(tokenizer)

# Random generator
rng     = np.random.default_rng(SEED)

# Sample documents
indices = rng.choice(len(test_ds), size=SHAP_N_DOCS, replace=False)

# Get sampled documents
docs    = [test_ds[int(i)] for i in indices]

# Store labels
labels  = np.array([d['labels'].numpy() for d in docs])

print(f'Sampled {len(docs)} documents.')

# ── Load models and compute
print('\n=== BASELINE ===')

# Load baseline
baseline_model = load_model(
    os.path.join(OUTPUT_DIR, 'baseline_model.pt'))

# Baseline sufficiency
base_suff = compute_sufficiency(baseline_model, docs, K_VALUES)

# Clear baseline model
del baseline_model; torch.cuda.empty_cache()

print('\n=== SCM (λ=0.1) ===')

# Load SCM model
scm_model = load_model(
    os.path.join(OUTPUT_DIR, 'contrastive_lam0.1.pt'))

# SCM sufficiency
scm_suff = compute_sufficiency(scm_model, docs, K_VALUES)

# Clear SCM model
del scm_model; torch.cuda.empty_cache()

# ── Save raw arrays

# Store raw scores
raw = {'doc_indices': indices.tolist(), 'doc_labels': labels.tolist(),
       'article_names': ARTICLE_NAMES}

# Add scores for each k
for k in K_VALUES:
    raw[f'k={k}'] = {'baseline': base_suff[k].tolist(),
                     'scm':      scm_suff[k].tolist()}

# Save raw arrays
with open(os.path.join(OUTPUT_DIR, 'per_instance_sufficiency.json'), 'w') as f:
    json.dump(raw, f, indent=2)

print('\nPer-instance arrays saved.')

# ── Wilcoxon tests
print('\n=== WILCOXON SIGNED-RANK TESTS ===')
print(f'Bonferroni α\' = {ALPHA_CORRECTED}')

# Store test results
wilcoxon_results = {}

# Run tests for each k
for k in K_VALUES:
    # Scores for this k
    ba, sa = base_suff[k], scm_suff[k]

    # Store results for this k
    wilcoxon_results[f'k={k}'] = {}

    print(f'\n─── k={int(k*100)}% ───')

    # Test macro and article levels
    for level in ['macro'] + RELIABLE_ARTICLES:
        # Macro uses all sampled documents
        if level == 'macro':
            bv, sv = ba, sa

        # Article level uses documents with that article
        else:
            art_idx = ARTICLE_NAMES.index(level)
            msk     = labels[:, art_idx] > 0.5
            bv, sv  = ba[msk], sa[msk]

        # Number of pairs
        n = len(bv)

        # Skip if too small
        if n < 10:
            print(f'  {level:<8} n={n:3d}  SKIPPED (n<10)')
            wilcoxon_results[f'k={k}'][level] = {'n': n, 'skipped': True}
            continue

        # Difference between SCM and baseline
        diffs = sv - bv

        # Skip if all differences are zero
        if np.all(diffs == 0):
            print(f'  {level:<8} n={n:3d}  SKIPPED (all diffs zero)')
            wilcoxon_results[f'k={k}'][level] = {'n': n, 'skipped': True}
            continue

        # Paired Wilcoxon test
        stat, p = wilcoxon(bv, sv, alternative='two-sided')

        # Mean difference
        diff    = float(diffs.mean())

        # Significance flag
        sig     = p < ALPHA_CORRECTED

        # Save statistics
        wilcoxon_results[f'k={k}'][level] = {
            'n': n, 'baseline_mean': float(bv.mean()),
            'scm_mean': float(sv.mean()), 'mean_diff': diff,
            'statistic': float(stat), 'p_value': float(p),
            'significant': bool(sig)
        }

        # Print result
        print(f'  {level:<8} n={n:3d}  '
              f'base={bv.mean():.4f}  scm={sv.mean():.4f}  '
              f'Δ={diff:+.4f}  W={stat:.1f}  p={p:.4f}  '
              f'{"✓ sig" if sig else "ns"}')

# Save Wilcoxon results
with open(os.path.join(OUTPUT_DIR, 'wilcoxon_results.json'), 'w') as f:
    json.dump(wilcoxon_results, f, indent=2)

print('\nResults saved to wilcoxon_results.json')

# Print thesis-style table for k=1%
print('\n=== THESIS TABLE (k=1%) ===')
print(f'{"Level":<10}{"n":>4}{"Baseline":>10}{"SCM":>10}{"Δ":>8}{"p":>10}{"Sig":>5}')
print('─'*52)

# Show macro and reliable-article rows
for level in ['macro'] + RELIABLE_ARTICLES:
    # Get saved result
    r = wilcoxon_results['k=0.01'].get(level, {})

    # Skipped row
    if r.get('skipped'):
        print(f'{level:<10}{r.get("n",0):>4}  {"SKIPPED":>35}')

    # Normal row
    else:
        print(f'{level:<10}{r["n"]:>4}{r["baseline_mean"]:>10.4f}'
              f'{r["scm_mean"]:>10.4f}{r["mean_diff"]:>+8.4f}'
              f'{r["p_value"]:>10.4f}{"✓":>5 if r["significant"] else "":>5}')
