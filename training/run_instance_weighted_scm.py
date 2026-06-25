```id="pob58h"
"""
Instance-weighted contrastive SCM training.

This script uses document-level weights for the SCM loss. A document gets a
higher weight when more SCM antonym pairs appear in the text window used by the
model.

Outputs are saved with instance_weighted_* names, so the normal contrastive
results are not overwritten.
"""

import os, json, random, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.metrics import f1_score
import shap

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ──────────────────────────────────────────────────────────────────────────────
# Paste the shared setup from run_contrastive.py here.
# The code below assumes that the model, data, constants, and evaluation
# functions already exist from that pasted section.
# ──────────────────────────────────────────────────────────────────────────────


# ### ===== INSTANCE WEIGHTING: NEW CODE (start) ===== ###

# Count SCM-pair matches per training document and turn them into weights.
def _pair_id_tensors(scm_pairs):
    """Tokenise SCM pairs once."""
    out = []
    for pos_word, neg_word in scm_pairs:
        pos_ids = TOKENIZER.encode(pos_word, add_special_tokens=False)
        neg_ids = TOKENIZER.encode(neg_word, add_special_tokens=False)
        out.append((torch.tensor(pos_ids), torch.tensor(neg_ids)))
    return out

def count_scm_pairs_in_ids(input_ids_1d, pair_id_tensors):
    """Count how many SCM pairs appear in one document."""
    cnt = 0
    for pos_t, neg_t in pair_id_tensors:
        has_pos = torch.isin(input_ids_1d, pos_t).any()
        has_neg = torch.isin(input_ids_1d, neg_t).any()
        if has_pos and has_neg:
            cnt += 1
    return cnt

def precompute_instance_weights(hf_train_split, scm_pairs):
    """Compute one weight per training document."""
    pair_tensors = _pair_id_tensors(scm_pairs)
    counts = np.zeros(len(hf_train_split), dtype=np.float32)
    for i in range(len(hf_train_split)):
        ids, _ = tokenize_head_tail(hf_train_split[i]['text'])
        counts[i] = count_scm_pairs_in_ids(ids, pair_tensors)
        if (i + 1) % 1000 == 0:
            print(f'  w_i precompute: {i+1}/{len(hf_train_split)}')
    max_count = counts.max() if counts.max() > 0 else 1.0
    w = counts / max_count

    # Save a small summary to check that the weights make sense.
    dist = {
        'max_pair_count'      : float(counts.max()),
        'mean_pair_count'     : float(counts.mean()),
        'frac_docs_zero_pairs': float((counts == 0).mean()),
        'mean_w_i'            : float(w.mean()),
        'std_w_i'             : float(w.std()),
        'max_w_i'             : float(w.max()),
        'n_docs'              : int(len(counts)),
    }
    print('  w_i distribution:', json.dumps(dist, indent=2))
    return torch.tensor(w, dtype=torch.float32), dist

# Same SCM loss idea, but scaled by each document's weight.
def compute_scm_loss_instance_weighted(hidden_states, input_ids, scm_pairs,
                                       batch_w, margin=0.5):
    """
    Instance-weighted SCM loss for one batch.
    """
    device = input_ids.device
    per_doc_losses = []
    B = input_ids.shape[0]

    # Tokenise pairs for the current device.
    pair_tensors = []
    for pos_word, neg_word in scm_pairs:
        pos_ids = torch.tensor(TOKENIZER.encode(pos_word, add_special_tokens=False),
                               device=device)
        neg_ids = torch.tensor(TOKENIZER.encode(neg_word, add_special_tokens=False),
                               device=device)
        pair_tensors.append((pos_ids, neg_ids))

    for b in range(B):
        doc_terms = []
        ids_b = input_ids[b]
        for pos_t, neg_t in pair_tensors:
            pos_mask = torch.isin(ids_b, pos_t)
            neg_mask = torch.isin(ids_b, neg_t)
            if pos_mask.any() and neg_mask.any():
                h_pos = hidden_states[b][pos_mask].mean(dim=0)
                h_neg = hidden_states[b][neg_mask].mean(dim=0)
                sim   = F.cosine_similarity(h_pos.unsqueeze(0), h_neg.unsqueeze(0))
                doc_terms.append(torch.clamp(sim + margin, min=0.0))
        if doc_terms:
            doc_loss = torch.stack(doc_terms).mean() * batch_w[b]
            per_doc_losses.append(doc_loss)
    if len(per_doc_losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(per_doc_losses).mean()
# ### ===== INSTANCE WEIGHTING: NEW CODE (end) ===== ###


# Dataset that also returns the document weight.
class ECTHRDatasetWeighted(Dataset):
    def __init__(self, hf_split, weights):
        self.data = hf_split
        self.weights = weights
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        example = self.data[idx]
        input_ids, attention_mask = tokenize_head_tail(example['text'])
        label_vector = torch.zeros(N_LABELS)
        for l in example['labels']:
            label_vector[l] = 1.0
        return {
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'labels'        : label_vector,
            'w_i'           : self.weights[idx],
        }

# ── Training loop (instance-weighted) ─────────────────────────────────────────
def train_epoch_iw(model, loader, optimizer, lam=0.0, use_scm=False):
    model.train()
    total_loss, total_ce, total_scm = 0.0, 0.0, 0.0
    for batch_idx, batch in enumerate(loader):
        ids    = batch['input_ids'].to(DEVICE)
        mask   = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        w_i    = batch['w_i'].to(DEVICE)
        out    = model(ids, mask, output_hidden_states=use_scm)
        logits = out['logits']
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=CLASS_WEIGHTS)
        if use_scm:
            hs = out['hidden_states'][-1]
            scm_loss = compute_scm_loss_instance_weighted(
                hs, ids, SCM_PAIRS, w_i, MARGIN)
            loss = ce_loss + lam * scm_loss
        else:
            scm_loss = torch.tensor(0.0)
            loss = ce_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item(); total_ce += ce_loss.item()
        total_scm  += scm_loss.item()
        if (batch_idx + 1) % 200 == 0:
            print(f'  Batch {batch_idx+1}/{len(loader)} | Total: {loss.item():.4f} '
                  f'| CE: {ce_loss.item():.4f} | SCM(w): {scm_loss.item():.4f}')
    n = len(loader)
    return total_loss/n, total_ce/n, total_scm/n


# ── Orchestration ─────────────────────────────────────────────────────────────
def main():
    print('Precomputing instance weights w_i over training set ...')
    train_weights, wi_dist = precompute_instance_weights(raw['train'], SCM_PAIRS)
    with open(os.path.join(OUTPUT_DIR, 'instance_weighted_wi_distribution.json'),
              'w') as f:
        json.dump(wi_dist, f, indent=2)

    # Stop if the weights do not vary.
    if wi_dist['std_w_i'] < 1e-6:
        raise RuntimeError('w_i has zero variance — weighting would be a no-op. '
                           'Check SCM pair detection before training.')

    # Only the training loader needs weights.
    train_ds_w = ECTHRDatasetWeighted(raw['train'], train_weights)
    train_loader_w = DataLoader(train_ds_w, batch_size=BATCH_SIZE, shuffle=True)

    # Train one instance-weighted model for each lambda.
    grid_results = {}
    for lam in LAMBDAS:
        print(f'\n{"="*55}\nInstance-weighted SCM  λ={lam}\n{"="*55}')
        model = build_model().to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        best_val = -1.0
        save_path = os.path.join(OUTPUT_DIR, f'instance_weighted_lam{lam}.pt')
        for epoch in range(EPOCHS):
            tr, ce, sc = train_epoch_iw(model, train_loader_w, optimizer,
                                        lam=lam, use_scm=True)
            val_f1 = eval_macro_f1(model, val_loader)
            print(f'  Epoch {epoch+1}/{EPOCHS} | Train {tr:.4f} | CE {ce:.4f} '
                  f'| SCM(w) {sc:.4f} | Val F1 {val_f1:.4f}')
            if val_f1 > best_val:
                best_val = val_f1
                torch.save({'model_state_dict': model.state_dict(),
                            'val_f1': val_f1, 'lambda': lam}, save_path)
        grid_results[str(lam)] = {'val_f1': best_val, 'lambda': lam}

    with open(os.path.join(OUTPUT_DIR, 'instance_weighted_grid_results.json'),
              'w') as f:
        json.dump(grid_results, f, indent=2)

    best_lam = max(grid_results, key=lambda k: grid_results[k]['val_f1'])
    print(f'\nBest instance-weighted λ = {best_lam} '
          f'(val F1 = {grid_results[best_lam]["val_f1"]:.4f})')

    # Load the best model for the normal evaluation steps.
    best_model = load_model(os.path.join(OUTPUT_DIR,
                                         f'instance_weighted_lam{best_lam}.pt'))

    # Use the existing evaluation functions from the main script.
    print('Now run your existing performance/fairness/faithfulness evaluators on '
          'best_model and dump to instance_weighted_*.json')

if __name__ == '__main__':
    main()
```
