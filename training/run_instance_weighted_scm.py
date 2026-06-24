"""
Instance-Weighted Contrastive SCM — Training & Evaluation
Yasmina El Kacemi — University of Amsterdam

PURPOSE
-------
This is the GENUINELY instance-weighted variant of the contrastive SCM model.
It implements Equation 2 of the thesis: each document's SCM loss is scaled by
w_i = (number of SCM antonym pairs present in document i) / (max count over the
training set). Documents with more stereotypical antonym pairs receive stronger
regularisation pressure; documents with none contribute zero SCM loss.

This differs from run_contrastive.py (the UNIFORM variant), where every
(pair, document) occurrence is pooled into one flat batch-level mean with no
per-document weighting.

HOW TO USE
----------
1. Copy the marked HEADER region (imports, constants, tokenizer, dataset,
   model class, class weights, eval helpers, fairness + faithfulness functions)
   from your WORKING run_contrastive.py into the places marked
   `### >>> PASTE FROM run_contrastive.py <<<`.
   Do NOT retype them; copy verbatim so nothing drifts.
2. The only NEW / CHANGED code is clearly fenced with
   `### ===== INSTANCE WEIGHTING: NEW CODE ===== ###` markers.
3. All outputs are saved as instance_weighted_*.json so your existing
   contrastive_*.json results are never overwritten.

Outputs:
  instance_weighted_grid_results.json
  instance_weighted_performance.json
  instance_weighted_fairness.json
  instance_weighted_faithfulness.json
  instance_weighted_wi_distribution.json   (sanity-check on the weights)
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
# ### >>> PASTE FROM run_contrastive.py <<< : HEADER REGION
# Paste, verbatim, everything from your working script that defines:
#   - all constants: MODEL_NAME, MAX_LEN, HEAD, TAIL, BATCH_SIZE, LR, N_LABELS,
#     EPOCHS, MARGIN, LAMBDAS, OUTPUT_DIR, ARTICLE_NAMES, etc.
#   - ALL_WARMTH_PAIRS, ALL_COMPETENCE_PAIRS, SCM_PAIRS
#   - GENDER_KEYWORDS, ETHNICITY_KEYWORDS
#   - TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
#   - tokenize_head_tail(text)
#   - class ECTHRDataset(Dataset)
#   - the dataset loading lines (raw = load_dataset('lex_glue','ecthr_a'); splits)
#   - the model class (the one with forward(input_ids, attention_mask,
#     output_hidden_states=False) returning {'logits':..., 'hidden_states':...})
#   - the class-weight block (pos_counts/neg_counts -> CLASS_WEIGHTS)
#   - eval_macro_f1 / per-article threshold tuning helpers
#   - the fairness evaluation function (gender+ethnicity, DPD+DI)
#   - the SHAP faithfulness function compute_faithfulness(...)
#   - load_model(path)
#
# After pasting, the names used below must exist:
#   SCM_PAIRS, TOKENIZER, MAX_LEN, HEAD, TAIL, train_dataset, raw,
#   ECTHRDataset, train_loader/val_loader/test_loader, CLASS_WEIGHTS,
#   MARGIN, LAMBDAS, OUTPUT_DIR, BATCH_SIZE, LR, EPOCHS, model factory, etc.
# ──────────────────────────────────────────────────────────────────────────────


# ### ===== INSTANCE WEIGHTING: NEW CODE (start) ===== ###
# Precompute, for every TRAIN document, how many of the 38 SCM antonym pairs
# have BOTH words present in the document's head+tail token window. Then
# normalise by the max count across the training set -> w_i in [0, 1].
#
# We count on the SAME 512-token head+tail window the model actually sees,
# so the weight reflects the stereotypical content the model is exposed to.
# The weight is a stable per-document property (computed once, not per batch).

def _pair_id_tensors(scm_pairs):
    """Pre-tokenise each antonym pair once."""
    out = []
    for pos_word, neg_word in scm_pairs:
        pos_ids = TOKENIZER.encode(pos_word, add_special_tokens=False)
        neg_ids = TOKENIZER.encode(neg_word, add_special_tokens=False)
        out.append((torch.tensor(pos_ids), torch.tensor(neg_ids)))
    return out

def count_scm_pairs_in_ids(input_ids_1d, pair_id_tensors):
    """Number of antonym pairs where BOTH words appear in this document."""
    cnt = 0
    for pos_t, neg_t in pair_id_tensors:
        has_pos = torch.isin(input_ids_1d, pos_t).any()
        has_neg = torch.isin(input_ids_1d, neg_t).any()
        if has_pos and has_neg:
            cnt += 1
    return cnt

def precompute_instance_weights(hf_train_split, scm_pairs):
    """Return a tensor w of length len(train) with w_i in [0,1], plus raw counts."""
    pair_tensors = _pair_id_tensors(scm_pairs)
    counts = np.zeros(len(hf_train_split), dtype=np.float32)
    for i in range(len(hf_train_split)):
        ids, _ = tokenize_head_tail(hf_train_split[i]['text'])
        counts[i] = count_scm_pairs_in_ids(ids, pair_tensors)
        if (i + 1) % 1000 == 0:
            print(f'  w_i precompute: {i+1}/{len(hf_train_split)}')
    max_count = counts.max() if counts.max() > 0 else 1.0
    w = counts / max_count
    # Sanity log: the weights MUST vary, otherwise this is just uniform.
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

# Instance-weighted contrastive SCM loss.
# Difference vs uniform: accumulate loss PER DOCUMENT, scale each document's
# mean pair-loss by that document's w_i, then average over documents in the batch.
def compute_scm_loss_instance_weighted(hidden_states, input_ids, scm_pairs,
                                       batch_w, margin=0.5):
    """
    hidden_states : [B, T, H] last hidden layer
    input_ids     : [B, T]
    batch_w       : [B] per-document weights w_i for this batch
    """
    device = input_ids.device
    per_doc_losses = []
    B = input_ids.shape[0]
    # Pre-tokenise pairs once per call.
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
            # mean over the pairs present in this doc, scaled by this doc's w_i
            doc_loss = torch.stack(doc_terms).mean() * batch_w[b]
            per_doc_losses.append(doc_loss)
    if len(per_doc_losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    # average over documents that contributed (those with >=1 pair)
    return torch.stack(per_doc_losses).mean()
# ### ===== INSTANCE WEIGHTING: NEW CODE (end) ===== ###


# We need w_i available inside the training loop, indexed by document.
# Build a weighted dataset that returns w_i alongside each example.
class ECTHRDatasetWeighted(Dataset):
    def __init__(self, hf_split, weights):
        self.data = hf_split
        self.weights = weights  # tensor aligned with hf_split order
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

    # ABORT GUARD: if weights don't vary, this would silently reduce to uniform.
    if wi_dist['std_w_i'] < 1e-6:
        raise RuntimeError('w_i has zero variance — weighting would be a no-op. '
                           'Check SCM pair detection before training.')

    # Weighted train loader (val/test do not need weights for eval).
    train_ds_w = ECTHRDatasetWeighted(raw['train'], train_weights)
    train_loader_w = DataLoader(train_ds_w, batch_size=BATCH_SIZE, shuffle=True)

    # NOTE: reuse your existing build_model(), evaluate_*, fairness, faithfulness
    # functions from the pasted header. The lines below mirror run_contrastive.py
    # but call train_epoch_iw and save to instance_weighted_* files.

    grid_results = {}
    for lam in LAMBDAS:
        print(f'\n{"="*55}\nInstance-weighted SCM  λ={lam}\n{"="*55}')
        model = build_model().to(DEVICE)            # from pasted header
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        best_val = -1.0
        save_path = os.path.join(OUTPUT_DIR, f'instance_weighted_lam{lam}.pt')
        for epoch in range(EPOCHS):
            tr, ce, sc = train_epoch_iw(model, train_loader_w, optimizer,
                                        lam=lam, use_scm=True)
            val_f1 = eval_macro_f1(model, val_loader)   # from pasted header
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

    # Evaluate best IW model on all three dimensions, reusing pasted functions.
    best_model = load_model(os.path.join(OUTPUT_DIR,
                                         f'instance_weighted_lam{best_lam}.pt'))

    # Performance, fairness, faithfulness — call your existing evaluation
    # functions here exactly as run_contrastive.py does, but write to:
    #   instance_weighted_performance.json
    #   instance_weighted_fairness.json
    #   instance_weighted_faithfulness.json
    # (left as direct calls so you reuse the verified code, not a reimplementation)
    print('Now run your existing performance/fairness/faithfulness evaluators on '
          'best_model and dump to instance_weighted_*.json')

if __name__ == '__main__':
    main()
