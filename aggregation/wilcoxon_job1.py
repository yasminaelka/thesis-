```python
"""
Job 1 — Wilcoxon signed-rank tests on per-instance SHAP sufficiency
Yasmina El Kacemi — University of Amsterdam

Token selection: first-k% of real tokens (position-based, deterministic).
Same selection applied to both models → valid paired Wilcoxon test.
No gradient computation, no attention extraction — two forward passes per doc.
"""
import os, json, random, warnings

# Ignore warning messages so the output is easier to read
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from scipy.stats import wilcoxon

# Set the random seed so the same documents are selected every time
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# Use GPU if it is available, otherwise use CPU
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# Print the GPU name when the script runs on CUDA
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Folder where the trained models and output files are stored
OUTPUT_DIR  = '/gpfs/home6/yelkacemi/thesis_outputs'

# Base Legal-BERT model used for the classifier
MODEL_NAME  = 'nlpaueb/legal-bert-base-uncased'

# Number of ECHR article labels
N_LABELS    = 10

# Maximum number of tokens used per document
MAX_LEN     = 512

# Number of test documents used for this sufficiency analysis
SHAP_N_DOCS = 200

# Different percentages of tokens that are kept
K_VALUES    = [0.01, 0.05, 0.10]

# Names of the article labels in the same order as the model outputs
ARTICLE_NAMES     = ['Art.2','Art.3','Art.5','Art.6','Art.8','Art.9',
                     'Art.10','Art.11','Art.14','P1-1']

# Only these articles are used for article-level tests because they are reliable enough
RELIABLE_ARTICLES = ['Art.2','Art.3','Art.5','Art.6','Art.8','P1-1']

# Bonferroni-corrected alpha level for significance
ALPHA_CORRECTED   = 0.005

class LegalBertClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Load the Legal-BERT encoder
        self.encoder    = AutoModel.from_pretrained(MODEL_NAME)

        # Dropout is used before the final classification layer
        self.dropout    = nn.Dropout(0.1)

        # Final layer predicts the article labels
        self.classifier = nn.Linear(self.encoder.config.hidden_size, N_LABELS)

    def forward(self, input_ids, attention_mask):
        # Run the input through Legal-BERT
        out  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use the CLS token as the document representation
        pool = out.last_hidden_state[:, 0, :]

        # Return the logits for all labels
        return self.classifier(self.dropout(pool))

class ECtHRDataset(Dataset):
    def __init__(self, tokenizer):
        # Load the ECHR-A test split from LexGLUE
        ds = load_dataset('coastalcph/lex_glue', 'ecthr_a', split='test')

        # This list will store the tokenized examples
        self.items = []

        # Tokenize every test example and prepare the multi-label target
        for ex in ds:
            # Some texts are stored as a list of paragraphs, so they are joined into one string
            txt = ' '.join(ex['text']) if isinstance(ex['text'], list) else ex['text']

            # Tokenize and pad/truncate each document to MAX_LEN
            enc = tokenizer(txt, max_length=MAX_LEN, truncation=True,
                            padding='max_length', return_tensors='pt')

            # Create a multi-label vector with zeros first
            label = torch.zeros(N_LABELS)

            # Mark the labels that are present for this document
            for l in ex['labels']:
                if l < N_LABELS:
                    label[l] = 1.0

            # Store the tensors needed later by the model
            self.items.append({
                'input_ids':      enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'labels':         label,
            })

    # Return the number of examples in the dataset
    def __len__(self):        return len(self.items)

    # Return one tokenized example
    def __getitem__(self, i): return self.items[i]

def load_model(path):
    # Create the model architecture
    model = LegalBertClassifier().to(DEVICE)

    # Load the saved model weights
    state = torch.load(path, map_location=DEVICE)

    # Some checkpoints store the weights inside model_state_dict
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    # Load the weights into the model
    model.load_state_dict(state, strict=False)

    # Set model to evaluation mode
    model.eval()

    return model

def compute_sufficiency(model, docs, k_values):
    """
    For each doc: keep first k% of real tokens, mask the rest.
    Sufficiency = mean drop in sigmoid prob over active labels.
    Returns {k: np.array(n_docs,)}
    """
    # Token ID used for masking tokens
    mask_id = tokenizer.mask_token_id

    # Store sufficiency scores separately for every k value
    results = {k: [] for k in k_values}

    # No gradients are needed because this is only evaluation
    with torch.no_grad():
        # Go through all sampled documents
        for idx, doc in enumerate(docs):
            # Print progress every 20 documents
            if idx % 20 == 0:
                print(f'  doc {idx}/{len(docs)}')

            # Add batch dimension and move tensors to the selected device
            ids  = doc['input_ids'].unsqueeze(0).to(DEVICE)    # (1,512)
            amsk = doc['attention_mask'].unsqueeze(0).to(DEVICE)

            # Get the original prediction probabilities
            probs_orig = torch.sigmoid(model(ids, amsk)).squeeze(0).cpu().numpy()

            # Select the labels that are active for this document
            active = np.where(doc['labels'].numpy() > 0.5)[0]

            # If a document has no active label, use all labels instead
            if len(active) == 0:
                active = np.arange(N_LABELS)

            # real content tokens: exclude [CLS]=0, [SEP], padding
            sep_id   = tokenizer.sep_token_id
            all_ids  = ids.squeeze(0).cpu().numpy()

            # Get the positions of real content tokens
            real_pos = [
                p for p in amsk.squeeze(0).cpu().numpy().nonzero()[0]
                if p != 0 and all_ids[p] != sep_id
            ]
            real_pos = np.array(real_pos)

            # Number of real content tokens in the document
            seq_len  = len(real_pos)

            # Repeat the masking experiment for every k value
            for k in k_values:
                # Calculate how many tokens should be kept
                top_n      = max(1, int(np.ceil(k * seq_len)))

                # keep first top_n content tokens (positional proxy for importance)
                keep_pos   = set(real_pos[:top_n].tolist())

                # Make a copy of the input IDs so the original document stays unchanged
                ids_masked = ids.clone()

                # Mask all real tokens that are not in the kept positions
                for pos in real_pos:
                    if int(pos) not in keep_pos:
                        ids_masked[0, int(pos)] = mask_id

                # Get predictions after masking most of the document
                probs_masked = torch.sigmoid(
                    model(ids_masked, amsk)).squeeze(0).cpu().numpy()

                # Sufficiency is the average probability drop for the active labels
                suff = float(np.mean(
                    probs_orig[active] - probs_masked[active]))

                # Store the score for this document and this k value
                results[k].append(suff)

    # Convert lists to numpy arrays for easier statistics later
    return {k: np.array(v) for k, v in results.items()}

# ── Load data
print('Loading tokenizer...')

# Load the tokenizer that belongs to Legal-BERT
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print('Loading dataset...')

# Load and tokenize the test dataset
test_ds = ECtHRDataset(tokenizer)

# Create a random generator with the fixed seed
rng     = np.random.default_rng(SEED)

# Randomly sample the documents used in this analysis
indices = rng.choice(len(test_ds), size=SHAP_N_DOCS, replace=False)

# Get the sampled documents
docs    = [test_ds[int(i)] for i in indices]

# Store the labels of the sampled documents for article-level tests
labels  = np.array([d['labels'].numpy() for d in docs])

print(f'Sampled {len(docs)} documents.')

# ── Load models and compute
print('\n=== BASELINE ===')

# Load the baseline model
baseline_model = load_model(
    os.path.join(OUTPUT_DIR, 'baseline_model.pt'))

# Compute sufficiency scores for the baseline model
base_suff = compute_sufficiency(baseline_model, docs, K_VALUES)

# Remove the model from memory before loading the next one
del baseline_model; torch.cuda.empty_cache()

print('\n=== SCM (λ=0.1) ===')

# Load the SCM model
scm_model = load_model(
    os.path.join(OUTPUT_DIR, 'contrastive_lam0.1.pt'))

# Compute sufficiency scores for the SCM model
scm_suff = compute_sufficiency(scm_model, docs, K_VALUES)

# Remove the model from memory after use
del scm_model; torch.cuda.empty_cache()

# ── Save raw arrays

# Store the raw per-document sufficiency scores
raw = {'doc_indices': indices.tolist(), 'doc_labels': labels.tolist(),
       'article_names': ARTICLE_NAMES}

# Add baseline and SCM sufficiency scores for every k value
for k in K_VALUES:
    raw[f'k={k}'] = {'baseline': base_suff[k].tolist(),
                     'scm':      scm_suff[k].tolist()}

# Save the raw arrays so the results can be checked later
with open(os.path.join(OUTPUT_DIR, 'per_instance_sufficiency.json'), 'w') as f:
    json.dump(raw, f, indent=2)

print('\nPer-instance arrays saved.')

# ── Wilcoxon tests
print('\n=== WILCOXON SIGNED-RANK TESTS ===')
print(f'Bonferroni α\' = {ALPHA_CORRECTED}')

# Dictionary for all Wilcoxon test results
wilcoxon_results = {}

# Run tests separately for every k value
for k in K_VALUES:
    # Baseline and SCM arrays for this k
    ba, sa = base_suff[k], scm_suff[k]

    # Create a place to store the results for this k
    wilcoxon_results[f'k={k}'] = {}

    print(f'\n─── k={int(k*100)}% ───')

    # Test both the macro level and each reliable article separately
    for level in ['macro'] + RELIABLE_ARTICLES:
        # Macro uses all sampled documents
        if level == 'macro':
            bv, sv = ba, sa

        # Article level only uses documents where that article is active
        else:
            art_idx = ARTICLE_NAMES.index(level)
            msk     = labels[:, art_idx] > 0.5
            bv, sv  = ba[msk], sa[msk]

        # Number of paired observations for this test
        n = len(bv)

        # Skip the test if there are too few examples
        if n < 10:
            print(f'  {level:<8} n={n:3d}  SKIPPED (n<10)')
            wilcoxon_results[f'k={k}'][level] = {'n': n, 'skipped': True}
            continue

        # Difference between SCM and baseline for each document
        diffs = sv - bv

        # Skip the test if both models have exactly the same scores
        if np.all(diffs == 0):
            print(f'  {level:<8} n={n:3d}  SKIPPED (all diffs zero)')
            wilcoxon_results[f'k={k}'][level] = {'n': n, 'skipped': True}
            continue

        # Run the paired Wilcoxon signed-rank test
        stat, p = wilcoxon(bv, sv, alternative='two-sided')

        # Mean difference shows the direction of the change
        diff    = float(diffs.mean())

        # Check whether the result is significant after correction
        sig     = p < ALPHA_CORRECTED

        # Save all statistics for this level
        wilcoxon_results[f'k={k}'][level] = {
            'n': n, 'baseline_mean': float(bv.mean()),
            'scm_mean': float(sv.mean()), 'mean_diff': diff,
            'statistic': float(stat), 'p_value': float(p),
            'significant': bool(sig)
        }

        # Print one readable result line
        print(f'  {level:<8} n={n:3d}  '
              f'base={bv.mean():.4f}  scm={sv.mean():.4f}  '
              f'Δ={diff:+.4f}  W={stat:.1f}  p={p:.4f}  '
              f'{"✓ sig" if sig else "ns"}')

# Save the Wilcoxon results to JSON
with open(os.path.join(OUTPUT_DIR, 'wilcoxon_results.json'), 'w') as f:
    json.dump(wilcoxon_results, f, indent=2)

print('\nResults saved to wilcoxon_results.json')

# Print a compact thesis-style table for k=1%
print('\n=== THESIS TABLE (k=1%) ===')
print(f'{"Level":<10}{"n":>4}{"Baseline":>10}{"SCM":>10}{"Δ":>8}{"p":>10}{"Sig":>5}')
print('─'*52)

# Show macro and reliable-article results in the thesis table
for level in ['macro'] + RELIABLE_ARTICLES:
    # Get the stored result for this level
    r = wilcoxon_results['k=0.01'].get(level, {})

    # Print skipped rows differently
    if r.get('skipped'):
        print(f'{level:<10}{r.get("n",0):>4}  {"SKIPPED":>35}')

    # Print normal rows with means, difference, p-value, and significance marker
    else:
        print(f'{level:<10}{r["n"]:>4}{r["baseline_mean"]:>10.4f}'
              f'{r["scm_mean"]:>10.4f}{r["mean_diff"]:>+8.4f}'
              f'{r["p_value"]:>10.4f}{"✓":>5 if r["significant"] else "":>5}')
```
