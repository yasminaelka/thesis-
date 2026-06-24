"""
Contrastive SCM Fairness — Full Training & Evaluation Script
Yasmina El Kacemi — University of Amsterdam

Trains:
  1. Baseline LegalBERT (no regularisation)
  2. Contrastive SCM model (best lambda selected on validation)

Evaluates all three dimensions:
  - Predictive performance (macro F1)
  - Fairness (DPD, DI) — separate gender and ethnicity groups
  - Explanation faithfulness (SHAP sufficiency, comprehensiveness)

Architecture matches fix-problem-scm-loss.ipynb exactly.

Multi-seed usage:
  python run_contrastive.py --seed 42            # full pipeline incl. SHAP
  python run_contrastive.py --seed 13 --skip_shap  # training + fairness only

The --seed flag sets all RNGs and writes to a per-seed output folder so runs
never overwrite each other. --skip_shap skips the expensive Phase 5 so
training + fairness can be swept cheaply across many seeds, with SHAP
faithfulness run on only a subset of seeds.
"""

import os, json, random, time, warnings, argparse, re
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

# ── Command-line arguments ──────────────────────────────────────────────────────
# Seed is now a CLI argument so a SLURM job array can sweep several seeds.
# --skip_shap lets cheap runs (training + fairness only) skip the costly
# SHAP faithfulness phase; run SHAP on a small subset of seeds instead.
parser = argparse.ArgumentParser(description='Contrastive SCM fairness training')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for all RNGs (default: 42)')
parser.add_argument('--skip_shap', action='store_true',
                    help='Skip Phase 5 SHAP faithfulness (for cheap multi-seed runs)')
# --- New: broader baselines (committee point 4) ---
# Encoder lets the same pipeline run LegalBERT, vanilla BERT, or RoBERTa, so
# the "is the effect specific to LegalBERT?" question can be answered.
parser.add_argument('--encoder', type=str, default='legal-bert',
                    choices=['legal-bert', 'bert', 'roberta'],
                    help='Backbone encoder (default: legal-bert)')
# --- New: SCM-specificity control (committee point 3) ---
# The regularised arm can use the real SCM antonym pairs, the same words with
# the pairings SHUFFLED (vocabulary held constant, antonym structure broken),
# or RANDOM vocabulary pairs. If 'shuffled'/'random' reproduces the same
# downstream effect as 'scm', the effect is NOT specific to the SCM construct.
parser.add_argument('--pairs', type=str, default='scm',
                    choices=['scm', 'shuffled', 'random'],
                    help='Pair set for the regularised arm (default: scm)')
# --- New: keyword set selection (Sahand / fairness proxy point) ---
parser.add_argument('--keyword_set', type=str, default='targeted',
                    choices=['original', 'extended', 'targeted'],
                    help='Active demographic keyword set for fairness (default: targeted)')
# --- New: fast end-to-end validation before launching on SLURM ---
parser.add_argument('--smoke_test', action='store_true',
                    help='Tiny subset, 1 epoch, single lambda, 4 SHAP docs. '
                         'Use to verify the script runs end-to-end in minutes.')
args = parser.parse_args()

# ── Reproducibility ────────────────────────────────────────────────────────────
# Every RNG is seeded from the CLI value so each run in the array is
# reproducible and the seeds are clearly separated on disk.
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f'Seed: {SEED}')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU   : {torch.cuda.get_device_name(0)}')

# ── Configuration ──────────────────────────────────────────────────────────────
# Encoder choice (committee point 4). Output is keyed by encoder + pair set +
# seed so no two runs in the campaign ever overwrite each other.
ENCODER_MODELS = {
    'legal-bert': 'nlpaueb/legal-bert-base-uncased',
    'bert'      : 'bert-base-uncased',
    'roberta'   : 'roberta-base',
}
MODEL_NAME   = ENCODER_MODELS[args.encoder]
RUN_TAG      = f'{args.encoder}_{args.pairs}_seed{SEED}'
OUTPUT_DIR   = f'/gpfs/home6/yelkacemi/output/{RUN_TAG}'
N_LABELS     = 10
MAX_LEN      = 512
HEAD         = 256
TAIL         = 256
BATCH_SIZE   = 8
LR           = 2e-5
N_EPOCHS     = 3
PATIENCE     = 3
MARGIN       = 0.5
LAMBDAS      = [0.01, 0.05, 0.1, 0.5]
SHAP_N_DOCS  = 200
SHAP_N_BG    = 10
K_VALUES     = [0.01, 0.05, 0.10]
ARTICLE_NAMES = ['Art.2','Art.3','Art.5','Art.6','Art.8','Art.9',
                 'Art.10','Art.11','Art.14','P1-1']

# Reliability is now DERIVED from an explicit rule, not hardcoded.
# RELIABILITY_BASIS documents exactly what the threshold counts, which is the
# clarification the committee asked for. 'test_pos' = ground-truth positive
# labels in the test set (the Table 1 numbers).
RELIABILITY_BASIS = 'test_pos'     # 'test_pos' | 'pred_pos' | 'protected_pos'
MIN_RELIABLE      = 30             # articles with >= this many are 'reliable'

# Smoke test: shrink everything so the full pipeline runs in minutes on CPU.
if args.smoke_test:
    N_EPOCHS    = 1
    LAMBDAS     = [0.1]
    SHAP_N_DOCS = 4
    SHAP_N_BG   = 4
    OUTPUT_DIR  = f'./smoke_output/{RUN_TAG}'
    print('*** SMOKE TEST: tiny subset, 1 epoch, 1 lambda, 4 SHAP docs ***')

print(f'Encoder   : {args.encoder} ({MODEL_NAME})')
print(f'Pair set  : {args.pairs}')
print(f'Run tag   : {RUN_TAG}')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── SCM antonym pairs (from notebook) ─────────────────────────────────────────
# ── SCM pairs: exactly matching unified_scm_notebook_1 ───────────────────────
# 14 original warmth pairs (Omrani et al., ACL 2023 / Fiske et al., 2002)
WARMTH_PAIRS_ORIGINAL = [
    ('sincere',       'dishonest'),
    ('trustworthy',   'untrustworthy'),
    ('kind',          'cruel'),
    ('warm',          'cold'),
    ('friendly',      'hostile'),
    ('generous',      'selfish'),
    ('caring',        'indifferent'),
    ('genuine',       'fake'),
    ('helpful',       'unhelpful'),
    ('tolerant',      'intolerant'),
    ('benevolent',    'malevolent'),
    ('compassionate', 'callous'),
    ('honest',        'deceptive'),
    ('loyal',         'disloyal'),
]
# 5 W&C-Sent enrichment pairs (Ayesh et al., 2026)
WARMTH_PAIRS_WC_SENT = [
    ('trustful',     'suspicious'),
    ('sociable',     'antisocial'),
    ('cooperative',  'uncooperative'),
    ('respectful',   'disrespectful'),
    ('well-meaning', 'ill-intentioned'),
]
# 14 original competence pairs
COMPETENCE_PAIRS_ORIGINAL = [
    ('intelligent',   'stupid'),
    ('capable',       'incapable'),
    ('skilled',       'unskilled'),
    ('competent',     'incompetent'),
    ('efficient',     'inefficient'),
    ('qualified',     'unqualified'),
    ('expert',        'amateur'),
    ('proficient',    'inept'),
    ('resourceful',   'helpless'),
    ('able',          'unable'),
    ('knowledgeable', 'ignorant'),
    ('talented',      'talentless'),
    ('experienced',   'inexperienced'),
    ('reliable',      'unreliable'),
]
# 5 W&C-Sent competence enrichment pairs
COMPETENCE_PAIRS_WC_SENT = [
    ('systematic',  'haphazard'),
    ('methodical',  'chaotic'),
    ('precise',     'imprecise'),
    ('thorough',    'careless'),
    ('coherent',    'incoherent'),
]
ALL_WARMTH_PAIRS     = WARMTH_PAIRS_ORIGINAL     + WARMTH_PAIRS_WC_SENT
ALL_COMPETENCE_PAIRS = COMPETENCE_PAIRS_ORIGINAL + COMPETENCE_PAIRS_WC_SENT
SCM_PAIRS = ALL_WARMTH_PAIRS + ALL_COMPETENCE_PAIRS
print(f'Total SCM pairs: {len(SCM_PAIRS)} ({len(ALL_WARMTH_PAIRS)} warmth + {len(ALL_COMPETENCE_PAIRS)} competence)')

# ── Demographic keywords ───────────────────────────────────────────────────────
# Two keyword sets are defined. The ACTIVE set used for the fairness analysis
# is selected by KEYWORD_SET below.
#
#  - "original": the initial narrow lists used in the first draft.
#  - "extended": the fuller validated lexicon adopted from Choenni et al.
#    (2021, EMNLP) and mirrored in Kocadag (2025, Fig. 3). Broadening the
#    lists addresses the weak-proxy / small-protected-group concern raised
#    in supervisor feedback. This change is disclosed and cited in the
#    methodology; the original lists are retained for a robustness comparison.
#
# NOTE: changing the active set changes which test documents fall into the
# protected groups, and therefore changes all downstream fairness numbers
# (DPD, DI, ΔDPD). Re-run the full fairness phase after switching.

KEYWORD_SET = args.keyword_set   # 'original' | 'extended' | 'targeted'

# --- Original (first-draft) lists -----------------------------------------------
GENDER_KEYWORDS_ORIGINAL = [
    'woman', 'women', 'female', 'girl', 'mother', 'wife',
    'daughter', 'sister', 'she', 'her', 'hers',
]
ETHNICITY_KEYWORDS_ORIGINAL = [
    'roma', 'romani', 'kurdish', 'kurd', 'chechen', 'asylum',
    'refugee', 'immigrant', 'minority', 'ethnic',
]

# --- Extended lists (Choenni et al. 2021; Kocadag 2025, Fig. 3) ------------------
# Gender: now includes male-referential and additional kinship/role terms, so
# the group captures gender-referential text rather than only female terms.
GENDER_KEYWORDS_EXTENDED = [
    # female-referential
    'woman', 'women', 'female', 'girl', 'mother', 'wife', 'daughter',
    'sister', 'she', 'her', 'hers', 'lady', 'bride', 'girlfriend',
    'stepmother', 'grandmother', 'schoolgirl', 'mommy', 'aunt', 'niece',
    # male-referential
    'man', 'men', 'male', 'boy', 'father', 'husband', 'son', 'brother',
    'he', 'him', 'his', 'gentleman', 'groom', 'boyfriend', 'stepfather',
    'grandfather', 'schoolboy', 'daddy', 'uncle', 'nephew',
]
# Ethnicity/nationality: broadened to the validated nationality + ethnicity
# term list rather than only asylum/minority vocabulary.
ETHNICITY_KEYWORDS_EXTENDED = [
    'roma', 'romani', 'kurdish', 'kurd', 'chechen', 'asylum', 'refugee',
    'immigrant', 'minority', 'ethnic', 'european', 'jewish', 'russian',
    'mexican', 'chinese', 'japanese', 'black', 'latina', 'latino', 'white',
    'hispanic', 'american', 'nigerian', 'ethiopian', 'ukrainian', 'sudanese',
    'afghan', 'iraqi', 'italian', 'somali', 'iranian', 'australian',
    'ghanaian', 'swedish', 'finnish', 'moroccan', 'syrian', 'pakistani',
    'british', 'french', 'greek', 'scottish', 'indonesian', 'vietnamese',
    'romanian', 'norwegian', 'nepali', 'korean', 'bengali', 'polish',
    'taiwanese', 'albanian', 'colombian', 'egyptian', 'persian',
    'portuguese', 'turkish', 'austrian', 'african', 'dutch', 'chilean',
    'lebanese',
]

# --- Targeted ethnicity list (Sahand / construct-validity fix) -------------------
# The 'extended' list above mixes ethnicity/minority terms with nationality
# adjectives (european, american, british, french, ...). Respondent states are
# named in almost every ECtHR judgment, so those adjectives match ~50% of docs
# and the "protected" group stops being a minority. The TARGETED list keeps only
# ethnicity / minority-status terms and drops nationality adjectives, yielding a
# genuine minority subgroup (~14% of the test set).
# NOTE: reconcile this against your validated Choenni et al. (2021) / Omrani et
# al. (2023) lists before final submission; it should be your validated list,
# not an ad-hoc one chosen for its group size.
ETHNICITY_KEYWORDS_TARGETED = [
    'roma', 'romani', 'gypsy', 'kurdish', 'kurd', 'chechen',
    'jewish', 'muslim', 'christian', 'orthodox',
    'asylum', 'refugee', 'immigrant', 'migrant', 'foreigner',
    'minority', 'ethnic', 'ethnicity', 'race', 'racial',
    'indigenous', 'aboriginal', 'caste',
]

# Gender keeps pronouns (Sahand's decision): pronouns carry genuine gender
# information in legal text, and we prioritise construct validity over a
# favourable group balance even though this leaves the split near-even.
GENDER_KEYWORDS_TARGETED = GENDER_KEYWORDS_EXTENDED

# Select the active sets used everywhere downstream.
if KEYWORD_SET == 'extended':
    GENDER_KEYWORDS    = GENDER_KEYWORDS_EXTENDED
    ETHNICITY_KEYWORDS = ETHNICITY_KEYWORDS_EXTENDED
elif KEYWORD_SET == 'targeted':
    GENDER_KEYWORDS    = GENDER_KEYWORDS_TARGETED
    ETHNICITY_KEYWORDS = ETHNICITY_KEYWORDS_TARGETED
else:
    GENDER_KEYWORDS    = GENDER_KEYWORDS_ORIGINAL
    ETHNICITY_KEYWORDS = ETHNICITY_KEYWORDS_ORIGINAL

# All sets kept addressable so fairness can be reported across them in one run
# (robustness of the null to the group definition).
KEYWORD_SETS = {
    'original': {'gender': GENDER_KEYWORDS_ORIGINAL, 'ethnicity': ETHNICITY_KEYWORDS_ORIGINAL},
    'extended': {'gender': GENDER_KEYWORDS_EXTENDED, 'ethnicity': ETHNICITY_KEYWORDS_EXTENDED},
    'targeted': {'gender': GENDER_KEYWORDS_TARGETED, 'ethnicity': ETHNICITY_KEYWORDS_TARGETED},
}

def build_keyword_pattern(keywords):
    """One case-insensitive regex matching any keyword as a WHOLE word, so
    'male' does not match inside 'female' and 'her' does not match inside
    'there'. Multi-word terms are escaped safely."""
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', flags=re.IGNORECASE)

def get_group_indices(hf_split, keywords):
    """Split a dataset split into protected / unprotected index arrays by
    whole-word presence of any demographic keyword in the document text."""
    pattern = build_keyword_pattern(keywords)
    protected, unprotected = [], []
    for idx, example in enumerate(hf_split):
        text = ' '.join(example['text']) if isinstance(example['text'], list) else example['text']
        (protected if pattern.search(text) else unprotected).append(idx)
    return np.array(protected), np.array(unprotected)

print(f'Keyword set: {KEYWORD_SET} '
      f'(gender={len(GENDER_KEYWORDS)}, ethnicity={len(ETHNICITY_KEYWORDS)})')

# ── Tokenizer & Dataset ────────────────────────────────────────────────────────
print('Loading tokenizer ...')
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Active pair set + control conditions (committee point 3) ───────────────────
# 'scm'      : real warmth/competence antonym pairs (the treatment).
# 'shuffled' : same words, pairings permuted -> antonym structure destroyed,
#              vocabulary identical. Best control for "is it the SCM construct?".
# 'random'   : random vocabulary pairs, same count. Control for "any auxiliary
#              contrastive term".
def _build_active_pairs(mode):
    if mode == 'scm':
        return list(SCM_PAIRS)
    rng = random.Random(SEED)              # deterministic, independent of globals
    if mode == 'shuffled':
        pos = [p for p, _ in SCM_PAIRS]
        neg = [n for _, n in SCM_PAIRS]
        rng.shuffle(neg)
        return list(zip(pos, neg))
    if mode == 'random':
        # sample clean alphabetic whole-word tokens from the vocabulary
        vocab = [w for w in TOKENIZER.get_vocab()
                 if w.lstrip('Ġ▁##').isalpha() and len(w.lstrip('Ġ▁##')) >= 3]
        words = rng.sample(vocab, 2 * len(SCM_PAIRS))
        clean = [w.lstrip('Ġ▁##') for w in words]
        return list(zip(clean[::2], clean[1::2]))
    raise ValueError(mode)

ACTIVE_PAIRS = _build_active_pairs(args.pairs)

def _word_token_ids(word):
    """Token ids for a word, robust across WordPiece and BPE tokenizers.
    Encodes both the bare word and a leading-space variant, because BPE
    (RoBERTa) represents mid-sentence words with a leading-space marker."""
    ids = set()
    for variant in (word, ' ' + word):
        ids.update(TOKENIZER.encode(variant, add_special_tokens=False))
    return torch.tensor(sorted(ids), dtype=torch.long)

# Precompute once (also avoids re-encoding every pair on every batch).
PAIR_ID_SETS = [(_word_token_ids(p), _word_token_ids(n)) for p, n in ACTIVE_PAIRS]
print(f'Active pair set: {args.pairs} ({len(ACTIVE_PAIRS)} pairs)')

def tokenize_head_tail(text):
    if isinstance(text, list):
        text = ' '.join(text)
    tokens = TOKENIZER(text, truncation=False, add_special_tokens=True,
                       return_tensors='pt')
    ids  = tokens['input_ids'][0]
    mask = tokens['attention_mask'][0]
    if len(ids) > MAX_LEN:
        ids  = torch.cat([ids[:HEAD],  ids[-TAIL:]])
        mask = torch.cat([mask[:HEAD], mask[-TAIL:]])
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
        example = self.data[idx]
        input_ids, attention_mask = tokenize_head_tail(example['text'])
        label_vector = torch.zeros(N_LABELS)
        for l in example['labels']:
            label_vector[l] = 1.0
        return {
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'labels'        : label_vector,
        }

print('Loading dataset ...')
raw          = load_dataset('coastalcph/lex_glue', 'ecthr_a')
if args.smoke_test:
    raw = {
        'train'     : raw['train'].select(range(64)),
        'validation': raw['validation'].select(range(32)),
        'test'      : raw['test'].select(range(64)),
    }
    print('Smoke test: train=64, val=32, test=64')
train_dataset = ECTHRDataset(raw['train'])
val_dataset   = ECTHRDataset(raw['validation'])  # LexGLUE uses 'validation'
test_dataset  = ECTHRDataset(raw['test'])
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}')

# ── Class weights ──────────────────────────────────────────────────────────────
pos_counts = np.zeros(N_LABELS)
for example in raw['train']:
    for label in example['labels']:
        pos_counts[label] += 1
N            = len(raw['train'])
neg_counts   = N - pos_counts
raw_weights  = np.log1p(neg_counts / np.maximum(pos_counts, 1))
log_weights  = raw_weights / raw_weights.mean()
CLASS_WEIGHTS = torch.tensor(log_weights, dtype=torch.float).to(DEVICE)
print('Class weights computed.')

# ── Model ──────────────────────────────────────────────────────────────────────
class BERTClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=output_hidden_states)
        cls    = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        result = {'logits': logits}
        if output_hidden_states:
            result['hidden_states'] = outputs.hidden_states
        return result

# ── Evaluation helpers ─────────────────────────────────────────────────────────
def get_probabilities(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(DEVICE)
            mask  = batch['attention_mask'].to(DEVICE)
            out   = model(ids, mask)
            probs = torch.sigmoid(out['logits']).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch['labels'].numpy())
    return np.vstack(all_probs), np.vstack(all_labels)

def tune_thresholds(probs, labels):
    thresholds = []
    for i in range(N_LABELS):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.05):
            preds = (probs[:, i] >= t).astype(int)
            f1    = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return thresholds

def evaluate_f1(model, loader, thresholds=None):
    probs, labels = get_probabilities(model, loader)
    if thresholds is None:
        thresholds = [0.5] * N_LABELS
    preds = np.zeros_like(probs)
    for i, t in enumerate(thresholds):
        preds[:, i] = (probs[:, i] >= t).astype(int)
    return f1_score(labels, preds, average='macro', zero_division=0)

# ── Contrastive SCM loss ───────────────────────────────────────────────────────
def compute_scm_loss_contrastive(hidden_states, input_ids, pair_id_sets, margin=0.5):
    losses = []
    for pos_tensor, neg_tensor in pair_id_sets:
        pos_tensor = pos_tensor.to(input_ids.device)
        neg_tensor = neg_tensor.to(input_ids.device)
        for b in range(input_ids.shape[0]):
            pos_mask = torch.isin(input_ids[b], pos_tensor)
            neg_mask = torch.isin(input_ids[b], neg_tensor)
            if pos_mask.any() and neg_mask.any():
                h_pos = hidden_states[b][pos_mask].mean(dim=0)
                h_neg = hidden_states[b][neg_mask].mean(dim=0)
                sim   = F.cosine_similarity(h_pos.unsqueeze(0), h_neg.unsqueeze(0))
                loss  = torch.clamp(sim + margin, min=0.0)
                losses.append(loss)
    if len(losses) == 0:
        return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
    return torch.stack(losses).mean()

# ── Training loop ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, lam=0.0, use_scm=False):
    model.train()
    total_loss, total_ce, total_scm = 0.0, 0.0, 0.0
    for batch_idx, batch in enumerate(loader):
        ids    = batch['input_ids'].to(DEVICE)
        mask   = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        out    = model(ids, mask, output_hidden_states=use_scm)
        logits = out['logits']
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=CLASS_WEIGHTS)
        if use_scm:
            hs      = out['hidden_states'][-1]
            scm_loss = compute_scm_loss_contrastive(hs, ids, PAIR_ID_SETS, MARGIN)
            loss     = ce_loss + lam * scm_loss
        else:
            scm_loss = torch.tensor(0.0)
            loss     = ce_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_ce   += ce_loss.item()
        total_scm  += scm_loss.item()
        if (batch_idx + 1) % 200 == 0:
            print(f'  Batch {batch_idx+1}/{len(loader)} | '
                  f'Total: {loss.item():.4f} | '
                  f'CE: {ce_loss.item():.4f} | '
                  f'SCM: {scm_loss.item():.4f}')
    n = len(loader)
    return total_loss/n, total_ce/n, total_scm/n

def train_model(save_path, lam=0.0, use_scm=False, label=''):
    print(f'\n{"="*55}')
    print(f'Training: {label}')
    print(f'{"="*55}')
    torch.manual_seed(SEED)
    model     = BERTClassifier(N_LABELS).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    best_val_f1, patience_cnt = 0.0, 0
    for epoch in range(N_EPOCHS):
        print(f'\nEpoch {epoch+1}/{N_EPOCHS}')
        train_loss, ce_loss, scm_loss = train_epoch(
            model, train_loader, optimizer, lam=lam, use_scm=use_scm)
        val_f1 = evaluate_f1(model, val_loader)
        print(f'  Train: {train_loss:.4f} | CE: {ce_loss:.4f} | '
              f'SCM: {scm_loss:.4f} | Val F1: {val_f1:.4f}')
        if val_f1 > best_val_f1:
            best_val_f1, patience_cnt = val_f1, 0
            torch.save({'model_state_dict': model.state_dict(),
                        'val_f1': val_f1, 'lambda': lam},
                       save_path)
            print(f'  ✓ Saved (val F1={val_f1:.4f})')
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f'  Early stopping at epoch {epoch+1}')
                break
    print(f'\nBest val F1: {best_val_f1:.4f}')
    return best_val_f1

# ── PHASE 1: Train baseline ────────────────────────────────────────────────────
baseline_path  = os.path.join(OUTPUT_DIR, 'contrastive_baseline.pt')
baseline_val_f1 = train_model(baseline_path, lam=0.0, use_scm=False,
                               label='Baseline (no regularisation)')

# ── PHASE 2: Lambda grid search ───────────────────────────────────────────────
print('\n' + '='*55)
print('PHASE 2: Lambda grid search (contrastive SCM)')
print('='*55)
grid_results = {}
for lam in LAMBDAS:
    save_path = os.path.join(OUTPUT_DIR, f'contrastive_lam{lam}.pt')
    val_f1 = train_model(save_path, lam=lam, use_scm=True,
                          label=f'Contrastive SCM λ={lam}')
    grid_results[str(lam)] = {'val_f1': val_f1, 'lambda': lam}

# Select best lambda
best_lam     = max(grid_results, key=lambda k: grid_results[k]['val_f1'])
best_lam_f1  = grid_results[best_lam]['val_f1']
print(f'\nBest λ = {best_lam} (val F1 = {best_lam_f1:.4f})')
print(f'Baseline val F1 = {baseline_val_f1:.4f}')

with open(os.path.join(OUTPUT_DIR, 'contrastive_grid_results.json'), 'w') as f:
    json.dump(grid_results, f, indent=2)
print('Grid results saved.')

# ── PHASE 3: Load best models and evaluate ────────────────────────────────────
print('\n' + '='*55)
print('PHASE 3: Full evaluation')
print('='*55)

def load_model(path):
    m = BERTClassifier(N_LABELS).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    sd   = ckpt.get('model_state_dict', ckpt)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

baseline_model = load_model(baseline_path)
scm_model      = load_model(os.path.join(OUTPUT_DIR, f'contrastive_lam{best_lam}.pt'))

# Tune thresholds on validation set
val_probs_b, val_labels = get_probabilities(baseline_model, val_loader)
val_probs_s, _          = get_probabilities(scm_model,      val_loader)
thresh_b = tune_thresholds(val_probs_b, val_labels)
thresh_s = tune_thresholds(val_probs_s, val_labels)

# Test set predictions
test_probs_b, test_labels = get_probabilities(baseline_model, test_loader)
test_probs_s, _           = get_probabilities(scm_model,      test_loader)

def apply_thresholds(probs, thresholds):
    preds = np.zeros_like(probs)
    for i, t in enumerate(thresholds):
        preds[:, i] = (probs[:, i] >= t).astype(int)
    return preds

test_preds_b = apply_thresholds(test_probs_b, thresh_b)
test_preds_s = apply_thresholds(test_probs_s, thresh_s)

# ── Reliability, derived from an explicit rule (committee point: 30 vs 37) ──────
# The old code hardcoded which articles were "reliable", so the stated rule
# (<30) did not match the set actually used (Art.11 has 37 test positives yet
# was excluded). Here reliability is COMPUTED from one documented basis and all
# three candidate bases are reported, so the thesis can state precisely what the
# threshold counts.
GROUP_INDICES = {
    name: {
        'gender'   : get_group_indices(raw['test'], s['gender']),
        'ethnicity': get_group_indices(raw['test'], s['ethnicity']),
    } for name, s in KEYWORD_SETS.items()
}
_active_eth_prot = GROUP_INDICES[KEYWORD_SET]['ethnicity'][0]

reliability_counts = {}
for i, art in enumerate(ARTICLE_NAMES):
    n_test_pos = int(test_labels[:, i].sum())                       # ground-truth positives
    n_pred_pos = int(test_preds_b[:, i].sum())                      # predicted positives (baseline)
    n_prot_pos = int(test_preds_b[np.asarray(_active_eth_prot), i].sum()) \
                 if len(_active_eth_prot) else 0                    # protected-group predicted positives
    reliability_counts[art] = {
        'test_pos': n_test_pos, 'pred_pos': n_pred_pos, 'protected_pos': n_prot_pos,
    }

RELIABLE = {art for art, c in reliability_counts.items()
            if c[RELIABILITY_BASIS] >= MIN_RELIABLE}
_legacy_reliable = {'Art.2','Art.3','Art.5','Art.6','Art.8','Art.10','P1-1'}

print('\n=== Reliability (basis: %s, threshold: >= %d) ===' % (RELIABILITY_BASIS, MIN_RELIABLE))
print(f'{"Article":8s}  {"test_pos":>8s}  {"pred_pos":>8s}  {"prot_pos":>8s}  {"reliable"}')
for art in ARTICLE_NAMES:
    c = reliability_counts[art]
    print(f'{art:8s}  {c["test_pos"]:>8d}  {c["pred_pos"]:>8d}  '
          f'{c["protected_pos"]:>8d}  {"yes" if art in RELIABLE else "no"}')
if RELIABLE != _legacy_reliable:
    print(f'NOTE: derived reliable set differs from the legacy hardcoded set.')
    print(f'  derived: {sorted(RELIABLE)}')
    print(f'  legacy : {sorted(_legacy_reliable)}')
    print('  Update the thesis text to match the derived set and the stated basis.')

with open(os.path.join(OUTPUT_DIR, 'reliability_report.json'), 'w') as f:
    json.dump({'basis': RELIABILITY_BASIS, 'min_reliable': MIN_RELIABLE,
               'counts': reliability_counts, 'reliable': sorted(RELIABLE)}, f, indent=2)

f1_b = f1_score(test_labels, test_preds_b, average='macro', zero_division=0)
f1_s = f1_score(test_labels, test_preds_s, average='macro', zero_division=0)

print(f'\nBaseline macro F1 : {f1_b:.4f}')
print(f'SCM (λ={best_lam}) macro F1: {f1_s:.4f}')
print(f'ΔF1               : {f1_s - f1_b:+.4f}')

per_article_b = {ARTICLE_NAMES[i]: round(
    f1_score(test_labels[:,i], test_preds_b[:,i], zero_division=0), 4)
    for i in range(N_LABELS)}
per_article_s = {ARTICLE_NAMES[i]: round(
    f1_score(test_labels[:,i], test_preds_s[:,i], zero_division=0), 4)
    for i in range(N_LABELS)}

performance_results = {
    'baseline': {'macro_f1': round(f1_b, 4), 'per_article': per_article_b},
    'scm'     : {'macro_f1': round(f1_s, 4), 'per_article': per_article_s,
                 'lambda': float(best_lam)},
    'delta_f1': round(f1_s - f1_b, 4),
}
with open(os.path.join(OUTPUT_DIR, 'contrastive_performance.json'), 'w') as f:
    json.dump(performance_results, f, indent=2)
print('Performance results saved.')

# ── PHASE 4: Fairness evaluation ───────────────────────────────────────────────
print('\n' + '='*55)
print('PHASE 4: Fairness evaluation')
print('='*55)

def compute_dpd_di(probs, thresholds, protected_idx, unprotected_idx):
    results = {}
    for i, art in enumerate(ARTICLE_NAMES):
        t = thresholds[i]
        prot_preds   = (probs[protected_idx,   i] >= t).astype(float)
        unprot_preds = (probs[unprotected_idx, i] >= t).astype(float)
        prot_rate    = prot_preds.mean()   if len(prot_preds)   > 0 else 0.0
        unprot_rate  = unprot_preds.mean() if len(unprot_preds) > 0 else 0.0
        dpd = abs(prot_rate - unprot_rate)
        di  = prot_rate / unprot_rate if unprot_rate > 0 else float('nan')
        results[art] = {
            'DPD'     : round(float(dpd), 4),
            'DI'      : round(float(di),  4) if not np.isnan(di) else None,
            'reliable': art in RELIABLE,
        }
    return results

# Group sizes for ALL keyword sets, so the robustness of the null to the group
# definition is visible (answers the weak-proxy concern directly).
print('Group sizes by keyword set (protected / total):')
for name in KEYWORD_SETS:
    g = len(GROUP_INDICES[name]['gender'][0])
    e = len(GROUP_INDICES[name]['ethnicity'][0])
    print(f'  {name:9s}  gender={g}/{len(raw["test"])}  ethnicity={e}/{len(raw["test"])}')

# Active set for the headline fairness tables.
gender_prot,    gender_unprot    = GROUP_INDICES[KEYWORD_SET]['gender']
ethnicity_prot, ethnicity_unprot = GROUP_INDICES[KEYWORD_SET]['ethnicity']

fair_b_gender    = compute_dpd_di(test_probs_b, thresh_b, gender_prot,    gender_unprot)
fair_s_gender    = compute_dpd_di(test_probs_s, thresh_s, gender_prot,    gender_unprot)
fair_b_ethnicity = compute_dpd_di(test_probs_b, thresh_b, ethnicity_prot, ethnicity_unprot)
fair_s_ethnicity = compute_dpd_di(test_probs_s, thresh_s, ethnicity_prot, ethnicity_unprot)

print('\n=== Gender Fairness ===')
print(f'{"Article":8s}  {"Base DPD":10s}  {"SCM DPD":10s}  {"ΔDPD":8s}  {"Reliable"}')
print('-' * 55)
for art in ARTICLE_NAMES:
    b = fair_b_gender[art]['DPD']
    s = fair_s_gender[art]['DPD']
    rel = '✓' if fair_b_gender[art]['reliable'] else '✗'
    print(f'{art:8s}  {b:10.4f}  {s:10.4f}  {s-b:8.4f}  {rel}')

print('\n=== Ethnicity Fairness ===')
print(f'{"Article":8s}  {"Base DPD":10s}  {"SCM DPD":10s}  {"ΔDPD":8s}  {"Reliable"}')
print('-' * 55)
for art in ARTICLE_NAMES:
    b = fair_b_ethnicity[art]['DPD']
    s = fair_s_ethnicity[art]['DPD']
    rel = '✓' if fair_b_ethnicity[art]['reliable'] else '✗'
    print(f'{art:8s}  {b:10.4f}  {s:10.4f}  {s-b:8.4f}  {rel}')

fairness_results = {
    'keyword_set': KEYWORD_SET,
    'group_info': {
        'gender_protected'   : int(len(gender_prot)),
        'gender_unprotected' : int(len(gender_unprot)),
        'ethnicity_protected': int(len(ethnicity_prot)),
        'ethnicity_unprotected': int(len(ethnicity_unprot)),
    },
    'baseline': {'gender': fair_b_gender, 'ethnicity': fair_b_ethnicity},
    'scm'     : {'gender': fair_s_gender, 'ethnicity': fair_s_ethnicity},
}
with open(os.path.join(OUTPUT_DIR, 'contrastive_fairness.json'), 'w') as f:
    json.dump(fairness_results, f, indent=2)
print('\nFairness results saved.')

# ── PHASE 4b: lambda trade-off + keyword-set robustness ────────────────────────
# Evaluating EVERY lambda (not just the val-selected one) makes the
# fairness/performance/faithfulness trade-off curve data-driven and removes the
# selection-optimism concern about reporting only the chosen lambda.
def reliable_mean_dpd(fair):
    vals = [fair[a]['DPD'] for a in ARTICLE_NAMES if a in RELIABLE]
    return round(float(np.mean(vals)), 4) if vals else None

print('\n' + '='*55)
print('PHASE 4b: lambda trade-off + keyword-set robustness')
print('='*55)
tradeoff = {}
for lam in LAMBDAS:
    ckpt = os.path.join(OUTPUT_DIR, f'contrastive_lam{lam}.pt')
    if not os.path.exists(ckpt):
        continue
    m = load_model(ckpt)
    vp_l, vl_l = get_probabilities(m, val_loader)
    thr_l      = tune_thresholds(vp_l, vl_l)
    probs_l, _ = get_probabilities(m, test_loader)
    preds_l    = apply_thresholds(probs_l, thr_l)
    f1_l       = round(f1_score(test_labels, preds_l, average='macro', zero_division=0), 4)
    eth_dpd    = reliable_mean_dpd(
        compute_dpd_di(probs_l, thr_l, ethnicity_prot, ethnicity_unprot))
    tradeoff[str(lam)] = {'macro_f1': f1_l, 'ethnicity_mean_DPD_reliable': eth_dpd}
    print(f'  λ={lam:<5}  F1={f1_l:.4f}  ethnicity mean DPD={eth_dpd}')
    del m

# Fairness across all keyword sets for baseline and the selected SCM model:
# shows whether the null survives the choice of group definition.
robustness = {}
for name in KEYWORD_SETS:
    gp, gu = GROUP_INDICES[name]['gender']
    ep, eu = GROUP_INDICES[name]['ethnicity']
    robustness[name] = {
        'gender'   : {'baseline': compute_dpd_di(test_probs_b, thresh_b, gp, gu),
                      'scm'     : compute_dpd_di(test_probs_s, thresh_s, gp, gu)},
        'ethnicity': {'baseline': compute_dpd_di(test_probs_b, thresh_b, ep, eu),
                      'scm'     : compute_dpd_di(test_probs_s, thresh_s, ep, eu)},
    }
with open(os.path.join(OUTPUT_DIR, 'tradeoff_and_robustness.json'), 'w') as f:
    json.dump({'pairs': args.pairs, 'encoder': args.encoder,
               'lambda_tradeoff': tradeoff, 'keyword_robustness': robustness}, f, indent=2)
print('Trade-off and robustness results saved.')

# ── PHASE 5: SHAP faithfulness ─────────────────────────────────────────────────
# Guarded by --skip_shap so cheap multi-seed runs do training + fairness only.
if args.skip_shap:
    print('\n' + '=' * 55)
    print('PHASE 5: SHAP faithfulness — SKIPPED (--skip_shap)')
    print('=' * 55)
    print('Training + fairness complete for this seed. No faithfulness run.')
    faithfulness_results = None
else:
    print('\n' + '=' * 55)
    print('PHASE 5: SHAP faithfulness evaluation')
    print('=' * 55)

    def make_predict_fn(model):
        def predict_fn(token_array):
            all_probs = []
            for i in range(0, len(token_array), 32):
                chunk = torch.tensor(
                    token_array[i:i+32], dtype=torch.long).to(DEVICE)
                attn  = (chunk != TOKENIZER.pad_token_id).long()
                with torch.no_grad():
                    out   = model(chunk, attn)
                    probs = torch.sigmoid(out['logits']).cpu().numpy()
                all_probs.append(probs)
            return np.vstack(all_probs)
        return predict_fn

    def compute_faithfulness(model, label=''):  # PATCHED_FAITHFULNESS_REALTOKENS_V1
        print(f'\nFaithfulness: {label}')
        model.eval()
        predict_fn = make_predict_fn(model)
        mask_id    = TOKENIZER.mask_token_id
        rng        = np.random.default_rng(SEED)
        indices    = rng.choice(len(test_dataset), size=SHAP_N_DOCS, replace=False)

        bg_idx  = rng.choice(len(train_dataset), size=SHAP_N_BG, replace=False)
        background = np.stack([
            train_dataset[int(i)]['input_ids'].numpy() for i in bg_idx])
        print(f'  Background shape: {background.shape}')
        explainer = shap.KernelExplainer(predict_fn, background)
        print('  KernelExplainer ready.')

        results_by_k = {k: {'sufficiency': [], 'comprehensiveness': []}
                        for k in K_VALUES}
        t_start = time.time()

        for doc_num, idx in enumerate(indices):
            if doc_num % 20 == 0 and doc_num > 0:
                elapsed   = (time.time() - t_start) / 60
                remaining = elapsed / doc_num * (SHAP_N_DOCS - doc_num)
                print(f'  [{doc_num}/{SHAP_N_DOCS}] {elapsed:.1f} min elapsed, '
                      f'~{remaining:.1f} min remaining')

            item      = test_dataset[int(idx)]
            input_ids = item['input_ids'].unsqueeze(0).to(DEVICE)
            attn_mask = item['attention_mask'].unsqueeze(0).to(DEVICE)
            ids_np    = input_ids.cpu().numpy()

            shap_vals        = explainer.shap_values(ids_np, nsamples=512, silent=True)
            shap_matrix      = np.stack([sv[0] for sv in shap_vals], axis=-1)
            token_importance = np.abs(shap_matrix).mean(axis=-1)

            # FIX: rank/mask only REAL tokens (attended, excluding [CLS]).
            attn_np  = attn_mask.squeeze(0).cpu().numpy()
            real_pos = np.where(attn_np == 1)[0]
            real_pos = real_pos[real_pos != 0]          # drop [CLS]
            seq_len  = len(real_pos)
            if seq_len < 2:
                continue
            imp_real = token_importance[real_pos]

            with torch.no_grad():
                orig_prob = torch.sigmoid(
                    model(input_ids, attn_mask)['logits']).cpu().numpy()

            for k in K_VALUES:
                top_k_n   = max(1, int(seq_len * k))
                order     = np.argsort(imp_real)
                top_k_pos = set(real_pos[order[-top_k_n:]].tolist())
                non_top_k = [int(p) for p in real_pos if int(p) not in top_k_pos]

                suf_ids = ids_np.copy()
                if non_top_k:
                    suf_ids[0, non_top_k] = mask_id
                suf_t = torch.tensor(suf_ids, dtype=torch.long).to(DEVICE)
                with torch.no_grad():
                    suf_prob = torch.sigmoid(
                        model(suf_t, attn_mask)['logits']).cpu().numpy()
                sufficiency = float(np.abs(orig_prob - suf_prob).mean())

                com_ids = ids_np.copy()
                com_ids[0, list(top_k_pos)] = mask_id
                com_t = torch.tensor(com_ids, dtype=torch.long).to(DEVICE)
                with torch.no_grad():
                    com_prob = torch.sigmoid(
                        model(com_t, attn_mask)['logits']).cpu().numpy()
                comprehensiveness = float(np.abs(orig_prob - com_prob).mean())

                results_by_k[k]['sufficiency'].append(sufficiency)
                results_by_k[k]['comprehensiveness'].append(comprehensiveness)

        aggregated = {}
        per_doc    = {}
        print(f'\n  {"k":6s}  {"Sufficiency":>12s}  {"Comprehensiveness":>18s}')
        print('  ' + '-'*40)
        for k in K_VALUES:
            suf_arr  = results_by_k[k]['sufficiency']
            comp_arr = results_by_k[k]['comprehensiveness']
            suf  = float(np.mean(suf_arr))
            comp = float(np.mean(comp_arr))
            aggregated[f'k={k}'] = {'sufficiency': suf, 'comprehensiveness': comp}
            per_doc[f'k={k}']    = {'sufficiency': suf_arr,
                                    'comprehensiveness': comp_arr}
            print(f'  {k:6.2f}  {suf:>12.4f}  {comp:>18.4f}')

        total_min = (time.time() - t_start) / 60
        print(f'  Total: {total_min:.1f} min')
        return aggregated, per_doc

    faith_baseline, perdoc_baseline = compute_faithfulness(baseline_model, 'Baseline')
    faith_scm,      perdoc_scm      = compute_faithfulness(scm_model, f'SCM (λ={best_lam})')

    faithfulness_results = {
        'baseline': faith_baseline,
        'scm'     : faith_scm,
        'lambda'  : float(best_lam),
    }
    with open(os.path.join(OUTPUT_DIR, 'contrastive_faithfulness.json'), 'w') as f:
        json.dump(faithfulness_results, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'faithfulness_per_doc.json'), 'w') as f:
        json.dump({'baseline': perdoc_baseline, 'scm': perdoc_scm,
                   'lambda': float(best_lam)}, f, indent=2)
    print('\nFaithfulness results saved (aggregated + per-doc).')

# ── Final summary ──────────────────────────────────────────────────────────────
print('\n' + '='*55)
print('ALL DONE — SUMMARY')
print('='*55)
print(f'Seed              : {SEED}')
print(f'Encoder           : {args.encoder}')
print(f'Pair set          : {args.pairs}')
print(f'Keyword set       : {KEYWORD_SET}')
print(f'Baseline macro F1 : {f1_b:.4f}')
print(f'SCM macro F1      : {f1_s:.4f} (λ={best_lam})')
print(f'ΔF1               : {f1_s - f1_b:+.4f}')
print(f'\nFiles saved to {OUTPUT_DIR}')
print('Done!')
