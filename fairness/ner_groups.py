"""
Precompute NER-based ethnicity groups.

This script builds an ethnicity protected-group mask from spaCy NER and saves it
so the fairness scripts can reuse it later.

It also saves the keyword-based mask for comparison.
"""

import os, re, json, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import spacy
from datasets import load_dataset

# -- CLI -----------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/gpfs/home6/yelkacemi/output')
ap.add_argument('--spacy_model', default='en_core_web_sm')
ap.add_argument('--include_gpe', action='store_true',
                help='also build a NORP-or-GPE mask as a secondary variant')
ap.add_argument('--batch_size', type=int, default=32)
args = ap.parse_args()

OUT_PATH = os.path.join(args.root, 'ner_groups.json')

# -- Keyword ethnicity set -----------------------------------------------------

ETHNICITY_KEYWORDS = [
    'roma', 'romani', 'gypsy', 'kurdish', 'kurd', 'chechen',
    'jewish', 'muslim', 'christian', 'orthodox',
    'asylum', 'refugee', 'immigrant', 'migrant', 'foreigner',
    'minority', 'ethnic', 'ethnicity', 'race', 'racial',
    'indigenous', 'aboriginal', 'caste',
]

def build_keyword_pattern(keywords):
    # Build one whole-word regex
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', flags=re.IGNORECASE)

def keyword_mask(texts, keywords):
    # Mark documents that match at least one keyword
    pat = build_keyword_pattern(keywords)
    return np.array([bool(pat.search(t)) for t in texts], dtype=bool)

# -- Load test split -----------------------------------------------------------

# Keep cased text for NER and lowercase text for keyword matching
print('Loading dataset ...')
raw = load_dataset('coastalcph/lex_glue', 'ecthr_a', trust_remote_code=True)

def join_text(ex):
    t = ex['text']
    return ' '.join(t) if isinstance(t, list) else t

test_texts_cased = [join_text(ex) for ex in raw['test']]      # for NER
test_texts_lower = [t.lower() for t in test_texts_cased]      # for keyword match
N = len(test_texts_cased)
print(f'Test documents: {N}')

# -- Keyword distribution ------------------------------------------------------

# Compute keyword-based ethnicity group for comparison
kw_mask = keyword_mask(test_texts_lower, ETHNICITY_KEYWORDS)
print(f'\n[keyword]  ethnicity protected = {int(kw_mask.sum())}  '
      f'unprotected = {int((~kw_mask).sum())}')

# -- NER -----------------------------------------------------------------------

print(f'\nLoading spaCy model: {args.spacy_model} ...')

# Only NER is needed here
nlp = spacy.load(args.spacy_model, disable=['lemmatizer', 'tagger', 'parser',
                                            'attribute_ruler'])

# Increase max length for longer documents
nlp.max_length = 5_000_000

norp_mask = np.zeros(N, dtype=bool)
gpe_mask  = np.zeros(N, dtype=bool)
norp_ents = [[] for _ in range(N)]
gpe_ents  = [[] for _ in range(N)]

print('Running NER over test set (this is the slow part) ...')
for i, doc in enumerate(nlp.pipe(test_texts_cased, batch_size=args.batch_size)):
    norps = sorted({e.text for e in doc.ents if e.label_ == 'NORP'})
    gpes  = sorted({e.text for e in doc.ents if e.label_ == 'GPE'})
    norp_ents[i] = norps
    gpe_ents[i]  = gpes
    norp_mask[i] = len(norps) > 0
    gpe_mask[i]  = len(gpes) > 0
    if (i + 1) % 100 == 0:
        print(f'  {i + 1}/{N}')

norp_gpe_mask = norp_mask | gpe_mask

# -- Distributions -------------------------------------------------------------

# Print group sizes for each method
print('\n=== ethnicity protected-group distribution (test set) ===')
print(f'{"method":22s} {"protected":>10s} {"unprotected":>12s}')
print('-' * 46)
print(f'{"keyword (current)":22s} {int(kw_mask.sum()):>10d} {int((~kw_mask).sum()):>12d}')
print(f'{"NER NORP":22s} {int(norp_mask.sum()):>10d} {int((~norp_mask).sum()):>12d}')
print(f'{"NER NORP or GPE":22s} {int(norp_gpe_mask.sum()):>10d} {int((~norp_gpe_mask).sum()):>12d}')

# Compare keyword and NER masks
agree = int((kw_mask == norp_mask).sum())
both  = int((kw_mask & norp_mask).sum())
print(f'\nkeyword vs NORP: agree on {agree}/{N} docs; '
      f'both-protected on {both}; '
      f'keyword-only={int((kw_mask & ~norp_mask).sum())}; '
      f'NORP-only={int((~kw_mask & norp_mask).sum())}')

# -- Save ----------------------------------------------------------------------

# Save masks and matched entities
out = {
    'meta': {
        'split': 'test',
        'n': N,
        'spacy_model': args.spacy_model,
        'note': 'NORP = nationalities/religious/political groups. '
                'Masks are aligned to load_dataset order, same as '
                'fairness_ci_eod_multiseed.py test_texts.',
    },
    'keyword_ethnicity_mask': kw_mask.astype(int).tolist(),
    'ethnicity_norp_mask': norp_mask.astype(int).tolist(),
    'ethnicity_norp_gpe_mask': norp_gpe_mask.astype(int).tolist(),
    'norp_entities': norp_ents,
    'gpe_entities': gpe_ents,
}
with open(OUT_PATH, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved: {OUT_PATH}')

# Print a few examples for checking
print('\nInspect a few NORP hits:')
shown = 0
for i in range(N):
    if norp_ents[i]:
        print(f'  doc {i}: {norp_ents[i][:6]}')
        shown += 1
    if shown >= 8:
        break
