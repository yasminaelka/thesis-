"""
NER-based protected-group precompute for the ethnicity axis.
Yasmina El Kacemi - University of Amsterdam

Independent second operationalisation of the ETHNICITY protected group,
as a robustness check on the keyword-based grouping used in
fairness_ci_eod_multiseed.py.

Method: a document is ETHNICITY-protected if spaCy NER (en_core_web_sm)
tags at least one NORP entity in its text. NORP = "nationalities,
religious or political groups", which maps directly onto the ethnicity
axis and is methodologically independent of keyword matching.

This grouping depends only on the document text, NOT on the model or the
seed, so it is computed ONCE over the test set and saved to JSON. The
per-seed fairness runs then load this mask instead of recomputing NER.

Run on a login node (CPU is fine):
  python ner_groups.py
Optional:
  python ner_groups.py --include_gpe   # also count GPE (countries/cities)

Output: ner_groups.json in --root, containing for the test split:
  - ethnicity_norp        : bool mask (NORP only)         [primary]
  - ethnicity_norp_gpe    : bool mask (NORP or GPE)       [secondary]
  - the matched entity strings per doc (for manual inspection)
The keyword ethnicity mask is also recomputed here so the two
distributions can be printed side by side.
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

# -- Keyword ethnicity set: EXACT copy from fairness_ci_eod_multiseed.py -------
ETHNICITY_KEYWORDS = [
    'roma', 'romani', 'gypsy', 'kurdish', 'kurd', 'chechen',
    'jewish', 'muslim', 'christian', 'orthodox',
    'asylum', 'refugee', 'immigrant', 'migrant', 'foreigner',
    'minority', 'ethnic', 'ethnicity', 'race', 'racial',
    'indigenous', 'aboriginal', 'caste',
]

def build_keyword_pattern(keywords):
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile(r'\b(?:' + '|'.join(escaped) + r')\b', flags=re.IGNORECASE)

def keyword_mask(texts, keywords):
    pat = build_keyword_pattern(keywords)
    return np.array([bool(pat.search(t)) for t in texts], dtype=bool)

# -- Load test split EXACTLY as the fairness script does -----------------------
# fairness_ci_eod_multiseed.py lowercases the joined text; we keep the original
# case for NER (NER needs case) but build texts in the SAME ORDER, and lowercase
# only for the keyword comparison.
print('Loading dataset ...')
raw = load_dataset('coastalcph/lex_glue', 'ecthr_a', trust_remote_code=True)

def join_text(ex):
    t = ex['text']
    return ' '.join(t) if isinstance(t, list) else t

test_texts_cased = [join_text(ex) for ex in raw['test']]      # for NER
test_texts_lower = [t.lower() for t in test_texts_cased]      # for keyword match
N = len(test_texts_cased)
print(f'Test documents: {N}')

# -- Keyword distribution (for side-by-side comparison) ------------------------
kw_mask = keyword_mask(test_texts_lower, ETHNICITY_KEYWORDS)
print(f'\n[keyword]  ethnicity protected = {int(kw_mask.sum())}  '
      f'unprotected = {int((~kw_mask).sum())}')

# -- NER -----------------------------------------------------------------------
print(f'\nLoading spaCy model: {args.spacy_model} ...')
# We only need the NER component; disabling others speeds it up a lot.
nlp = spacy.load(args.spacy_model, disable=['lemmatizer', 'tagger', 'parser',
                                            'attribute_ruler'])
# spaCy default max_length is 1_000_000 chars; ECtHR docs can be long but the
# join is well under that. Bump anyway to be safe.
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
print('\n=== ethnicity protected-group distribution (test set) ===')
print(f'{"method":22s} {"protected":>10s} {"unprotected":>12s}')
print('-' * 46)
print(f'{"keyword (current)":22s} {int(kw_mask.sum()):>10d} {int((~kw_mask).sum()):>12d}')
print(f'{"NER NORP":22s} {int(norp_mask.sum()):>10d} {int((~norp_mask).sum()):>12d}')
print(f'{"NER NORP or GPE":22s} {int(norp_gpe_mask.sum()):>10d} {int((~norp_gpe_mask).sum()):>12d}')

# overlap between keyword and NORP, to show they are genuinely different proxies
agree = int((kw_mask == norp_mask).sum())
both  = int((kw_mask & norp_mask).sum())
print(f'\nkeyword vs NORP: agree on {agree}/{N} docs; '
      f'both-protected on {both}; '
      f'keyword-only={int((kw_mask & ~norp_mask).sum())}; '
      f'NORP-only={int((~kw_mask & norp_mask).sum())}')

# -- Save ----------------------------------------------------------------------
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
print('\nInspect a few NORP hits:')
shown = 0
for i in range(N):
    if norp_ents[i]:
        print(f'  doc {i}: {norp_ents[i][:6]}')
        shown += 1
    if shown >= 8:
        break
