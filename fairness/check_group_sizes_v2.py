"""
Check protected-group sizes on the ECtHR test set.

The script compares different gender and ethnicity keyword lists using
whole-word matching.
"""

import re
import numpy as np
from datasets import load_dataset

# ── GENDER ──────────────────────────────────────────────────────────────────

# Gender keywords
GENDER_KEYWORDS = [
    'woman', 'women', 'female', 'girl', 'mother', 'wife',
    'daughter', 'sister', 'she', 'her', 'hers',
]

# Stricter gender list without pronouns
GENDER_KEYWORDS_NOPRONOUN = [
    'woman', 'women', 'female', 'girl', 'mother', 'wife',
    'daughter', 'sister',
]

# ── ETHNICITY ───────────────────────────────────────────────────────────────

# Original ethnicity keywords
ETHNICITY_ORIGINAL = [
    'roma', 'romani', 'kurdish', 'kurd', 'chechen', 'asylum',
    'refugee', 'immigrant', 'minority', 'ethnic',
]

# Larger ethnicity list, mainly kept for comparison
ETHNICITY_FULL_EXTENDED = ETHNICITY_ORIGINAL + [
    'european', 'jewish', 'russian', 'mexican', 'chinese', 'japanese',
    'black', 'latina', 'latino', 'white', 'hispanic', 'american', 'nigerian',
    'ethiopian', 'ukrainian', 'sudanese', 'afghan', 'iraqi', 'italian',
    'somali', 'iranian', 'australian', 'ghanaian', 'swedish', 'finnish',
    'moroccan', 'syrian', 'pakistani', 'british', 'french', 'greek',
    'scottish', 'indonesian', 'vietnamese', 'romanian', 'norwegian',
    'nepali', 'korean', 'bengali', 'polish', 'taiwanese', 'albanian',
    'colombian', 'egyptian', 'persian', 'portuguese', 'turkish', 'austrian',
    'african', 'dutch', 'chilean', 'lebanese',
]

# Targeted ethnicity list
ETHNICITY_TARGETED = ETHNICITY_ORIGINAL + [
    # regional ethnic minorities common in ECtHR case law
    'tatar', 'ingush', 'dagestani', 'bosniak', 'uzbek', 'tamil', 'kurds',
    'romany', 'gypsy', 'traveller',
    # origin / status terms that co-occur with origin-based claims
    'stateless', 'deportation', 'expulsion', 'naturalisation',
    'naturalization', 'nationality', 'foreigner', 'alien', 'migrant',
    'racial', 'race', 'ethnicity',
]

def pattern(keywords):
    # Build one whole-word regex pattern
    return re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in keywords) + r')\b',
                      flags=re.IGNORECASE)

def group_sizes(test, keywords):
    # Count protected documents for one keyword list
    pat = pattern(keywords)
    prot = sum(
        1 for ex in test
        if pat.search(' '.join(ex['text']) if isinstance(ex['text'], list) else ex['text'])
    )

    # Return protected and unprotected counts
    return prot, len(test) - prot

# Load test split
print('Loading ECtHR test set ...')
raw = load_dataset('coastalcph/lex_glue', 'ecthr_a')
test = raw['test']

# Total test documents
total = len(test)
print(f'Test documents: {total}\n')

def report(name, kws):
    # Print group size for one keyword setting
    p, u = group_sizes(test, kws)
    print(f'{name:32s} protected={p:4d}  unprotected={u:4d}  ({100*p/total:.1f}%)')

# Compare gender keyword settings
print('=== GENDER (whole-word matched) ===')
report('Gender (curated, whole-word)',  GENDER_KEYWORDS)
report('Gender (no pronouns)',          GENDER_KEYWORDS_NOPRONOUN)

# Compare ethnicity keyword settings
print('\n=== ETHNICITY ===')
report('Ethnicity (original)',          ETHNICITY_ORIGINAL)
report('Ethnicity (full extended)',     ETHNICITY_FULL_EXTENDED)
report('Ethnicity (TARGETED) <-- new',  ETHNICITY_TARGETED)

# Final note
print('\nAim: a protected group that is a meaningful minority, not ~50%+.')
print('If TARGETED is still too high, we trim the noisier terms further.')
