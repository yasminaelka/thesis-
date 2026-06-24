"""
check_group_sizes_v2.py
-----------------------
Compares protected-group sizes on the ECtHR test set under several
keyword definitions, so you can pick a sensible one BEFORE launching runs.

Gender is matched WHOLE-WORD (bug fix: 'her' no longer matches 'there').
Ethnicity is shown three ways: original, full-extended (too noisy), and a
new TARGETED extension that adds validated origin terms but excludes the
high-frequency nationality words that inflated the group to ~51%.

Usage:
    conda activate thesis
    python check_group_sizes_v2.py
"""

import re
import numpy as np
from datasets import load_dataset

# ── GENDER ──────────────────────────────────────────────────────────────────
# Curated list from fairness_ci_eod.py (pronouns kept but matched WHOLE-WORD,
# so 'her' will not match 'there'/'where'/'other').
GENDER_KEYWORDS = [
    'woman', 'women', 'female', 'girl', 'mother', 'wife',
    'daughter', 'sister', 'she', 'her', 'hers',
]
# Optional stricter gender variant: drop the standalone pronouns entirely.
GENDER_KEYWORDS_NOPRONOUN = [
    'woman', 'women', 'female', 'girl', 'mother', 'wife',
    'daughter', 'sister',
]

# ── ETHNICITY ───────────────────────────────────────────────────────────────
ETHNICITY_ORIGINAL = [
    'roma', 'romani', 'kurdish', 'kurd', 'chechen', 'asylum',
    'refugee', 'immigrant', 'minority', 'ethnic',
]
# Full Choenni extension — KNOWN TOO NOISY (~51%); shown for comparison only.
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
# TARGETED extension: original + specific origin / minority-population /
# origin-claim terms that, in ECtHR text, refer to a litigant rather than a
# government or court. High-frequency nationality adjectives are DELIBERATELY
# excluded to avoid the noise that pushed the full list to ~51%.
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
    return re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in keywords) + r')\b',
                      flags=re.IGNORECASE)

def group_sizes(test, keywords):
    pat = pattern(keywords)
    prot = sum(
        1 for ex in test
        if pat.search(' '.join(ex['text']) if isinstance(ex['text'], list) else ex['text'])
    )
    return prot, len(test) - prot

print('Loading ECtHR test set ...')
raw = load_dataset('coastalcph/lex_glue', 'ecthr_a')
test = raw['test']
total = len(test)
print(f'Test documents: {total}\n')

def report(name, kws):
    p, u = group_sizes(test, kws)
    print(f'{name:32s} protected={p:4d}  unprotected={u:4d}  ({100*p/total:.1f}%)')

print('=== GENDER (whole-word matched) ===')
report('Gender (curated, whole-word)',  GENDER_KEYWORDS)
report('Gender (no pronouns)',          GENDER_KEYWORDS_NOPRONOUN)

print('\n=== ETHNICITY ===')
report('Ethnicity (original)',          ETHNICITY_ORIGINAL)
report('Ethnicity (full extended)',     ETHNICITY_FULL_EXTENDED)
report('Ethnicity (TARGETED) <-- new',  ETHNICITY_TARGETED)

print('\nAim: a protected group that is a meaningful minority, not ~50%+.')
print('If TARGETED is still too high, we trim the noisier terms further.')
