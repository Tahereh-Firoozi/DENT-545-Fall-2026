"""Hand-authored fictional oral-pathology snippets; no real patient data.

Markup: {SITE|surface}, {PROCEDURE|surface}, {FINDING|assertion|surface}.
Splits are fixed before fitting. Gold labels are educational conventions.
"""
from ner_lab import parse_report

MARKED = {
"train": [
"{PROCEDURE|Biopsy} of the {SITE|left lateral tongue} shows {FINDING|affirmed|mild epithelial dysplasia}.",
"{PROCEDURE|Biopsy} of the {SITE|right buccal mucosa} shows {FINDING|affirmed|hyperkeratosis}.",
"{PROCEDURE|Incisional biopsy} of the {SITE|floor of mouth} shows {FINDING|affirmed|moderate epithelial dysplasia}.",
"{PROCEDURE|Excision} of the {SITE|lower lip} shows {FINDING|affirmed|squamous cell carcinoma}.",
"{PROCEDURE|Excisional biopsy} of the {SITE|hard palate} shows {FINDING|affirmed|fibroma}.",
"{PROCEDURE|Biopsy} of the {SITE|ventral tongue} shows {FINDING|affirmed|severe epithelial dysplasia}.",
"{PROCEDURE|Biopsy} of the {SITE|left gingiva} shows {FINDING|affirmed|chronic inflammation}.",
"{PROCEDURE|Incisional biopsy} of the {SITE|soft palate} shows {FINDING|affirmed|epithelial dysplasia}.",
"{PROCEDURE|Excision} of the {SITE|right lateral tongue} shows {FINDING|affirmed|OED}.",
"{PROCEDURE|Biopsy} of the {SITE|left buccal mucosa} shows no {FINDING|negated|epithelial dysplasia}.",
"{PROCEDURE|Excision} of the {SITE|upper lip} shows no {FINDING|negated|carcinoma}.",
"{PROCEDURE|Biopsy} of the {SITE|right gingiva} shows possible {FINDING|uncertain|epithelial dysplasia}.",
"{PROCEDURE|Biopsy} of the {SITE|floor of mouth} shows {FINDING|affirmed|hyperkeratosis} without {FINDING|negated|epithelial dysplasia}.",
"{PROCEDURE|Incisional biopsy} of the {SITE|left lateral tongue}: {FINDING|affirmed|moderate epithelial dysplasia}.",
"{PROCEDURE|Biopsy} from the {SITE|hard palate}: {FINDING|affirmed|chronic inflammation}.",
"{PROCEDURE|Excision} from the {SITE|ventral tongue}: {FINDING|affirmed|mild epithelial dysplasia}.",
"{PROCEDURE|Excisional biopsy} from the {SITE|lower lip}: {FINDING|affirmed|fibroma}.",
"{PROCEDURE|Biopsy} from the {SITE|right buccal mucosa}: no {FINDING|negated|carcinoma}.",
"{PROCEDURE|Biopsy} from the {SITE|soft palate}: possible {FINDING|uncertain|squamous cell carcinoma}.",
"{PROCEDURE|Excision} of the {SITE|left gingiva}. History of {FINDING|historical|epithelial dysplasia}. Current finding: {FINDING|affirmed|hyperkeratosis}.",
"{PROCEDURE|Biopsy} of the {SITE|right lateral tongue}. Negative for {FINDING|negated|epithelial dysplasia}.",
"{PROCEDURE|Incisional biopsy} of the {SITE|left buccal mucosa}. Cannot exclude {FINDING|uncertain|mild epithelial dysplasia}.",
"{PROCEDURE|Excision} of the {SITE|upper lip}. Diagnosis: {FINDING|affirmed|OED}.",
"{PROCEDURE|Biopsy} of the {SITE|right gingiva}. Diagnosis: {FINDING|affirmed|fibroma}.",
],
"dev": [
"{PROCEDURE|Excision} of the {SITE|right gingiva} shows {FINDING|affirmed|mild epithelial dysplasia}.",
"{PROCEDURE|Incisional biopsy} from the {SITE|lower lip}: no {FINDING|negated|epithelial dysplasia}.",
"{PROCEDURE|Biopsy} of the {SITE|posterior tongue} shows {FINDING|affirmed|hyperkeratosis}.",
"{PROCEDURE|Excisional biopsy} of the {SITE|soft palate} shows possible {FINDING|uncertain|carcinoma}.",
"{PROCEDURE|Biopsy} of the {SITE|upper lip}. Diagnosis: {FINDING|affirmed|OED}.",
"{PROCEDURE|Biopsy} of the {SITE|left lateral tongue}. History of {FINDING|historical|carcinoma}. Current finding: {FINDING|affirmed|reactive atypia}.",
],
"test": [
"{PROCEDURE|Excisional biopsy} from the {SITE|floor of mouth}: {FINDING|affirmed|severe epithelial dysplasia}.",
"{PROCEDURE|Excision} of the {SITE|hard palate} shows no {FINDING|negated|epithelial dysplasia}.",
"{PROCEDURE|Incisional biopsy} of the {SITE|right gingiva} shows {FINDING|affirmed|OED}.",
"{PROCEDURE|Biopsy} of the {SITE|lower lip} shows possible {FINDING|uncertain|moderate epithelial dysplasia}.",
"{PROCEDURE|Excision} from the {SITE|left buccal mucosa}: {FINDING|affirmed|squamous cell carcinoma}.",
"{PROCEDURE|Biopsy} of the {SITE|posterior buccal mucosa} shows {FINDING|affirmed|hyperkeratosis}.",
"{PROCEDURE|Incisional biopsy} of the {SITE|ventral tongue}. Diagnosis: {FINDING|affirmed|mild epithelial dysplasia}.",
"{PROCEDURE|Biopsy} of the {SITE|retromolar pad} shows {FINDING|affirmed|lichenoid mucositis}.",
],
"demo": [
"{PROCEDURE|Biopsy} from the {SITE|left lateral tongue} shows {FINDING|affirmed|mild epithelial dysplasia}.",
"{PROCEDURE|Biopsy} of the {SITE|right buccal mucosa} shows no {FINDING|negated|epithelial dysplasia}.",
"{PROCEDURE|Excision} of the {SITE|floor of mouth} shows {FINDING|affirmed|OED}.",
"{PROCEDURE|Incisional biopsy} of the {SITE|lower lip} shows possible {FINDING|uncertain|epithelial dysplasia}.",
"{PROCEDURE|Biopsy} of the {SITE|hard palate} shows {FINDING|affirmed|squamous cell carcinoma}.",
"{PROCEDURE|Biopsy} of the {SITE|left gingiva}. History of {FINDING|historical|epithelial dysplasia}. Current finding: {FINDING|affirmed|hyperkeratosis}.",
"{PROCEDURE|Excisional biopsy} of the {SITE|ventral tongue} shows {FINDING|affirmed|moderate epithelial dysplasia}.",
"{PROCEDURE|Biopsy} of the {SITE|soft palate} shows {FINDING|affirmed|chronic inflammation} without {FINDING|negated|epithelial dysplasia}.",
],
"challenge": [
"{PROCEDURE|Biopsy} of the {SITE|lateral border of tongue} shows {FINDING|affirmed|oral epithelial dysplasia}.",
"{PROCEDURE|Biopsy} of the {SITE|lower lip}: {FINDING|negated|epithelial dysplasia} is not identified.",
"{PROCEDURE|Biopsy} of the {SITE|hard palate} shows not only {FINDING|affirmed|epithelial dysplasia} but also {FINDING|affirmed|chronic inflammation}.",
"{PROCEDURE|Biopsy} of the {SITE|left lateral tongue} shows {FINDING|affirmed|mild epithelial dysplasia}; {PROCEDURE|biopsy} of the {SITE|right gingiva} shows {FINDING|affirmed|hyperkeratosis}.",
"{PROCEDURE|Biopsy} of the {SITE|left buccal mucosa}: no {FINDING|negated|carcinoma}, but {FINDING|affirmed|epithelial dysplasia} is present.",
"{PROCEDURE|Punch biopsy} of the {SITE|upper labial mucosa} shows {FINDING|affirmed|verrucous hyperplasia}.",
]}

PREFIX = dict(train="T", dev="V", test="E", demo="D", challenge="C")
DATA = {split: [parse_report(f"{PREFIX[split]}{i:02}", marked) for i, marked in enumerate(reports, 1)]
        for split, reports in MARKED.items()}
