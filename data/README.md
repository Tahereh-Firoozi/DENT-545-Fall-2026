# Annotation and data guide

This corpus contains 52 fictional snippets written specifically for DENT 545. No source clinical records, patient identifiers or private datasets were used. The style is intentionally simplified. Gold labels express the exercise's annotation conventions, not an external clinical adjudication.

## Splits

| Prefix | Split | Reports | Purpose |
|---|---|---:|---|
| T | Training | 24 | Fit the dictionary and HMM |
| V | Development | 6 | Inspect mistakes and choose an intervention |
| E | Test | 8 | Final evaluation after freezing the intervention |
| D | Opening demo | 8 | Keyword-search and structured-data task |
| C | Challenge | 6 | Inspect assertion scope, unfamiliar terms and multiple specimens |

No whole report appears in more than one split. Vocabulary and sentence templates deliberately overlap. This is not a randomized sample, and small score differences have no clinical significance. Once a test report influences a model change, it is development material and a new test set is needed.

## Span conventions

- **SITE:** The complete anatomical phrase. Include laterality and location modifiers: `left lateral tongue`, `floor of mouth`. Exclude `of the` and punctuation.
- **PROCEDURE:** The full procedure phrase, including a modifier: `Incisional biopsy`, `Excision`.
- **FINDING:** The complete finding phrase. Include severity: `mild epithelial dysplasia`. Exclude negation, uncertainty and historical cues: `no`, `possible`, `history of`.
- Annotate a finding even when it is negated, uncertain or historical. Store assertion separately.
- Use one non-overlapping span per mention. Nested and discontinuous entities are outside this exercise.
- BIO uses `B-LABEL` for the first token, `I-LABEL` for later tokens in the same span, and `O` for other tokens. A one-token entity has no I token.

The tokenizer groups letters/numbers and internal hyphens/apostrophes; punctuation forms separate tokens. Character offsets use Python string indexing, start included and end excluded. For every entity, `text[start:end] == entity['text']`.

## JSON schema

`reports.json` is an object keyed by split. Each report has `id`, `text` and an `entities` list. Every entity has `start`, `end`, `text` and `label`. FINDING also has `assertion`: `affirmed`, `negated`, `uncertain` or `historical`.

Gold assertion values are used for reference answers and isolated assertion evaluation. HMM and dictionary fitting use only BIO labels derived from training entity spans. Predicted table assertions are produced by a separate rule function and never read gold assertion labels.

## Known limitations to teach

The HMM lowercases words, uses additive smoothing (alpha 0.1), and maps all unseen words to one unknown symbol. It learns initial, transition and emission probabilities from annotated training counts. An explicit END transition models sequence termination; illegal BIO starts/transitions have zero probability, with the remaining distributions normalized. Viterbi returns the most probable complete tag path under this model. Its joint score is not a calibrated confidence score.

The assertion rule inspects preceding text since the last `.`, `;`, `!` or `?`. It checks uncertainty first, then history, then negation, and otherwise assumes affirmed. It misses trailing negation (C02) and lets a negation cue carry too far across a contrast (C05).

The table lists all recognized sites and procedures in a report. It does not link a finding to a particular specimen. C04 has two specimens and exposes that limitation. Normalization is a small explicit map for the teaching query, not a general terminology service.
