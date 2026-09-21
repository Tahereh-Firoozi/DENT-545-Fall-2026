# Instructor guide · HMM and NER

**DENT 545 · Fall 2026 · Session 2 · September 21, 2026**

This exercise follows the lecture's progression from observed data and hidden states to initial, transition and emission probabilities, Viterbi decoding, and NER. It turns those ideas into a small dental-report task that students can complete in pairs.

## Before class

Open the [Colab notebook](https://colab.research.google.com/github/Tahereh-Firoozi/DENT-545-Fall-2026/blob/main/notebooks/02_hmm_ner.ipynb), save a copy and run the two setup cells. They contain all data and code; no installation, model downloads, API keys or GPU are required. Each student or pair should save their own copy. The student worksheet provides a route for students who cannot use Colab.

The notebook includes all gold labels, and this guide is public. Ask students to attempt each task before revealing answers. Treat this as a formative activity, not a secure assessment. All text is fictional and simplified.

## Learning objectives

By the end, students should be able to:

1. Explain a practical failure of keyword search and the value of structured entity spans.
2. Apply a consistent SITE/PROCEDURE/FINDING annotation scheme and BIO encoding.
3. Identify the observations, states, transitions and emissions in an NER HMM.
4. Interpret exact-span precision, recall and F1, and use development errors to propose a change.
5. Distinguish NER from normalization, assertion detection and relation extraction.

## Suggested facilitation (70 minutes)

| Minutes | Activity | What to say / ask |
|---|---|---|
| 0–10 | Read the opening reports; try keyword search | “Our question is which reports affirm dysplasia now. First decide by reading. Then we will see what a word search gets wrong.” |
| 10–22 | Annotate one report | “Agree on the exact phrase before choosing a label. Would including `no` change what this label means? Keep entity identity and assertion separate.” |
| 22–34 | Train dictionary and HMM; inspect development output | “The dictionary remembers phrases. The HMM also learns tag sequences. Neither has dental understanding. What might each do with an unfamiliar site?” |
| 34–44 | Read the Viterbi path | “The words are observed. At prediction time the tags are hidden. Follow one B tag into its I tag and explain the transition and emission columns.” |
| 44–54 | Add one example; freeze; evaluate | “Choose a change from development errors. Write its expected effect before you run it. Once you inspect the test set, do not tune to it.” |
| 54–64 | Produce the table and CSV | “We now add normalization and assertion rules. Which part of this result came from NER, and which part came from those extra rules?” |
| 64–70 | Challenge cases and exit discussion | “Find a case where perfect NER is still not enough. What evidence would you want beside each extracted row?” |

Allow extra time after the lab for the written exit ticket. For a 40-minute version: opening task (8), annotation (8), train and trace (12), table export (8), C02/C04 discussion (4). Skip model improvement and use the default HMM.

## Answer key: opening task

The correct IDs are **D01, D03 and D07**. This is a count of three reports, not a count of all entity mentions.

| Report | Include? | Reason |
|---|---|---|
| D01 | Yes | Affirmed mild epithelial dysplasia |
| D02 | No | Negated epithelial dysplasia |
| D03 | Yes | OED is the stipulated abbreviation for oral epithelial dysplasia |
| D04 | No | Possible finding, not affirmed |
| D05 | No | A different finding: squamous cell carcinoma |
| D06 | No | Dysplasia is historical; current finding is hyperkeratosis |
| D07 | Yes | Affirmed moderate epithelial dysplasia |
| D08 | No | Dysplasia is negated; chronic inflammation is affirmed |

The keyword `dysplasia` returns D01, D02, D04, D06, D07 and D08: TP 2, FP 4, FN 1. At the **report-retrieval level**, precision is 2/6 = 0.333, recall is 2/3 = 0.667 and F1 is 0.444. These are different from the **entity-level** NER metrics below.

The default HMM plus the explicit normalization and assertion rules returns D01, D03 and D07 on this curated demo. This illustrates the pipeline; it does not validate the pipeline for clinical use. Changing training examples can change its output.

## Answer key: annotation

Report: `Incisional biopsy of the left lateral tongue shows no mild epithelial dysplasia.`

| Token | BIO label |
|---|---|
| Incisional | B-PROCEDURE |
| biopsy | I-PROCEDURE |
| of | O |
| the | O |
| left | B-SITE |
| lateral | I-SITE |
| tongue | I-SITE |
| shows | O |
| no | O |
| mild | B-FINDING |
| epithelial | I-FINDING |
| dysplasia | I-FINDING |
| . | O |

Complete the notebook answer with:

```python
my_entities = [
    ("Incisional biopsy", "PROCEDURE"),
    ("left lateral tongue", "SITE"),
    ("mild epithelial dysplasia", "FINDING"),
]
```

The finding remains annotated even though its assertion is negated. `no` is not part of the finding's name. Including severity is a stated annotation convention; other projects may choose another convention, but consistency matters. `I-FINDING` after `O` is illegal in the BIO scheme used here because no finding span has begun.

The starter answer includes only the procedure, so its initial precision is 1.0, recall 0.333 and F1 0.5. That feedback is intentional. Correcting all three spans gives 1.0 on each measure.

## Answer key: HMM

- **Observations:** The sequence of word and punctuation tokens.
- **Hidden states:** Seven BIO states: O, B/I-SITE, B/I-PROCEDURE, B/I-FINDING. Gold states are available during supervised training, but absent for a new report.
- **Initial probabilities:** Probabilities of valid first tags, estimated from training starts.
- **Transitions:** Probabilities of the next tag given the previous tag. Valid transitions plus END sum to one; invalid BIO transitions have probability zero.
- **Emissions:** Probabilities of observed lowercase words given a tag, with additive smoothing and a shared unknown-word symbol.
- **Viterbi:** Dynamic programming that retains the best prefix for each ending state, then uses backpointers to recover the best complete path. This maximizes the path's joint probability with the observations; it does not sum over all paths.

For a specific observed sequence, maximizing the joint score also maximizes the posterior over tag sequences because the observation probability is constant across candidate paths. Transition/emission columns are not calibrated tag confidence. The ending transition is included in the printed joint log score.

The HMM can label `posterior tongue` in V03 even though `posterior` was unseen, because the sequence pattern supports a SITE phrase. It mislabels the unfamiliar `reactive atypia` in V06 as SITE. This contrast is useful: statistical generalization can succeed and fail under the same simple model.

## Expected scores

These results are reproduced by `scripts/export_examples.py` and recorded in [examples/metrics.json](examples/metrics.json). There is no random training step.

| Split | System | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Development | Dictionary | 17 | 0 | 2 | 1.000 | 0.895 | 0.944 |
| Development | Original HMM | 18 | 1 | 1 | 0.947 | 0.947 | 0.947 |
| Test | Dictionary | 21 | 0 | 3 | 1.000 | 0.875 | 0.933 |
| Test | Original HMM | 23 | 1 | 1 | 0.958 | 0.958 | 0.958 |

Evaluation pools entities across reports and requires exact start/end offsets plus label. A wrong entity type produces both a false positive and a false negative. Correct `O` tokens do not inflate these scores.

Test errors for the dictionary are missed spans: E06 `posterior buccal mucosa`, E08 `retromolar pad`, and E08 `lichenoid mucositis`. The HMM recognizes the E08 finding boundaries but calls `lichenoid mucositis` SITE instead of FINDING. No assertion labels enter these NER scores.

Do not conclude that the HMM is generally superior: this corpus is tiny, templated and constructed for demonstration. High scores are expected under its restricted language.

### Example development-driven intervention

After students diagnose V06, one possible addition is:

```python
EXTRA_MARKED = [
    "{PROCEDURE|Excision} of the {SITE|hard palate} shows {FINDING|affirmed|reactive atypia}."
]
```

This new fictional training sentence is not a copied report. It teaches the observed development vocabulary. Development entity F1 rises to 1.000. The held-out test score remains 0.958: learning one phrase does not solve every unseen phrase. Adding this example after looking at V06 is a development-informed change, so the improved development score is not an unbiased performance estimate.

Accept other well-justified changes and explanations, including changes that do not improve the score. Do not award credit merely for optimizing a number on eight reports.

## Answer key: structured output and failure cases

`ner_results.csv` has one row per predicted finding, including its text, normalized concept, assertion, report ID and offsets. SITE and PROCEDURE fields contain report-level lists. They are not extracted relationships. A record with no predicted finding contributes no row, so a missing entity can silently remove a report from a downstream query.

| Case | What to notice |
|---|---|
| C01 | New anatomical wording and the added word `oral` challenge entity boundaries. The explicit normalization recognizes a dysplasia phrase only if an appropriate finding is extracted. |
| C02 | `epithelial dysplasia is not identified`: gold assertion is negated, but the rule reads only preceding text and returns affirmed. This fails even with gold entity spans. |
| C03 | `not only ... but also ...` affirms both findings. The current rule happens to get both right; a careless rule matching every `not` would fail. |
| C04 | Two specimens: dysplasia belongs to left lateral tongue; hyperkeratosis belongs to right gingiva. Listing both sites beside each finding does not establish those pairings. |
| C05 | `no carcinoma, but epithelial dysplasia is present`: carcinoma is negated and dysplasia affirmed. The rule lets `no` carry across the contrast and incorrectly negates dysplasia. |
| C06 | New procedure, anatomy and finding vocabulary expose the limits of the small training vocabulary. Inspect predictions rather than assuming a particular failure. |

Useful human-review fields include the original report, exact mention, offsets, assertion context and specimen identity. Additional work would require broader and representative annotated data, consistent guidelines, separate validation of assertion and relations, and appropriate review. Ask students to identify these needs from observed errors.

## Suggested formative rubric (10 points)

| Item | Points | Evidence |
|---|---:|---|
| Practical motivation | 2 | Correct report set and a specific explanation of keyword failures |
| Annotation | 2 | Correct entity boundaries/BIO and separation of assertion |
| HMM interpretation | 2 | Correct observations/states and transition/emission explanation |
| Evaluation and intervention | 2 | Correct metrics, development-based rationale and honest error analysis |
| Pipeline limitations | 2 | A concrete assertion failure and the multi-specimen relationship issue |

This is a suggested classroom rubric, not a change to syllabus grading policy.

## Troubleshooting

- **A function or DATA is undefined:** Re-run setup cells 0A and 0B, then the preceding activity cells in order.
- **Cannot edit the original notebook:** Save a copy in Drive first.
- **Annotation entry rejected:** Copy the phrase exactly, preserving capitalization, and use SITE, PROCEDURE or FINDING.
- **No development errors after a change:** Compare with the original HMM and explain the change; do not invent an error.
- **CSV not visible:** Run the export cell, then refresh the Colab Files panel. The file is `ner_results.csv` in the runtime's working directory.
- **Students cannot use Colab:** Use the worksheet and project the instructor's run. A local Python/Jupyter environment can also open the self-contained notebook.

## Verification record

The notebook has been executed cell by cell in a clean temporary directory, using only the Python standard library. Eight automated checks cover span round-trips, corpus duplication, normalized probability distributions, Viterbi versus exhaustive enumeration, valid BIO/unknown input, longest dictionary matches, exact-span scoring, the known assertion failure and full notebook execution. The sample-intervention and rich-display paths are also checked during preparation. These checks verify the teaching implementation, not clinical validity.
