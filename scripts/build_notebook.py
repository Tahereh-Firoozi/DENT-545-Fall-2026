"""Rebuild the self-contained notebook from auditable source, no dependencies."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, text, **metadata):
    result = dict(cell_type=kind, id=f"dent545-{len(cells):02}", metadata=metadata,
                  source=dedent(text).strip().splitlines(keepends=True))
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    cells.append(result)


def md(text):
    cell("markdown", text)


def code(text, **metadata):
    cell("code", text, **metadata)


md("""
# From dental reports to useful data
## DENT 545 · Fall 2026 · Session 2 · HMM and named entity recognition

**Your mission:** A researcher has eight fictional oral-pathology reports. Identify reports with **currently affirmed epithelial dysplasia**, then list the finding, site and procedure for review.

By the end, you will annotate entity spans, train a small Hidden Markov Model, explain a predicted tag sequence, measure NER errors, and export a table. You will also see which parts of this task **NER alone cannot solve**.

**Time:** 70 minutes, in pairs. Basic Python is helpful; only small edits are needed. All data and tools are embedded. No package installation, GPU, API key or patient records are needed.

**Start:** In Google Colab choose **File → Save a copy in Drive**, then run each cell with ▶ or Shift+Enter. Do not run every cell at once: stop at each **Your turn** before revealing answers. A Google account is needed to run and save in Colab. The downloaded notebook also runs in a local Jupyter installation.

These hand-authored, simplified reports are educational examples, not clinical evidence. Use only fictional text in this exercise.
""")
code('#@title 0A. Load the teaching tools (run once; expand to inspect the HMM)\n' + (ROOT/'lab/ner_lab.py').read_text(), cellView="form")
code('#@title 0B. Load the fictional reports (run once)\n' + (ROOT/'lab/corpus.py').read_text().replace('from ner_lab import parse_report\n', '') + '\nprint("Ready:", {name: len(reports) for name, reports in DATA.items()})', cellView="form")
md("""
## 1 · Why do we need NER? (10 minutes)

Read the eight reports below. In this exercise **OED means oral epithelial dysplasia**. Include current, affirmed dysplasia of any grade. Exclude negated, possible and historical mentions. Count reports, not mentions.

**Your turn:** Before running the keyword search, put your selected report IDs into `my_report_ids`. Explain one inclusion and one exclusion to your partner. This is a report-review task, not a treatment decision.
""")
code('table([dict(report_id=r["id"], report=r["text"]) for r in DATA["demo"]])')
code('''
my_report_ids = []  # Example format: ["D01", "D03"]. Complete your own answer.
my_reason = ""     # Why is finding a word different from finding an affirmed condition?
print("Your selection:", my_report_ids)
''')
code('''
keyword_ids = [r["id"] for r in DATA["demo"] if "dysplasia" in r["text"].lower()]
print("Keyword search returns:", keyword_ids)
print("Number of reports:", len(keyword_ids))
print("Which false alarms or missed reports can you explain?")
''')
md("""
<details><summary>Reveal after discussing: the answer and the reason</summary>

The correct report set is **D01, D03 and D07**. Keyword search returns six reports: D01, D02, D04, D06, D07 and D08. It finds two of the three target reports, includes four false alarms, and misses D03 because it uses OED.

NER finds spans such as **left lateral tongue → SITE**, **Biopsy → PROCEDURE**, and **mild epithelial dysplasia → FINDING**. A useful pipeline also needs **normalization** (OED → epithelial dysplasia) and **assertion detection** (affirmed, negated, uncertain, historical). A negated finding still has an entity span.

For several specimens in one report, linking each finding to its correct site requires additional relation extraction or review.
</details>
""")
md("""
## 2 · Become the annotator (12 minutes)

Use these conventions consistently:

| Label | Include | Exclude |
|---|---|---|
| SITE | Full anatomical phrase, including laterality | `of the`, punctuation |
| PROCEDURE | Full procedure phrase, e.g. `Incisional biopsy` | Prepositions, punctuation |
| FINDING | Full finding, including severity when written | `no`, `possible`, `history of` |

We use **BIO**: `B-` begins an entity, `I-` continues the same entity, and `O` is outside. `O` is a tag, not the digit zero. A one-word entity has only `B-`. Whitespace and sentence punctuation are not included in entity spans.

**Your turn:** Complete `my_entities` using exact phrases copied from the report. One phrase is already provided. Then run the checker. This report contains one specimen; phrases are unique here.
""")
code('''
annotation_text = "Incisional biopsy of the left lateral tongue shows no mild epithelial dysplasia."
my_entities = [
    ("Incisional biopsy", "PROCEDURE"),
    # Add the SITE phrase and FINDING phrase as ("exact text", "LABEL").
]
print(annotation_text)
''')
code('''
annotation_gold = parse_report("A01", "{PROCEDURE|Incisional biopsy} of the {SITE|left lateral tongue} shows no {FINDING|negated|mild epithelial dysplasia}.")
student_spans = []
for phrase, label in my_entities:
    if label not in LABELS or not phrase or annotation_text.count(phrase) != 1:
        print("Check this entry:", (phrase, label), "— use one unique exact phrase and a listed label.")
        continue
    start = annotation_text.index(phrase)
    student_spans.append(dict(start=start, end=start+len(phrase), label=label, text=phrase))
annotation_score, annotation_errors = evaluate([annotation_gold], lambda _: student_spans)
table([annotation_score])
print("Perfect span-and-label agreement?", annotation_score["F1"] == 1.0)
SHOW_ANNOTATION_ANSWER = False  # Change to True only after your attempt.
if SHOW_ANNOTATION_ANSWER:
    highlight(annotation_text, annotation_gold["entities"])
    tokens, gold_tags = bio_tags(annotation_gold)
    table([dict(token=t["text"], BIO_tag=tag) for t, tag in zip(tokens, gold_tags)])
''')
md("""
**Discuss:** Is `no` a FINDING? Should a negated entity be omitted? Why is `I-FINDING` invalid immediately after `O`? What would happen to annotation agreement if one person included `mild` and another did not?

Exact matching requires both the correct boundaries and the correct entity label. A partial span gives one false positive and one false negative.
""")
md("""
## 3 · Teach a dictionary and an HMM (12 minutes)

The **dictionary baseline** memorizes entity phrases in the 24 training reports. It takes the longest matching token sequence. The **HMM** learns how BIO labels follow each other and how likely words are under each label.

| HMM ingredient | Here | Link to the lecture |
|---|---|---|
| Observations | Words and punctuation tokens | Observed moods in the weather example |
| Hidden states | BIO entity tags | Hidden weather states |
| Initial probability | Probability of the first tag | Initial weather distribution |
| Transition probability | Probability of a tag given the previous tag | Weather-to-weather probability |
| Emission probability | Probability of a word given its tag | Mood given weather |
| Decoding | Best complete BIO sequence for this text | Viterbi best weather sequence |

During supervised training, the hidden labels are supplied by annotators. At prediction time we infer them. This model estimates probabilities with counts and additive smoothing, maps unseen words to an unknown-word symbol, and forbids invalid BIO transitions. It uses first-order label context, not a language model's understanding of a report.

**Your turn:** Predict which system will do better on a site phrase it has never seen. Can either system guarantee correct boundaries?
""")
code('''
baseline = DictionaryNER(DATA["train"])
hmm = HMMNER(DATA["train"], alpha=0.1)
dev_comparison = []
for name, model in [("Dictionary", baseline), ("HMM", hmm)]:
    metrics, errors = evaluate(DATA["dev"], model.predict)
    dev_comparison.append(dict(model=name, **metrics))
table(dev_comparison)
''')
md("""
These are **development** results: use them to diagnose errors and choose changes. The training, development and test reports are separate. Entity vocabulary intentionally overlaps, as it does in many tasks, but whole reports are not duplicated. The tiny, repetitive synthetic set is not evidence of real clinical performance.

**Read the numbers:** Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = their harmonic mean. TP requires exact character boundaries **and** entity type. Counts are pooled across reports (micro averaging). We do not use token accuracy because many tokens are simply `O`.
""")
code('''
dev_report = DATA["dev"][2]
print(dev_report["id"], dev_report["text"])
print("Dictionary prediction:")
highlight(dev_report["text"], baseline.predict(dev_report["text"]))
print("HMM prediction:")
highlight(dev_report["text"], hmm.predict(dev_report["text"]))
print("Human reference:")
highlight(dev_report["text"], dev_report["entities"])
''')
md("""
## 4 · Follow the hidden tags (10 minutes)

For observations x and tags y, the model maximizes:

`P(y1) × P(x1|y1) × ∏[P(yt|y(t−1)) × P(xt|yt)] × P(END|yT)`

Viterbi keeps the best partial score for each possible ending tag and saves backpointers to recover the best complete sequence. The code sums log probabilities to avoid tiny floating-point products.

The table below shows the **selected path**, with its transition and emission probabilities. These are components of the joint score; they are **not** confidence percentages for each prediction. The whole dynamic program is in the expandable setup cell and `lab/ner_lab.py`.

**Your turn:** Locate the first `B-SITE`, its `I-SITE` continuation, and any unseen word. Explain why choosing a tag from only the current word can produce a different answer from choosing the best whole path.
""")
code('''
trace_text = dev_report["text"]
trace_rows, joint_log_score = hmm.trace(trace_text)
table(trace_rows)
print("Best joint log score, including END:", round(joint_log_score, 4))
print("A larger joint score compares paths for this same text; it is not a clinical certainty score.")
''')
md("""
## 5 · Make one improvement, then freeze it (10 minutes)

Inspect development errors. Try adding **one newly written fictional training sentence** that addresses an observed error (for example, a site boundary). Use the annotation syntax below. Do not copy a development/test report verbatim into training, and do not inspect test errors until you freeze your choice.

`{PROCEDURE|Biopsy} of the {SITE|your fictional site phrase} shows {FINDING|affirmed|hyperkeratosis}.`

**Your turn:** Write your prediction before running the model. If scores do not improve, explain why; improvement is not guaranteed. Unknown words are deliberately handled very simply here. Additional data, richer features, or a different model could help.
""")
code('''
_, dev_errors = evaluate(DATA["dev"], hmm.predict)
table(dev_errors)
''')
code('''
my_change_prediction = ""  # Which error should your added example address?
EXTRA_MARKED = []          # Add one annotated fictional sentence as a quoted string.
extra_training = [parse_report(f"NEW{i}", text) for i, text in enumerate(EXTRA_MARKED, 1)]
existing_texts = {r["text"] for split in DATA.values() for r in split}
if any(r["text"] in existing_texts for r in extra_training):
    raise ValueError("Write a new sentence; do not copy an existing report into training.")
final_hmm = HMMNER(DATA["train"] + extra_training, alpha=0.1)
final_dev_metrics, _ = evaluate(DATA["dev"], final_hmm.predict)
table([dict(model="Your HMM on development data", **final_dev_metrics)])
''')
md("""
**Freeze your change now.** Run the held-out test once for the final comparison. If you subsequently change the model based on these test results, this set has become development data and you would need a new test set for an unbiased evaluation.
""")
code('''
test_results = []
for name, model in [("Dictionary", baseline), ("Original HMM", hmm), ("Your HMM", final_hmm)]:
    metrics, _ = evaluate(DATA["test"], model.predict)
    test_results.append(dict(model=name, **metrics))
table(test_results)
_, test_errors = evaluate(DATA["test"], final_hmm.predict)
table(test_errors)
''')
md("""
## 6 · Turn spans into a table (10 minutes)

Return to the eight opening reports. We now add two **separate, deliberately limited** steps:

1. Normalize OED and dysplasia phrases to `epithelial dysplasia` using an explicit mapping.
2. Inspect preceding text in the same clause for uncertainty, history and negation cues; otherwise default to affirmed.

These rules were written for teaching, not learned by the HMM. They can fail on trailing negation and complex scope. We retain mention offsets and the original report so a person can check the output.

One row represents **one finding mention**. `sites_in_report` and `procedures_in_report` list co-occurring entities; they do not assert which site/procedure belongs to a finding. That distinction matters when a report has more than one specimen.
""")
code('''
predicted_rows = structure(DATA["demo"], final_hmm.predict)
table(predicted_rows)
predicted_cohort = cohort_ids(predicted_rows)
gold_cohort = cohort_ids(structure(DATA["demo"]))
print("Pipeline-selected reports:", predicted_cohort)
print("Human-reference reports:  ", gold_cohort)
print("False inclusions:", sorted(set(predicted_cohort)-set(gold_cohort)))
print("Missed reports:", sorted(set(gold_cohort)-set(predicted_cohort)))
print("Keyword-only reports:     ", keyword_ids)
output_path = export_csv(predicted_rows)
print("Created", output_path, "— open the Colab Files panel to download it.")
''')
md("""
## 7 · Break the pipeline (6 minutes / extension)

Try these challenge cases **after** final evaluation. Their purpose is error analysis, not estimating population performance. Use the human entity spans first to isolate assertion mistakes; then compare the HMM predictions if you have time.

**Your turn:** Find a report where perfect NER is still insufficient. Explain why the two-specimen report cannot safely produce finding–site pairs from a list of all sites.
""")
code('''
challenge_assertions = []
for report in DATA["challenge"]:
    print(report["id"], report["text"])
    for entity in report["entities"]:
        if entity["label"] == "FINDING":
            rule_result = assertion_rule(report["text"], entity)
            challenge_assertions.append(dict(report=report["id"], finding=entity["text"],
                                              gold=entity["assertion"], rule=rule_result,
                                              correct=rule_result == entity["assertion"]))
table(challenge_assertions)
''')
md("""
## Exit ticket (submit your copied notebook)

1. In two sentences, explain why keyword search returned the wrong report set. Include one report ID.
2. Show your completed annotation and explain one BIO boundary choice.
3. Identify observations, hidden states, transitions and emissions in this HMM.
4. Report exact-span precision, recall and F1 for both baseline systems. Describe your change and one error that remains.
5. Explain one failure that belongs to assertion detection or relation extraction rather than NER. Attach `ner_results.csv` and state what would require human review.

**Before leaving:** Save your notebook copy. Do not upload real patient information.

### Resources

- [Course repository](https://github.com/Tahereh-Firoozi/DENT-545-Fall-2026)
- [Student worksheet](https://github.com/Tahereh-Firoozi/DENT-545-Fall-2026/blob/main/student_worksheet.md)
- Jurafsky & Martin, [Sequence labeling for parts of speech and named entities](https://web.stanford.edu/~jurafsky/slp3/old_aug24/17.pdf), for further reading on BIO and HMMs.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html), for notebook access and saving.

The instructor guide contains solutions and is publicly accessible; attempt the activities first.
""")

notebook = dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                     "language_info": {"name": "python", "version": "3.11"},
                                     "colab": {"name": "02_hmm_ner.ipynb", "toc_visible": True}},
                nbformat=4, nbformat_minor=5)
dest = ROOT / "notebooks/02_hmm_ner.ipynb"
dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Built {dest.name}: {len(cells)} cells")
