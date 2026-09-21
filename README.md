# DENT 545 · Fall 2026

Practical natural language processing exercises for dentistry.

## Session 2: From dental reports to useful data

**Hidden Markov Models and named entity recognition · September 21, 2026**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tahereh-Firoozi/DENT-545-Fall-2026/blob/main/notebooks/02_hmm_ner.ipynb)

**Start here:** Open the notebook above, choose **File → Save a copy in Drive**, and run its cells in order. Pause at each “Your turn” before revealing an answer. No installation, GPU or API key is needed. Colab requires a Google account to run and save. You can also download the notebook and use a local Jupyter installation.

### The practical problem

A researcher wants to find reports with **currently affirmed epithelial dysplasia**, along with the anatomical site and procedure. Searching for “dysplasia” returns negated and historical mentions and misses the abbreviation “OED.” How can we turn these reports into usable, reviewable data?

In this 70-minute activity, students will:

1. Compare a keyword search with manual review of eight fictional reports.
2. Annotate SITE, PROCEDURE and FINDING spans using BIO tags.
3. Train a dictionary baseline and a small HMM, then inspect its Viterbi tag sequence.
4. Evaluate exact entity boundaries and labels with precision, recall and F1.
5. Add an annotated training example and check its effect on development data.
6. Export a CSV table and investigate failures involving negation, uncertainty and multiple specimens.

NER identifies mentions. Normalization, assertion detection and finding–site relationships are separate tasks. The notebook makes each distinction visible.

### Materials

| Material | Purpose |
|---|---|
| [Student notebook](notebooks/02_hmm_ner.ipynb) | Self-contained activity; all tools and data embedded |
| [Student worksheet](student_worksheet.md) | Questions and submission checklist |
| [Instructor guide](instructor_guide.md) | Timing, spoken prompts, answers and expected results |
| [Annotation guide](data/README.md) | Label boundaries, data splits and limitations |
| [Annotated reports](data/reports.json) | All 52 fictional reports, with character offsets and assertions |
| [Example results](examples/ner_results.csv) | Default pipeline output from the opening eight reports |
| [Teaching code](lab/ner_lab.py) | Readable HMM, dictionary baseline and evaluation |

The instructor guide and all gold annotations are public. Try the exercises before opening the solutions.

### Teaching options

- **70 minutes:** Complete all activities in pairs, followed by the exit ticket.
- **40 minutes:** Complete the keyword comparison, annotation, HMM training/trace and table export. Skip the model-change activity and use one challenge report in the discussion.
- **Without coding:** Review the eight reports, complete the worksheet, and use the instructor's projected notebook for the model demonstration.

### Data and scope

All reports were hand-authored for this exercise. They contain **no real patient records**. There are 24 training, 6 development, 8 test, 8 opening-demo and 6 challenge reports. The examples are short and repetitive by design; scores on them do not estimate real clinical performance. The simple assertion rules have documented failures. Use fictional text only.

### Reproduce or adapt

The notebook needs only Python's standard library; IPython, when available, supplies colored spans and formatted tables. Notebook code is embedded, so no repository download is needed in Colab.

To update the source or verify the exercise locally (Python 3.9+):

```bash
python3 scripts/build_notebook.py
python3 scripts/export_examples.py
python3 -m unittest discover -s tests -v
```

Edit `lab/ner_lab.py` or `lab/corpus.py`, then rebuild the notebook so the embedded copies remain consistent. Tests check annotation offsets, BIO validity, probability distributions, Viterbi against exhaustive decoding, exact-span scoring and full notebook execution. Test runs need no third-party packages or network access.

### Further reading

- Jurafsky & Martin, [Sequence labeling for parts of speech and named entities](https://web.stanford.edu/~jurafsky/slp3/old_aug24/17.pdf).
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html).
