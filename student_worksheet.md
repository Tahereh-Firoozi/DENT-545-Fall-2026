# DENT 545 · NER lab worksheet

Name(s): ____________________  Date: ____________________

Open the [notebook in Colab](https://colab.research.google.com/github/Tahereh-Firoozi/DENT-545-Fall-2026/blob/main/notebooks/02_hmm_ner.ipynb) and save your own copy. Work in pairs: one person annotates or edits; the other explains the decision. Switch roles halfway through.

## 1. Why NER matters

Which opening reports contain **currently affirmed epithelial dysplasia**? OED means oral epithelial dysplasia for this exercise.

My selected IDs: ____________________

One false inclusion from keyword search and why: ____________________

One report keyword search misses and why: ____________________

What extra information would the researcher need besides report IDs? ____________________

## 2. Annotate the report

> Incisional biopsy of the left lateral tongue shows no mild epithelial dysplasia.

| Entity phrase | Label | BIO sequence for this phrase |
|---|---|---|
| | | |
| | | |
| | | |

Should “no” be inside the FINDING span? Why? ____________________

Does “negated” mean there is no entity to annotate? ____________________

## 3. Explain the HMM

| Concept | Meaning in this exercise |
|---|---|
| Observation | |
| Hidden state | |
| Transition | |
| Emission | |
| Viterbi decoding | |

Choose a transition from the trace table. Explain it in one sentence: ____________________

Why is the best whole tag sequence not necessarily the same as choosing a tag independently for each word? ____________________

## 4. Evaluate and improve

| System | Test precision | Test recall | Test F1 |
|---|---|---|---|
| Dictionary | | | |
| Original HMM | | | |
| My HMM | | | |

Development error I targeted: ____________________

My new fictional training sentence and expected effect: ____________________

What changed on development data? ____________________

One remaining test error (record ID, phrase, expected label, predicted label): ____________________

Why should I stop changing the model after looking at the test results? ____________________

## 5. Use the output responsibly

Which report IDs does the final pipeline select? ____________________

Where does the assertion rule fail even with correct entity spans? ____________________

For C04, why are all sites in the report not enough to establish finding–site pairs? ____________________

What information would a human need to review before using the table for research? ____________________

## Submit

- Your saved notebook with annotations, explanations, intervention and metrics.
- The exported `ner_results.csv`.
- The five exit-ticket answers, in the notebook or this worksheet.

Use fictional text only. The public instructor guide contains solutions; attempt the activities first.
