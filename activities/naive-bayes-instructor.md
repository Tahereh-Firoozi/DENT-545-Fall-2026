# Naive Bayes pipeline: instructor notes

Audience: first-time NLP learners. Allow 25–35 minutes, ideally in pairs. No submission or student information is collected. Students run the first Colab cell, then interact with the embedded exercise. If their browser blocks the embedded frame, they can use the download cell and open the HTML locally. The standalone HTML has no external dependencies.

## Learning goals

- Explain the difference between cleaning, annotation, training, testing and applying a model.
- Explain Naive Bayes as learning label frequencies and word counts from labeled training examples.
- Read a confusion matrix and explain precision, recall and F1 in everyday words.
- Recognize that missing status is not non-smoking, and a confident prediction can be wrong.

## Suggested facilitation

| Steps | Minutes | Prompt |
|---|---:|---|
| Collect and clean | 4 | What should come out of the model? What happens if we delete “never”? |
| Label | 5 | Which note tells us nothing about smoking status? |
| Split | 4 | Why keep the exam questions separate from practice? |
| Train | 5 | What did the computer actually learn? Which group saw “never”? |
| Test | 5 | What do the diagonal numbers tell us? |
| Apply and recap | 7 | Can you make the model fail? Who checks the prediction? |

Students must complete the opening checks, labeling and split before moving forward. They can revisit unlocked steps. Restart clears the activity. Responses remain in memory only while the page is open.

## Expected answers and behavior

- Labels: A/B/G = Smoker; C/D/H = Non-smoker; E/F/I = Unknown.
- Fixed training split: A–F; test split: G–I. Each label appears twice in training and once in test. This is a teaching split; the figure describes a random split.
- Cleaning lowercases text and extracts alphabetic words. It retains “never”; no stemming or stopword removal is applied.
- The trained vocabulary has 10 words. Each class has six training word occurrences and two notes. All priors are 1/3.
- Multinomial Naive Bayes uses Laplace smoothing with alpha = 1, log scores and normalized scores. Words outside the training vocabulary are ignored and explicitly listed. Entirely unfamiliar notes show a lack-of-evidence warning. Ties are displayed as ties in the application view.
- All three held-out notes are correctly classified: precision, recall and F1 are 1.00 for each class. Each class has only one test example, so these results provide no evidence of clinical generalizability.
- “Never smokes cigarettes.” is predicted as Smoker despite its plain-language negation. “Smokes” and “cigarettes” outweigh “never.” Ask students to compare it with “Never smoked.” Bag-of-words counts do not reliably represent negation or word order.
- “Quit smoking years ago.” is predicted as Unknown: only “smoking” is in the vocabulary. Discuss former smoking, absent training examples and missing context. It should not be treated as a verified clinical prediction.
- Hypothetical precision question: 4/5 = 80%.
- Smoker subgroup question: unknown intensity.
- Exit checks: human annotation; training; human review of the suspicious prediction.

## Relationship to the supplied figure

The exercise uses only the supplied image as its paper-specific source. The paper title, authors, full methods, actual cleaning choices and Naive Bayes implementation were not provided. Do not present classroom cleaning rules or model results as the paper’s methods/results.

The figure reports one visit per patient (Jan 1, 2009–Dec 31, 2011), 6,410 cleaned records, 3,296 manually annotated records, 2,176 training and 1,120 test records, and application to 3,114 remaining records. It includes three broad labels followed by six smoking subgroups: unknown intensity, light, intermediate, intermittent, past and heavy. It reports precision, recall and F1 evaluation and manual checking of 315 predictions. It identifies SVM as the final model. Naive Bayes here teaches the process; the second-stage classifier is discussed rather than implemented.

## Closing discussion

Ask a pair to explain the pipeline without using technical words. Ask another pair where labels enter and where they are hidden from the model. Then ask why labeling new notes by machine does not create new verified ground truth. For a more advanced discussion, distinguish a validation set for model selection from an untouched final test set. The provided figure compares classifiers on its test set; avoid portraying that as independent final validation after selection.
