import itertools
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))
from ner_lab import (HMMNER, DictionaryNER, STATES, allowed, bio_tags, spans_from_tags,
                     entity_key, evaluate, structure, cohort_ids, assertion_rule, parse_report)
from corpus import DATA


class LabTests(unittest.TestCase):
    def setUp(self):
        self.model = HMMNER(DATA["train"])

    def test_gold_boundaries_and_round_trip(self):
        texts = []
        for split in DATA.values():
            for report in split:
                texts.append(report["text"])
                tokens, tags = bio_tags(report)
                spans = spans_from_tags(report["text"], tokens, tags)
                self.assertEqual([entity_key(e) for e in spans], [entity_key(e) for e in report["entities"]])
        self.assertEqual(len(texts), len(set(texts)), "Exact duplicate reports across splits")

    def test_probabilities_sum_to_one(self):
        m = self.model
        self.assertAlmostEqual(sum(m.start_p(s) for s in STATES), 1)
        for state in STATES:
            self.assertAlmostEqual(sum(m.transition_p(state, s) for s in STATES + ("<END>",)), 1)
            self.assertAlmostEqual(sum(m.emission_p(state, w) for w in m.vocab), 1)

    def test_viterbi_matches_exhaustive_search(self):
        m = self.model
        text = "Biopsy tongue ."
        tokens, predicted, score = m.decode(text)
        best = -math.inf
        for path in itertools.product(STATES, repeat=len(tokens)):
            if not allowed("<START>", path[0]) or any(not allowed(a, b) for a, b in zip(path, path[1:])):
                continue
            value = math.log(m.start_p(path[0])) + math.log(m.transition_p(path[-1], "<END>"))
            value += sum(math.log(m.emission_p(s, t["text"])) for s, t in zip(path, tokens))
            value += sum(math.log(m.transition_p(a, b)) for a, b in zip(path, path[1:]))
            best = max(best, value)
        self.assertAlmostEqual(score, best)
        self.assertEqual(len(tokens), len(predicted))

    def test_valid_bio_all_splits_and_unknown(self):
        self.assertNotIn("retromolar", self.model.vocab)
        for reports in DATA.values():
            for report in reports:
                self.model.predict(report["text"])
        self.assertEqual(self.model.predict(""), [])
        self.model.predict("Completely unfamiliar tokens!")

    def test_dictionary_longest_match_and_boundaries(self):
        baseline = DictionaryNER(DATA["train"])
        self.assertEqual([e["text"] for e in baseline.predict("Biopsy: mild epithelial dysplasia.")],
                         ["Biopsy", "mild epithelial dysplasia"])
        self.assertEqual(baseline.predict("biopsychology"), [])
        self.assertNotIn(("retromolar", "pad"), baseline.lexicon)

    def test_exact_entity_scoring(self):
        report = parse_report("X", "{FINDING|affirmed|mild epithelial dysplasia}")
        metric, _ = evaluate([report], lambda _: report["entities"])
        self.assertEqual(metric["F1"], 1)
        wrong = [dict(start=5, end=len(report["text"]), label="FINDING", text="epithelial dysplasia")]
        metric, _ = evaluate([report], lambda _: wrong)
        self.assertEqual((metric["TP"], metric["FP"], metric["FN"]), (0, 1, 1))

    def test_practical_task_and_assertion_limits(self):
        gold = cohort_ids(structure(DATA["demo"]))
        self.assertEqual(gold, ["D01", "D03", "D07"])
        keyword = [r["id"] for r in DATA["demo"] if "dysplasia" in r["text"].lower()]
        self.assertEqual(keyword, ["D01", "D02", "D04", "D06", "D07", "D08"])
        c2 = DATA["challenge"][1]
        finding = next(e for e in c2["entities"] if e["label"] == "FINDING")
        self.assertEqual(finding["assertion"], "negated")
        self.assertEqual(assertion_rule(c2["text"], finding), "affirmed")  # Known teaching failure.

    def test_notebook_self_contained(self):
        import contextlib
        import io
        import os
        import tempfile
        notebook = json.loads((ROOT / "notebooks/02_hmm_ner.ipynb").read_text())
        namespace = {"__name__": "__main__"}
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                with contextlib.redirect_stdout(io.StringIO()):
                    for index, cell in enumerate(notebook["cells"]):
                        if cell["cell_type"] == "code":
                            exec(compile("".join(cell["source"]), f"cell-{index}", "exec"), namespace)
                self.assertTrue(Path("ner_results.csv").exists())
                self.assertEqual(namespace["gold_cohort"], ["D01", "D03", "D07"])
                self.assertEqual(namespace["baseline"].lexicon, DictionaryNER(DATA["train"]).lexicon)
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
