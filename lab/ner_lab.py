"""Small, transparent NER teaching tools. Python standard library only.

All examples in this project are fictional. This is not a clinical system.
"""
import csv
import html
import math
import re
from collections import Counter, defaultdict

LABELS = ("SITE", "PROCEDURE", "FINDING")
STATES = ("O",) + tuple(tag + "-" + label for label in LABELS for tag in ("B", "I"))
TOKEN = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text):
    return [{"text": m.group(), "start": m.start(), "end": m.end()} for m in TOKEN.finditer(text)]


def parse_report(report_id, marked):
    """Convert author markup into text and half-open character offsets."""
    chunks, entities, cursor = [], [], 0
    for match in re.finditer(r"\{([^{}]+)\}", marked):
        chunks.append(marked[cursor:match.start()])
        parts = match.group(1).split("|")
        label, surface = parts[0], parts[-1]
        assert label in LABELS and len(parts) in (2, 3)
        start = sum(map(len, chunks))
        entity = dict(start=start, end=start + len(surface), label=label, text=surface)
        if label == "FINDING":
            assert len(parts) == 3 and parts[1] in {"affirmed", "negated", "uncertain", "historical"}
            entity["assertion"] = parts[1]
        chunks.append(surface)
        entities.append(entity)
        cursor = match.end()
    chunks.append(marked[cursor:])
    report = dict(id=report_id, text="".join(chunks), entities=entities)
    bio_tags(report)  # Validate every boundary during authoring.
    return report


def bio_tags(report):
    tokens = tokenize(report["text"])
    tags = ["O"] * len(tokens)
    last_end = -1
    for entity in sorted(report["entities"], key=lambda e: e["start"]):
        assert entity["start"] >= last_end, "Overlapping spans are unsupported."
        assert report["text"][entity["start"]:entity["end"]] == entity["text"]
        indices = [i for i, token in enumerate(tokens)
                   if entity["start"] <= token["start"] and token["end"] <= entity["end"]]
        assert indices and tokens[indices[0]]["start"] == entity["start"]
        assert tokens[indices[-1]]["end"] == entity["end"]
        for j, i in enumerate(indices):
            tags[i] = ("B-" if j == 0 else "I-") + entity["label"]
        last_end = entity["end"]
    return tokens, tags


def allowed(previous, current):
    if current.startswith("I-"):
        return previous in ("B-" + current[2:], "I-" + current[2:])
    return True


def spans_from_tags(text, tokens, tags):
    assert len(tokens) == len(tags)
    entities, active, previous = [], None, "<START>"
    for token, tag in zip(tokens, tags):
        assert tag in STATES and allowed(previous, tag), "Invalid BIO sequence."
        if tag.startswith("B-") or tag == "O":
            if active:
                entities.append(active)
            active = None
        if tag.startswith("B-"):
            active = dict(start=token["start"], end=token["end"], label=tag[2:])
        elif tag.startswith("I-"):
            active["end"] = token["end"]
        previous = tag
    if active:
        entities.append(active)
    return [dict(e, text=text[e["start"]:e["end"]]) for e in entities]


class DictionaryNER:
    """Longest non-overlapping exact token sequence, case insensitive, TRAIN only."""
    def __init__(self, training):
        counts = defaultdict(Counter)
        for report in training:
            for entity in report["entities"]:
                key = tuple(t["text"].lower() for t in tokenize(entity["text"]))
                counts[key][entity["label"]] += 1
        self.lexicon = {key: labels.most_common(1)[0][0] for key, labels in counts.items()}

    def predict(self, text):
        tokens, result, i = tokenize(text), [], 0
        words = [t["text"].lower() for t in tokens]
        while i < len(tokens):
            candidates = [key for key in self.lexicon if tuple(words[i:i+len(key)]) == key]
            if not candidates:
                i += 1
                continue
            key = max(candidates, key=len)
            start, end = tokens[i]["start"], tokens[i+len(key)-1]["end"]
            result.append(dict(start=start, end=end, label=self.lexicon[key], text=text[start:end]))
            i += len(key)
        return result


class HMMNER:
    """Supervised first-order HMM with additive smoothing and log-space Viterbi.

    Word identities are lowercase; words absent from TRAIN map to <UNK>.
    BIO legality constrains and renormalizes the start/transition distributions.
    END is an explicit transition outcome; no token-level confidence is claimed.
    """
    def __init__(self, training, alpha=0.1):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.start_counts, self.transitions, self.emissions = Counter(), defaultdict(Counter), defaultdict(Counter)
        self.vocab = {"<UNK>"}
        self.n_reports = len(training)
        if not training:
            raise ValueError("Provide at least one annotated training report")
        for report in training:
            tokens, tags = bio_tags(report)
            if not tokens:
                raise ValueError("Empty training reports are unsupported")
            self.start_counts[tags[0]] += 1
            for i, (token, tag) in enumerate(zip(tokens, tags)):
                word = token["text"].lower()
                self.vocab.add(word)
                self.emissions[tag][word] += 1
                if i:
                    self.transitions[tags[i-1]][tag] += 1
            self.transitions[tags[-1]]["<END>"] += 1
        self.emission_totals = {s: sum(self.emissions[s].values()) for s in STATES}
        self.transition_totals = {s: sum(self.transitions[s].values()) for s in STATES}

    def start_p(self, state):
        legal = [s for s in STATES if allowed("<START>", s)]
        return ((self.start_counts[state] + self.alpha) /
                (self.n_reports + self.alpha * len(legal))) if state in legal else 0.0

    def transition_p(self, previous, state):
        legal = [s for s in STATES if allowed(previous, s)] + ["<END>"]
        return ((self.transitions[previous][state] + self.alpha) /
                (self.transition_totals[previous] + self.alpha * len(legal))) if state in legal else 0.0

    def emission_p(self, state, word):
        word = word.lower() if word.lower() in self.vocab else "<UNK>"
        return ((self.emissions[state][word] + self.alpha) /
                (self.emission_totals[state] + self.alpha * len(self.vocab)))

    def decode(self, text):
        tokens = tokenize(text)
        if not tokens:
            return tokens, [], 0.0
        scores, back = [], []
        scores.append({s: (math.log(self.start_p(s)) + math.log(self.emission_p(s, tokens[0]["text"])))
                       if self.start_p(s) else -math.inf for s in STATES})
        back.append({})
        for token in tokens[1:]:
            current, pointers = {}, {}
            for state in STATES:
                options = [(scores[-1][prev] + math.log(self.transition_p(prev, state)), prev)
                           for prev in STATES if self.transition_p(prev, state)]
                best, predecessor = max(options, key=lambda pair: pair[0])
                current[state] = best + math.log(self.emission_p(state, token["text"]))
                pointers[state] = predecessor
            scores.append(current)
            back.append(pointers)
        best_score, final_state = max((scores[-1][s] + math.log(self.transition_p(s, "<END>")), s)
                                      for s in STATES)
        path = [final_state]
        for i in range(len(tokens)-1, 0, -1):
            path.append(back[i][path[-1]])
        return tokens, list(reversed(path)), best_score

    def predict(self, text):
        tokens, tags, _ = self.decode(text)
        return spans_from_tags(text, tokens, tags)

    def trace(self, text):
        tokens, tags, score = self.decode(text)
        rows = []
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            previous = tags[i-1] if i else "<START>"
            rows.append(dict(token=token["text"], hidden_tag=tag, previous_tag=previous,
                             transition_or_start=round(self.transition_p(previous, tag) if i else self.start_p(tag), 5),
                             emission=round(self.emission_p(tag, token["text"]), 5),
                             unseen=token["text"].lower() not in self.vocab))
        return rows, score


def entity_key(entity):
    return entity["start"], entity["end"], entity["label"]


def evaluate(reports, predictor):
    tp = fp = fn = 0
    errors = []
    for report in reports:
        gold = {entity_key(e) for e in report["entities"]}
        predicted_entities = predictor(report["text"])
        predicted = {entity_key(e) for e in predicted_entities}
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        for kind, keys in (("extra/wrong span", predicted-gold), ("missed gold span", gold-predicted)):
            for start, end, label in sorted(keys):
                errors.append(dict(report=report["id"], error=kind, text=report["text"][start:end], label=label))
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return dict(TP=tp, FP=fp, FN=fn, precision=round(precision, 3), recall=round(recall, 3), F1=round(f1, 3)), errors


def assertion_rule(text, entity):
    """Intentionally limited: preceding clause cues, then affirmed by default.

    It does NOT solve general negation, trailing cues, scope, or relation extraction.
    """
    before = re.split(r"[.;!?]", text[:entity["start"]].lower())[-1]
    if re.search(r"\b(possible|suspicious for|cannot exclude)\b", before):
        return "uncertain"
    if re.search(r"\b(history of|previous)\b", before):
        return "historical"
    if re.search(r"\b(no|without|negative for)\b", before):
        return "negated"
    return "affirmed"


def normalize_finding(surface):
    """An explicit task-specific mapping, separate from the learned NER model."""
    value = surface.lower()
    return "epithelial dysplasia" if "dysplasia" in value or value == "oed" else value


def structure(reports, predictor=None):
    """One row per finding mention. Sites/procedures are report-level lists.

    predictor=None uses gold spans AND gold assertions to establish the answer.
    With a predictor, assertions come from assertion_rule, never from gold labels.
    """
    rows = []
    for report in reports:
        entities = report["entities"] if predictor is None else predictor(report["text"])
        sites = "; ".join(e["text"] for e in entities if e["label"] == "SITE")
        procedures = "; ".join(e["text"] for e in entities if e["label"] == "PROCEDURE")
        for entity in entities:
            if entity["label"] == "FINDING":
                rows.append(dict(report_id=report["id"], finding=entity["text"],
                                 concept=normalize_finding(entity["text"]),
                                 assertion=entity["assertion"] if predictor is None else assertion_rule(report["text"], entity),
                                 sites_in_report=sites, procedures_in_report=procedures,
                                 start=entity["start"], end=entity["end"]))
    return rows


def cohort_ids(rows):
    return sorted({r["report_id"] for r in rows if r["concept"] == "epithelial dysplasia" and r["assertion"] == "affirmed"})


def table(rows):
    if not rows:
        print("No rows.")
        return
    try:
        from IPython.display import HTML, display
        columns = list(rows[0])
        markup = '<table style="border-collapse:collapse"><tr>' + ''.join('<th style="padding:6px;text-align:left">'+html.escape(str(c))+'</th>' for c in columns) + '</tr>'
        for row in rows:
            markup += '<tr>'+''.join('<td style="padding:6px;border-top:1px solid #ddd">'+html.escape(str(row[c]))+'</td>' for c in columns)+'</tr>'
        display(HTML(markup+'</table>'))
    except ImportError:
        for row in rows:
            print(row)


def highlight(text, entities):
    colors = dict(SITE="#d8edff", PROCEDURE="#eadfff", FINDING="#ffe2aa")
    chunks, cursor = [], 0
    for entity in sorted(entities, key=lambda e: e["start"]):
        chunks.append(html.escape(text[cursor:entity["start"]]))
        chunks.append('<mark style="background:'+colors[entity["label"]]+';padding:3px;border-radius:3px">'+html.escape(entity["text"])+' <small>['+entity["label"]+']</small></mark>')
        cursor = entity["end"]
    chunks.append(html.escape(text[cursor:]))
    markup = '<p style="line-height:2.4">'+''.join(chunks)+'</p>'
    try:
        from IPython.display import HTML, display
        display(HTML(markup))
    except ImportError:
        print(text)
        table(entities)


def export_csv(rows, path="ner_results.csv"):
    if not rows:
        raise ValueError("There are no finding rows to export")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
