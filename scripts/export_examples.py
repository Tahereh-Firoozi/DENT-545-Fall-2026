"""Generate shareable corpus JSON, example predictions and a reproducible score summary."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))
from corpus import DATA
from ner_lab import HMMNER, DictionaryNER, evaluate, structure, export_csv, cohort_ids

model = HMMNER(DATA["train"])
baseline = DictionaryNER(DATA["train"])
(ROOT / "data/reports.json").write_text(json.dumps(DATA, indent=2) + "\n")
(ROOT / "examples").mkdir(exist_ok=True)
export_csv(structure(DATA["demo"], model.predict), ROOT / "examples/ner_results.csv")
metrics = {split: {name: evaluate(DATA[split], system.predict)[0]
                   for name, system in [("Dictionary", baseline), ("HMM", model)]}
           for split in ("dev", "test")}
metrics["opening_demo"] = dict(keyword_ids=[r["id"] for r in DATA["demo"] if "dysplasia" in r["text"].lower()],
                                gold_ids=cohort_ids(structure(DATA["demo"])),
                                predicted_ids=cohort_ids(structure(DATA["demo"], model.predict)))
(ROOT / "examples/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
