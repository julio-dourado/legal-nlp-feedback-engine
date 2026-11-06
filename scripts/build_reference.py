"""Build the offline reference profile used for drift detection + demo examples.

Runs the published model on the held-out test split and stores:
  * reference.json : label mix, confidence histogram, test accuracy / macro-F1
  * examples.json  : a few public test-set decisions to try in the UI

Usage: python scripts/build_reference.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.metrics import confidence_hist  # noqa: E402
from core.model import HFClassifier  # noqa: E402

clf = HFClassifier()
test = load_dataset("joelniklaus/brazilian_court_decisions", split="test")
test = test.filter(lambda x: x["judgment_label"] in clf.labels and x["decision_description"])

preds = [clf.predict(t) for t in test["decision_description"]]
y_true, y_pred = test["judgment_label"], [p.label for p in preds]
conf = np.array([p.confidence for p in preds])

reference = {
    "model_version": clf.model_version,
    "n": len(preds),
    "labels": clf.labels,
    "label_mix": {l: y_pred.count(l) / len(y_pred) for l in clf.labels},
    "confidence_hist": confidence_hist(conf).tolist(),
    "test_accuracy": accuracy_score(y_true, y_pred),
    "test_f1_macro": f1_score(y_true, y_pred, average="macro"),
    "latency_p50_ms": float(np.median([p.latency_ms for p in preds])),
}
Path("reference.json").write_text(json.dumps(reference, indent=2))

random.seed(7)
by_label = {l: [r for r in test if r["judgment_label"] == l] for l in clf.labels}
examples = [
    {"text": r["decision_description"], "true_label": l}
    for l in clf.labels
    for r in random.sample(by_label[l], k=min(3, len(by_label[l])))
]
Path("examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in reference.items() if k != "confidence_hist"}, indent=2))
