import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
from core.metrics import confidence_hist, psi, snapshot
from core.model import Prediction
from core.store import Store

LABELS = ["no", "partial", "yes"]


class FakeClassifier:
    model_version = "fake@0000000"
    labels = LABELS

    def predict(self, text: str) -> Prediction:
        label = "yes" if "PROVIDO" in text.upper() else "no"
        scores = {l: (0.8 if l == label else 0.1) for l in LABELS}
        return Prediction(label, 0.8, scores, 1.5, self.model_version)


REFERENCE = {
    "labels": LABELS,
    "label_mix": {"no": 0.5, "partial": 0.1, "yes": 0.4},
    "confidence_hist": confidence_hist(np.full(100, 0.85)).tolist(),
}


@pytest.fixture
def client():
    store = Store(":memory:")
    api.app.dependency_overrides = {
        api.get_classifier: lambda: FakeClassifier(),
        api.get_store: lambda: store,
        api.get_reference: lambda: REFERENCE,
    }
    yield TestClient(api.app)
    api.app.dependency_overrides = {}


TEXT = "APELAÇÃO CÍVEL. MANDADO DE SEGURANÇA. SERVIDOR PÚBLICO. SUPRESSÃO DE GRATIFICAÇÃO."


def test_predict_logs_prediction(client):
    r = client.post("/predict", json={"text": TEXT})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction_id"] == 1 and body["label"] in LABELS
    assert client.get("/metrics").json()["n_predictions"] == 1


def test_predict_rejects_short_text(client):
    assert client.post("/predict", json={"text": "curto"}).status_code == 422


def test_feedback_flow_and_export(client):
    pid = client.post("/predict", json={"text": TEXT}).json()["prediction_id"]
    r = client.post("/feedback", json={"prediction_id": pid, "is_correct": False, "corrected_label": "partial"})
    assert r.status_code == 204
    m = client.get("/metrics").json()
    assert m["n_reviewed"] == 1 and m["online_accuracy"] == 0.0
    rows = [json.loads(l) for l in client.get("/export").text.strip().splitlines()]
    assert rows == [{"text": TEXT, "gold_label": "partial", "label": "no", "model_version": "fake@0000000"}]


def test_feedback_is_upserted(client):
    pid = client.post("/predict", json={"text": TEXT}).json()["prediction_id"]
    client.post("/feedback", json={"prediction_id": pid, "is_correct": False, "corrected_label": "yes"})
    client.post("/feedback", json={"prediction_id": pid, "is_correct": True})
    m = client.get("/metrics").json()
    assert m["n_reviewed"] == 1 and m["online_accuracy"] == 1.0


def test_feedback_validation(client):
    pid = client.post("/predict", json={"text": TEXT}).json()["prediction_id"]
    assert client.post("/feedback", json={"prediction_id": pid, "is_correct": False}).status_code == 422
    bad = {"prediction_id": pid, "is_correct": False, "corrected_label": "maybe"}
    assert client.post("/feedback", json=bad).status_code == 422
    assert client.post("/feedback", json={"prediction_id": 999, "is_correct": True}).status_code == 404


def test_psi_properties():
    a = np.array([0.2, 0.3, 0.5])
    assert psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert psi(a, np.array([0.6, 0.3, 0.1])) > 0.25


def test_drift_detected_when_label_mix_shifts():
    store = Store(":memory:")
    clf = FakeClassifier()
    for _ in range(30):  # all "no" vs reference 50% "no"
        store.log_prediction(TEXT, clf.predict("negado"))
    snap = snapshot(store.frame(), REFERENCE, min_n=20)
    assert snap.psi_label_mix > 0.25 and snap.drift == "significant"


def test_snapshot_needs_minimum_traffic():
    store = Store(":memory:")
    store.log_prediction(TEXT, FakeClassifier().predict(TEXT))
    assert snapshot(store.frame(), REFERENCE).drift == "insufficient data"


def test_head_tail_truncation_keeps_both_ends():
    from core.model import head_tail_ids

    ids = list(range(1000))
    out = head_tail_ids(ids, max_length=512, head=128)
    assert len(out) == 510
    assert out[:128] == ids[:128] and out[-382:] == ids[-382:]
    assert head_tail_ids(ids[:100]) == ids[:100]
