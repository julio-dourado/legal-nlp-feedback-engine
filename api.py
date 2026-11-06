"""FastAPI service: inference, human feedback and observability endpoints.

Run:  uvicorn api:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, model_validator

from core.metrics import load_reference, snapshot
from core.model import Classifier, HFClassifier
from core.store import Store

app = FastAPI(
    title="Legal NLP Feedback Engine",
    description="Classificação de decisões judiciais (PT-BR) com human-in-the-loop e observabilidade.",
    version="1.0.0",
)


@lru_cache
def get_classifier() -> Classifier:
    return HFClassifier()


@lru_cache
def get_store() -> Store:
    return Store(os.getenv("DB_PATH", "data/feedback.db"))


@lru_cache
def get_reference() -> dict | None:
    return load_reference(os.getenv("REFERENCE_PATH", "reference.json"))


class PredictIn(BaseModel):
    text: str = Field(min_length=20, max_length=20_000, description="Descrição da decisão / ementa")


class PredictOut(BaseModel):
    prediction_id: int
    label: str
    confidence: float
    scores: dict[str, float]
    latency_ms: float
    model_version: str


class FeedbackIn(BaseModel):
    prediction_id: int
    is_correct: bool
    corrected_label: str | None = None
    comment: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def _needs_correction(self):
        if not self.is_correct and not self.corrected_label:
            raise ValueError("corrected_label is required when is_correct is false")
        return self


@app.get("/health")
def health(clf: Classifier = Depends(get_classifier)) -> dict:
    return {"status": "ok", "model_version": clf.model_version, "labels": clf.labels}


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn, clf: Classifier = Depends(get_classifier), store: Store = Depends(get_store)):
    pred = clf.predict(body.text)
    pid = store.log_prediction(body.text, pred)
    return PredictOut(prediction_id=pid, **pred.__dict__)


@app.post("/feedback", status_code=204)
def feedback(body: FeedbackIn, clf: Classifier = Depends(get_classifier), store: Store = Depends(get_store)):
    if body.corrected_label and body.corrected_label not in clf.labels:
        raise HTTPException(422, f"corrected_label must be one of {clf.labels}")
    try:
        store.log_feedback(body.prediction_id, body.is_correct, body.corrected_label, body.comment)
    except KeyError:
        raise HTTPException(404, "prediction not found") from None


@app.get("/metrics")
def metrics(store: Store = Depends(get_store), ref: dict | None = Depends(get_reference)) -> dict:
    return snapshot(store.frame(), ref).to_dict()


@app.get("/export", response_class=PlainTextResponse)
def export(store: Store = Depends(get_store)) -> str:
    """Human-validated examples as JSONL, ready for the next fine-tuning round."""
    return store.training_export().to_json(orient="records", lines=True, force_ascii=False)
