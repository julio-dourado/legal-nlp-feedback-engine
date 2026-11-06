"""Online observability metrics computed from the prediction/feedback log.

* Online accuracy: share of human-reviewed predictions marked correct.
* Latency: p50 / p95 in milliseconds.
* Drift: Population Stability Index (PSI) between the offline reference
  distribution (model on the held-out test set, see `reference.json`) and live traffic,
  for both the confidence score and the predicted-label mix.
  Rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONF_BINS = np.linspace(0, 1, 11)
EPS = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def confidence_hist(conf: pd.Series | np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(conf, float), bins=CONF_BINS)
    return counts / max(counts.sum(), 1)


def drift_status(value: float | None) -> str:
    if value is None:
        return "insufficient data"
    return "stable" if value < 0.1 else "moderate" if value < 0.25 else "significant"


@dataclass
class Snapshot:
    n_predictions: int
    n_reviewed: int
    online_accuracy: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    psi_confidence: float | None
    psi_label_mix: float | None
    drift: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_reference(path: str | Path = "reference.json") -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def snapshot(df: pd.DataFrame, reference: dict | None, window: int = 200, min_n: int = 20) -> Snapshot:
    reviewed = df.dropna(subset=["is_correct"]) if "is_correct" in df else df.iloc[0:0]
    recent = df.tail(window)
    psi_conf = psi_mix = None
    if reference and len(recent) >= min_n:
        psi_conf = psi(reference["confidence_hist"], confidence_hist(recent["confidence"]))
        labels = reference["labels"]
        live_mix = recent["label"].value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()
        psi_mix = psi([reference["label_mix"][l] for l in labels], live_mix)
    worst = max((v for v in (psi_conf, psi_mix) if v is not None), default=None)
    return Snapshot(
        n_predictions=len(df),
        n_reviewed=len(reviewed),
        online_accuracy=float(reviewed["is_correct"].mean()) if len(reviewed) else None,
        latency_p50_ms=float(df["latency_ms"].quantile(0.5)) if len(df) else None,
        latency_p95_ms=float(df["latency_ms"].quantile(0.95)) if len(df) else None,
        psi_confidence=psi_conf,
        psi_label_mix=psi_mix,
        drift=drift_status(worst),
    )
