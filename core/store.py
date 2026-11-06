"""SQLite persistence for predictions and human feedback."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.model import Prediction

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    text          TEXT    NOT NULL,
    text_chars    INTEGER NOT NULL,
    label         TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    scores        TEXT    NOT NULL,
    latency_ms    REAL    NOT NULL,
    model_version TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id   INTEGER NOT NULL UNIQUE REFERENCES predictions(id),
    created_at      TEXT    NOT NULL,
    is_correct      INTEGER NOT NULL,
    corrected_label TEXT,
    comment         TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = "data/feedback.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self):
        with self._conn:
            yield self._conn

    def log_prediction(self, text: str, pred: Prediction) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO predictions (created_at, text, text_chars, label, confidence, scores,"
                " latency_ms, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(), text, len(text), pred.label, pred.confidence, json.dumps(pred.scores),
                 pred.latency_ms, pred.model_version),
            )
            return int(cur.lastrowid)

    def get_prediction(self, prediction_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()

    def log_feedback(
        self, prediction_id: int, is_correct: bool, corrected_label: str | None = None,
        comment: str | None = None,
    ) -> None:
        """Upsert: a reviewer may change their mind; the latest judgement wins."""
        if self.get_prediction(prediction_id) is None:
            raise KeyError(prediction_id)
        with self._tx() as c:
            c.execute(
                "INSERT INTO feedback (prediction_id, created_at, is_correct, corrected_label, comment)"
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT(prediction_id) DO UPDATE SET"
                " created_at = excluded.created_at, is_correct = excluded.is_correct,"
                " corrected_label = excluded.corrected_label, comment = excluded.comment",
                (prediction_id, _now(), int(is_correct), None if is_correct else corrected_label, comment),
            )

    def frame(self) -> pd.DataFrame:
        """Predictions left-joined with feedback, oldest first."""
        return pd.read_sql_query(
            "SELECT p.*, f.is_correct, f.corrected_label, f.comment, f.created_at AS feedback_at"
            " FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id ORDER BY p.id",
            self._conn,
        )

    def training_export(self) -> pd.DataFrame:
        """Human-validated examples, ready to be appended to the next fine-tuning run."""
        df = self.frame().dropna(subset=["is_correct"])
        df["gold_label"] = df.apply(
            lambda r: r["label"] if r["is_correct"] == 1 else r["corrected_label"], axis=1
        )
        return df.dropna(subset=["gold_label"])[["text", "gold_label", "label", "model_version"]]
